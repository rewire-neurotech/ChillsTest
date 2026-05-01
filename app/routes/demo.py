import uuid
import time
import json
import csv
import io
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from pydub import AudioSegment

from app.core.config import cfg
from app.db import get_db, SessionLocal
from app.models import DemoSession
from app.services.prompt import build_user_prompt
from app.services.llm import generate_speech
from app.services.tts import synth
from app.services.mix import mix as mix_audio
from app.utils.audio import load_audio, duration_ms, content_duration_ms

r = APIRouter(prefix="/api/demo", tags=["demo"])

MUSIC_FADEIN_MS = 10     # Joaquin's new mix: near-zero fade-in, music enters immediately
TAIL_BUFFER_MS = 4000    # Music plays 4 seconds after voice ends


# --- In-memory job tracking for progress ---

_jobs = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# --- Request / Response schemas ---

class GenerateRequest(BaseModel):
    q1_wound: str
    q2_chills_trigger: str
    q3_hidden_truth: str


class GenerateJobResponse(BaseModel):
    job_id: str


class GenerateStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: int
    session_id: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None


class DemographicsRequest(BaseModel):
    session_id: str
    age: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str

    # --- Original fields (kept for backward compat) ---
    felt_chills: Optional[bool] = None
    chills_count: Optional[int] = 0
    chills_timestamps_json: Optional[str] = None
    experience_driver: Optional[str] = None
    feedback_note: Optional[str] = None

    # --- Universal questions ---
    crying_response: Optional[str] = None
    eyes_open_closed: Optional[str] = None
    inspired_to_do: Optional[str] = None

    # --- Chills YES branch ---
    chills_intensity: Optional[int] = None
    chills_trigger_json: Optional[str] = None
    chills_trigger_other: Optional[str] = None
    chills_body_location_json: Optional[str] = None
    chills_peak_timing: Optional[str] = None
    chills_reflection: Optional[str] = None

    # --- Chills NO branch ---
    no_chills_barriers_json: Optional[str] = None
    no_chills_closeness: Optional[str] = None
    no_chills_emotional_shift: Optional[bool] = None
    no_chills_emotional_describe: Optional[str] = None

    # --- EBI (1-7 Likert) ---
    ebi_faced_difficult: Optional[int] = None
    ebi_resolution: Optional[int] = None
    ebi_explore: Optional[int] = None
    ebi_breakthrough: Optional[int] = None
    ebi_closure: Optional[int] = None
    ebi_release: Optional[int] = None
    ebi_stuck_resisting: Optional[int] = None
    ebi_stuck_throughout: Optional[int] = None

    # --- Content quality ---
    content_relevance: Optional[int] = None
    content_inauthentic: Optional[bool] = None
    content_inauthentic_detail: Optional[str] = None
    content_music_match: Optional[str] = None
    content_pacing: Optional[str] = None

    # --- Open-ended ---
    open_improve: Optional[str] = None
    open_final_thoughts: Optional[str] = None

    # --- Optional ---
    prolific_id: Optional[str] = None
    email: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


# --- Helper functions ---

def _word_count(txt: str) -> int:
    return len((txt or "").strip().split())


def _within(ms: int, target: int, tol: float = 0.04) -> bool:
    return abs(ms - target) <= int(target * tol)


# --- Background generation pipeline ---

def _run_generate(job_id: str, q1: str, q2: str, q3: str):
    """Run the full generation pipeline in a background thread."""
    try:
        start = time.time()
        session_id = job_id  # reuse job_id as session_id

        _update_job(job_id, stage="writing", progress=5)

        # 1. Get music content duration and calculate target words
        music_path = cfg.MUSIC_FILE
        if not music_path.exists():
            _update_job(job_id, stage="error", error="Music file not found in assets")
            return

        music_audio = load_audio(str(music_path))
        music_ms = content_duration_ms(music_audio)
        # Voice fills most of the music duration, leaving tail buffer at the end
        spoken_target_ms = max(int(music_ms - TAIL_BUFFER_MS), int(0.75 * music_ms))
        # ElevenLabs v3 at style=1.0 speaks at ~2.0 words per second
        target_words = min(int((spoken_target_ms / 1000) * 2.0), 1200)

        print(f"[Demo] Music content: {music_ms}ms, spoken target: {spoken_target_ms}ms, target words: {target_words}")

        # 2. Build prompt with target word count (AI selects format)
        user_prompt = build_user_prompt(
            q1_wound=q1,
            q2_chills_trigger=q2,
            q3_hidden_truth=q3,
            target_words=target_words,
        )
        speech_format = "AI_SELECTED"

        _update_job(job_id, stage="writing", progress=10)

        # 3. Generate speech text via Claude
        print(f"[Demo] Generating speech for session {session_id}, target_words={target_words}")
        speech_text = generate_speech(user_prompt)
        print(f"[Demo] Speech generated, {_word_count(speech_text)} words")

        _update_job(job_id, stage="synthesizing", progress=35)

        # 4. TTS via ElevenLabs
        print(f"[Demo] Synthesizing TTS...")
        voice_id = cfg.ELEVENLABS_VOICE_ID
        voice_wav_path = synth(
            text=speech_text,
            voice_id=voice_id,
            key=cfg.ELEVENLABS_API_KEY,
        )
        print(f"[Demo] TTS done: {voice_wav_path}")

        _update_job(job_id, stage="refining", progress=60)

        # 5. Check duration and correct if needed (up to 3 attempts)
        tts_ms = duration_ms(load_audio(voice_wav_path))
        ema_wps = 2.0
        best_script = speech_text
        best_wav = voice_wav_path

        print(f"[Demo] TTS duration: {tts_ms}ms (target: {spoken_target_ms}ms)")

        for attempt in range(3):
            wc = _word_count(best_script)
            observed_wps = wc / max(1.0, tts_ms / 1000.0)
            ema_wps = 0.7 * ema_wps + 0.3 * observed_wps

            if _within(tts_ms, spoken_target_ms):
                print(f"[Demo] Duration within tolerance, done.")
                break

            delta_ms = spoken_target_ms - tts_ms
            delta_words = int(abs(delta_ms) / 1000.0 * ema_wps)
            delta_words = max(30, min(delta_words, 200))

            if delta_ms > 0:
                # Too short -- generate more
                print(f"[Demo] Speech too short by {delta_ms}ms, extending by ~{delta_words} words")
                tail = " ".join(best_script.strip().split()[-40:])
                extend_prompt = f"""Continue this speech naturally. Write approximately {delta_words} more words. Maintain the same tone, style and emotional arc. Do not repeat what was already said. Pick up exactly where this left off:

...{tail}

Continue now. Only the continuation text. No preamble."""
                more = generate_speech(extend_prompt)
                if more and more not in best_script:
                    best_script = (best_script + " " + more).strip()
            else:
                # Too long -- trim from the end at a sentence boundary
                print(f"[Demo] Speech too long by {abs(delta_ms)}ms, trimming ~{delta_words} words")
                words = best_script.strip().split()
                trimmed = " ".join(words[:-delta_words])
                # Find last sentence boundary to avoid cutting mid-sentence
                last_end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
                if last_end > len(trimmed) * 0.5:
                    trimmed = trimmed[: last_end + 1]
                best_script = trimmed

            # Re-synthesize
            best_wav = synth(
                text=best_script,
                voice_id=voice_id,
                key=cfg.ELEVENLABS_API_KEY,
            )
            tts_ms = duration_ms(load_audio(best_wav))
            print(f"[Demo] Correction attempt {attempt + 1}: {tts_ms}ms (target: {spoken_target_ms}ms)")

            _update_job(job_id, progress=60 + (attempt + 1) * 5)

        speech_text = best_script
        voice_wav_path = best_wav

        # --- Hard-cap: voice MUST NOT exceed spoken_target_ms ---
        # The correction loop has 4% tolerance which can leave voice too long.
        # If voice exceeds the target, the music gets stretched past its content
        # and you hear voice playing over silence at the end.
        final_tts_ms = duration_ms(load_audio(voice_wav_path))
        if final_tts_ms > spoken_target_ms:
            print(f"[Demo] Voice {final_tts_ms}ms exceeds spoken target {spoken_target_ms}ms, trimming audio to fit")
            vc = AudioSegment.from_file(voice_wav_path)
            vc = vc[:spoken_target_ms].fade_out(800)
            capped_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            vc.export(capped_wav.name, format="wav")
            voice_wav_path = capped_wav.name
            print(f"[Demo] Voice hard-capped to {spoken_target_ms}ms")

        _update_job(job_id, stage="mixing", progress=78)

        # 6. Pad silence at end of voice (music plays TAIL_BUFFER_MS after speech stops)
        voice_audio = AudioSegment.from_file(voice_wav_path)
        tail_silence = AudioSegment.silent(duration=TAIL_BUFFER_MS, frame_rate=voice_audio.frame_rate)
        padded = voice_audio + tail_silence

        # --- Hard-cap: padded voice must not exceed music content duration ---
        # This ensures music retiming is ~1:1, never stretched past its content.
        padded_total_ms = duration_ms(padded)
        if padded_total_ms > music_ms:
            print(f"[Demo] Padded voice {padded_total_ms}ms exceeds music content {music_ms}ms, capping to {music_ms}ms")
            padded = padded[:music_ms]

        padded_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        padded.export(padded_wav.name, format="wav")
        voice_wav_path = padded_wav.name

        print(f"[Demo] Final voice file: {duration_ms(padded)}ms (music content: {music_ms}ms)")

        # Save raw inputs for DSP analysis
        raw_voice_name = f"{session_id}_voice_raw.wav"
        raw_music_name = f"{session_id}_music_raw.mp3"
        shutil.copy2(voice_wav_path, str(cfg.out_dir_path / raw_voice_name))
        shutil.copy2(str(music_path), str(cfg.out_dir_path / raw_music_name))
        print(f"[Demo] Saved raw inputs: {raw_voice_name}, {raw_music_name}")

        # 7. Mix with music (Joaquin's new DSP pipeline)
        print(f"[Demo] Mixing with music...")
        audio_filename = f"{session_id}.mp3"
        out_path = cfg.out_dir_path / audio_filename

        _update_job(job_id, stage="mixing", progress=82)

        mix_audio(
            voice_path=voice_wav_path,
            music_path=str(music_path),
            out_path=str(out_path),
            sync_mode="retime_music_to_voice",
            music_fadein_ms=MUSIC_FADEIN_MS,
            music_premix_gain_db=-1.5,
            ffmpeg_bin=cfg.FFMPEG_BIN,
        )
        print(f"[Demo] Mix done: {out_path}")

        elapsed = round(time.time() - start, 2)

        _update_job(job_id, stage="saving", progress=92)

        # 8. Save session to DB (manual session since we're in a thread)
        db = SessionLocal()
        try:
            session = DemoSession(
                session_id=session_id,
                q1_wound=q1,
                q2_chills_trigger=q2,
                q3_hidden_truth=q3,
                speech_format=speech_format,
                speech_text=speech_text,
                voice_id=voice_id,
                music_track="heroes_wwii.mp3",
                audio_filename=audio_filename,
                generation_time_seconds=elapsed,
            )
            db.add(session)
            db.commit()
        finally:
            db.close()

        # Build audio URL
        base = cfg.PUBLIC_BASE_URL.rstrip("/") if cfg.PUBLIC_BASE_URL else ""
        audio_url = f"{base}/api/demo/audio/{audio_filename}"

        print(f"[Demo] Session {session_id} complete in {elapsed}s")

        _update_job(
            job_id,
            stage="done",
            progress=100,
            session_id=session_id,
            audio_url=audio_url,
        )

    except Exception as e:
        print(f"[Demo] Generation error for job {job_id}: {e}")
        _update_job(job_id, stage="error", error=str(e))


# --- Endpoints ---

@r.post("/generate", response_model=GenerateJobResponse)
def generate(req: GenerateRequest):
    """
    Start generation in background. Returns a job_id immediately.
    Poll /generate/status/{job_id} for progress.
    """
    job_id = uuid.uuid4().hex[:16]

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "stage": "queued",
            "progress": 0,
            "session_id": None,
            "audio_url": None,
            "error": None,
        }

    t = threading.Thread(
        target=_run_generate,
        args=(job_id, req.q1_wound, req.q2_chills_trigger, req.q3_hidden_truth),
        daemon=True,
    )
    t.start()

    return GenerateJobResponse(job_id=job_id)


@r.get("/generate/status/{job_id}", response_model=GenerateStatusResponse)
def generate_status(job_id: str):
    """Poll this endpoint for generation progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return GenerateStatusResponse(**job)


@r.post("/demographics", response_model=FeedbackResponse)
def save_demographics(req: DemographicsRequest, db: Session = Depends(get_db)):
    """Save demographics collected during the wait."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.age is not None:
        session.age = req.age
    if req.gender is not None:
        session.gender = req.gender
    if req.ethnicity is not None:
        session.ethnicity = req.ethnicity

    db.commit()
    return FeedbackResponse(status="ok")


@r.post("/feedback", response_model=FeedbackResponse)
def save_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Save the full post-experience survey."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Original fields
    session.felt_chills = req.felt_chills
    session.chills_count = req.chills_count or 0
    session.chills_timestamps_json = req.chills_timestamps_json
    session.experience_driver = req.experience_driver
    session.feedback_note = req.feedback_note

    # Universal questions
    session.crying_response = req.crying_response
    session.eyes_open_closed = req.eyes_open_closed
    session.inspired_to_do = req.inspired_to_do

    # Chills YES branch
    session.chills_intensity = req.chills_intensity
    session.chills_trigger_json = req.chills_trigger_json
    session.chills_trigger_other = req.chills_trigger_other
    session.chills_body_location_json = req.chills_body_location_json
    session.chills_peak_timing = req.chills_peak_timing
    session.chills_reflection = req.chills_reflection

    # Chills NO branch
    session.no_chills_barriers_json = req.no_chills_barriers_json
    session.no_chills_closeness = req.no_chills_closeness
    session.no_chills_emotional_shift = req.no_chills_emotional_shift
    session.no_chills_emotional_describe = req.no_chills_emotional_describe

    # EBI
    session.ebi_faced_difficult = req.ebi_faced_difficult
    session.ebi_resolution = req.ebi_resolution
    session.ebi_explore = req.ebi_explore
    session.ebi_breakthrough = req.ebi_breakthrough
    session.ebi_closure = req.ebi_closure
    session.ebi_release = req.ebi_release
    session.ebi_stuck_resisting = req.ebi_stuck_resisting
    session.ebi_stuck_throughout = req.ebi_stuck_throughout

    # Content quality
    session.content_relevance = req.content_relevance
    session.content_inauthentic = req.content_inauthentic
    session.content_inauthentic_detail = req.content_inauthentic_detail
    session.content_music_match = req.content_music_match
    session.content_pacing = req.content_pacing

    # Open-ended
    session.open_improve = req.open_improve
    session.open_final_thoughts = req.open_final_thoughts

    # Optional
    session.prolific_id = req.prolific_id
    session.email = req.email

    session.completed_at = datetime.now(timezone.utc)

    db.commit()
    return FeedbackResponse(status="ok")


@r.get("/audio/{filename}")
def serve_audio(filename: str):
    """Serve a generated audio file."""
    filepath = cfg.out_dir_path / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    return StreamingResponse(
        open(filepath, "rb"),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


@r.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    """Export all demo sessions as CSV for Felix."""
    sessions = db.query(DemoSession).order_by(DemoSession.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "session_id",
        "q1_wound",
        "q2_chills_trigger",
        "q3_hidden_truth",
        "age",
        "gender",
        "ethnicity",
        "speech_format",
        "speech_text",
        "voice_id",
        "music_track",
        "audio_filename",
        # Original feedback
        "felt_chills",
        "chills_count",
        "chills_timestamps",
        "experience_driver",
        "feedback_note",
        # Universal
        "crying_response",
        "eyes_open_closed",
        "inspired_to_do",
        # Chills YES
        "chills_intensity",
        "chills_trigger",
        "chills_trigger_other",
        "chills_body_location",
        "chills_peak_timing",
        "chills_reflection",
        # Chills NO
        "no_chills_barriers",
        "no_chills_closeness",
        "no_chills_emotional_shift",
        "no_chills_emotional_describe",
        # EBI
        "ebi_faced_difficult",
        "ebi_resolution",
        "ebi_explore",
        "ebi_breakthrough",
        "ebi_closure",
        "ebi_release",
        "ebi_stuck_resisting",
        "ebi_stuck_throughout",
        # Content quality
        "content_relevance",
        "content_inauthentic",
        "content_inauthentic_detail",
        "content_music_match",
        "content_pacing",
        # Open-ended
        "open_improve",
        "open_final_thoughts",
        # Optional
        "prolific_id",
        "email",
        # Timing
        "generation_time_seconds",
        "created_at",
        "completed_at",
    ])

    for s in sessions:
        writer.writerow([
            s.session_id,
            s.q1_wound,
            s.q2_chills_trigger,
            s.q3_hidden_truth,
            s.age,
            s.gender,
            s.ethnicity,
            s.speech_format,
            s.speech_text,
            s.voice_id,
            s.music_track,
            s.audio_filename,
            # Original feedback
            s.felt_chills,
            s.chills_count,
            s.chills_timestamps_json,
            s.experience_driver,
            s.feedback_note,
            # Universal
            s.crying_response,
            s.eyes_open_closed,
            s.inspired_to_do,
            # Chills YES
            s.chills_intensity,
            s.chills_trigger_json,
            s.chills_trigger_other,
            s.chills_body_location_json,
            s.chills_peak_timing,
            s.chills_reflection,
            # Chills NO
            s.no_chills_barriers_json,
            s.no_chills_closeness,
            s.no_chills_emotional_shift,
            s.no_chills_emotional_describe,
            # EBI
            s.ebi_faced_difficult,
            s.ebi_resolution,
            s.ebi_explore,
            s.ebi_breakthrough,
            s.ebi_closure,
            s.ebi_release,
            s.ebi_stuck_resisting,
            s.ebi_stuck_throughout,
            # Content quality
            s.content_relevance,
            s.content_inauthentic,
            s.content_inauthentic_detail,
            s.content_music_match,
            s.content_pacing,
            # Open-ended
            s.open_improve,
            s.open_final_thoughts,
            # Optional
            s.prolific_id,
            s.email,
            # Timing
            s.generation_time_seconds,
            s.created_at,
            s.completed_at,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rewire_demo_sessions.csv"},
    )
