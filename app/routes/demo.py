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
from app.services.music_selector import select_track
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
    q1_low_voice: str
    q2_chills: str
    q3_first_call: str
    q4_unseen: str = ""


class GenerateJobResponse(BaseModel):
    job_id: str
    meditation_url: str


class GenerateStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: int
    session_id: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    chills_count: Optional[int] = 0
    chills_timestamps_json: Optional[str] = None
    feedback_text: Optional[str] = None


class EmailRequest(BaseModel):
    session_id: str
    email: str


class StatusResponse(BaseModel):
    status: str


# --- Helper functions ---

def _word_count(txt: str) -> int:
    return len((txt or "").strip().split())


def _within(ms: int, target: int, tol: float = 0.04) -> bool:
    return abs(ms - target) <= int(target * tol)


def _trim_at_boundary(text: str, words_to_cut: int) -> str:
    """Trim words from the end, cutting at a clean boundary.
    Prefers --- section breaks > paragraph ends > sentence ends."""
    words = text.strip().split()
    if words_to_cut >= len(words):
        return text
    rough_cut = " ".join(words[:-words_to_cut])

    # 1. Try to find a --- section break in the last 40% of the rough cut
    search_zone_start = int(len(rough_cut) * 0.6)
    last_section = rough_cut.rfind("\n---", search_zone_start)
    if last_section == -1:
        last_section = rough_cut.rfind("---", search_zone_start)
    if last_section > search_zone_start:
        return rough_cut[:last_section].rstrip()

    # 2. Try to find a double newline (paragraph break)
    last_para = rough_cut.rfind("\n\n", search_zone_start)
    if last_para > search_zone_start:
        return rough_cut[:last_para].rstrip()

    # 3. Fall back to last sentence boundary
    last_end = max(
        rough_cut.rfind(". ", search_zone_start),
        rough_cut.rfind("! ", search_zone_start),
        rough_cut.rfind("? ", search_zone_start),
        rough_cut.rfind(".\n", search_zone_start),
        rough_cut.rfind("!\n", search_zone_start),
        rough_cut.rfind("?\n", search_zone_start),
    )
    if last_end > search_zone_start:
        return rough_cut[:last_end + 1].rstrip()

    # 4. Try sentence boundary anywhere in the back half
    last_end = max(rough_cut.rfind("."), rough_cut.rfind("!"), rough_cut.rfind("?"))
    if last_end > len(rough_cut) * 0.5:
        return rough_cut[:last_end + 1].rstrip()

    return rough_cut


# --- Background generation pipeline ---

def _run_generate(job_id: str, q1: str, q2: str, q3: str, q4: str, track_name: str):
    """Run the full generation pipeline in a background thread."""
    try:
        start = time.time()
        session_id = job_id  # reuse job_id as session_id

        _update_job(job_id, stage="writing", progress=5)

        # 0. Get track info (track already selected in /generate endpoint)
        track = cfg.get_track(track_name)
        print(f"[Demo] Using track: {track['name']} ({track['description']})")

        # 1. Get music duration info
        music_path = track["file"]
        if not music_path.exists():
            _update_job(job_id, stage="error", error="Music file not found in assets")
            return

        music_audio = load_audio(str(music_path))
        full_music_ms = duration_ms(music_audio)
        content_ms = content_duration_ms(music_audio)

        # Voice should end before the music content ends
        spoken_target_ms = max(int(content_ms - TAIL_BUFFER_MS), int(0.75 * content_ms))
        # ElevenLabs v3 at style=1.0 speaks at ~2.0 words per second
        target_words = min(int((spoken_target_ms / 1000) * 2.0), 1200)

        print(f"[Demo] Music file: {full_music_ms}ms, content: {content_ms}ms, spoken target: {spoken_target_ms}ms, target words: {target_words}")

        # 2. Build prompt with target word count (AI selects format)
        user_prompt = build_user_prompt(
            q1_low_voice=q1,
            q2_chills=q2,
            q3_first_call=q3,
            q4_unseen=q4,
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
        voice_id = track["voice_id"]
        voice_wav_path = synth(
            text=speech_text,
            voice_id=voice_id,
            key=cfg.ELEVENLABS_API_KEY,
        )
        print(f"[Demo] TTS done: {voice_wav_path}")

        _update_job(job_id, stage="refining", progress=60)

        # 5. Check duration and correct if needed (up to 5 attempts)
        tts_ms = duration_ms(load_audio(voice_wav_path))
        ema_wps = 2.0
        best_script = speech_text
        best_wav = voice_wav_path

        print(f"[Demo] TTS duration: {tts_ms}ms (target: {spoken_target_ms}ms)")

        for attempt in range(5):
            wc = _word_count(best_script)
            observed_wps = wc / max(1.0, tts_ms / 1000.0)
            ema_wps = 0.7 * ema_wps + 0.3 * observed_wps

            if _within(tts_ms, spoken_target_ms):
                print(f"[Demo] Duration within tolerance, done.")
                break

            delta_ms = spoken_target_ms - tts_ms
            # Dampen corrections: each attempt is gentler to avoid oscillation
            damping = 0.7 ** attempt
            delta_words = int(abs(delta_ms) / 1000.0 * ema_wps * damping)
            delta_words = max(10, min(delta_words, 200))

            if delta_ms > 0:
                # Too short -- generate more
                print(f"[Demo] Speech too short by {delta_ms}ms, extending by ~{delta_words} words (damping={damping:.2f})")
                tail = " ".join(best_script.strip().split()[-40:])
                extend_prompt = f"""Continue this speech naturally. Write approximately {delta_words} more words. Maintain the same tone, style and emotional arc. Do not repeat what was already said. Pick up exactly where this left off:

...{tail}

Continue now. Only the continuation text. No preamble."""
                more = generate_speech(extend_prompt)
                if more and more not in best_script:
                    best_script = (best_script + " " + more).strip()
            else:
                # Too long -- trim at a clean boundary
                print(f"[Demo] Speech too long by {abs(delta_ms)}ms, trimming ~{delta_words} words (damping={damping:.2f})")
                best_script = _trim_at_boundary(best_script, delta_words)

            # Re-synthesize
            best_wav = synth(
                text=best_script,
                voice_id=voice_id,
                key=cfg.ELEVENLABS_API_KEY,
            )
            tts_ms = duration_ms(load_audio(best_wav))
            print(f"[Demo] Correction attempt {attempt + 1}: {tts_ms}ms (target: {spoken_target_ms}ms)")

            _update_job(job_id, progress=60 + (attempt + 1) * 4)

        speech_text = best_script
        voice_wav_path = best_wav

        # --- Hard-cap: voice MUST NOT exceed spoken_target_ms ---
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

        # 6. Pad silence so voice file matches content_ms exactly.
        #    This ensures retime_music_to_voice has a ~1:1 ratio
        #    and the music plays at its natural speed.
        voice_audio = AudioSegment.from_file(voice_wav_path)
        voice_len = duration_ms(voice_audio)
        pad_needed = max(0, content_ms - voice_len)
        if pad_needed > 0:
            tail_silence = AudioSegment.silent(duration=pad_needed, frame_rate=voice_audio.frame_rate)
            padded = voice_audio + tail_silence
        else:
            padded = voice_audio[:content_ms]

        padded_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        padded.export(padded_wav.name, format="wav")
        voice_wav_path = padded_wav.name

        print(f"[Demo] Voice speech: {voice_len}ms, padded to: {duration_ms(padded)}ms (music content: {content_ms}ms)")

        # 7. Trim music file to content duration only.
        #    The raw file may have trailing silence. If we pass the full file
        #    to mix, retime_music_to_voice compresses it (e.g. 360s -> 330s = 1.09x),
        #    making the actual music end BEFORE the voice does.
        #    By trimming first, the retiming factor is ~1:1 and music plays at natural speed.
        music_trimmed = music_audio[:content_ms]
        trimmed_music_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        music_trimmed.export(trimmed_music_tmp.name, format="mp3", bitrate="256k")
        music_mix_path = trimmed_music_tmp.name

        print(f"[Demo] Music trimmed from {full_music_ms}ms to {content_ms}ms for mix")

        # Save raw inputs for DSP analysis (original untrimmed)
        # The voice-only file is also accessible to Felix via /api/demo/audio/
        raw_voice_name = f"{session_id}_voice_raw.wav"
        raw_music_name = f"{session_id}_music_raw.mp3"
        shutil.copy2(voice_wav_path, str(cfg.out_dir_path / raw_voice_name))
        shutil.copy2(str(music_path), str(cfg.out_dir_path / raw_music_name))
        print(f"[Demo] Saved raw inputs: {raw_voice_name}, {raw_music_name}")

        # 8. Mix with music (Joaquin's DSP pipeline, untouched)
        print(f"[Demo] Mixing with music...")
        audio_filename = f"{session_id}.mp3"
        out_path = cfg.out_dir_path / audio_filename

        _update_job(job_id, stage="mixing", progress=82)

        mix_audio(
            voice_path=voice_wav_path,
            music_path=music_mix_path,
            out_path=str(out_path),
            sync_mode="retime_music_to_voice",
            music_fadein_ms=MUSIC_FADEIN_MS,
            music_premix_gain_db=-5.0,
            ffmpeg_bin=cfg.FFMPEG_BIN,
        )
        print(f"[Demo] Mix done: {out_path}")

        elapsed = round(time.time() - start, 2)

        _update_job(job_id, stage="saving", progress=92)

        # 9. Save session to DB (manual session since we're in a thread)
        db = SessionLocal()
        try:
            session = DemoSession(
                session_id=session_id,
                q1_low_voice=q1,
                q2_chills=q2,
                q3_first_call=q3,
                q4_unseen=q4,
                speech_format=speech_format,
                speech_text=speech_text,
                voice_id=voice_id,
                music_track=track["name"],
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
    Start generation in background. Returns a job_id immediately,
    along with the meditation audio URL so the frontend can start
    playing meditation while generation runs.
    Poll /generate/status/{job_id} for progress.
    """
    job_id = uuid.uuid4().hex[:16]

    # Select track (single-track mode, always returns a_thousand_hearts)
    track_name = select_track(
        q1_low_voice=req.q1_low_voice,
        q2_chills=req.q2_chills,
        q3_first_call=req.q3_first_call,
        q4_unseen=req.q4_unseen,
    )

    # Meditation URL: raw unmixed audio served from /assets/ static mount
    base = cfg.PUBLIC_BASE_URL.rstrip("/") if cfg.PUBLIC_BASE_URL else ""
    meditation_url = f"{base}/assets/{cfg.MEDITATION_FILE}"

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
        args=(job_id, req.q1_low_voice, req.q2_chills, req.q3_first_call, req.q4_unseen, track_name),
        daemon=True,
    )
    t.start()

    return GenerateJobResponse(job_id=job_id, meditation_url=meditation_url)


@r.get("/generate/status/{job_id}", response_model=GenerateStatusResponse)
def generate_status(job_id: str):
    """Poll this endpoint for generation progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return GenerateStatusResponse(**job)


@r.post("/feedback", response_model=StatusResponse)
def save_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Save chills data and post-experience feedback."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.chills_count = req.chills_count or 0
    session.chills_timestamps_json = req.chills_timestamps_json
    session.feedback_text = req.feedback_text
    session.completed_at = datetime.now(timezone.utc)

    db.commit()
    return StatusResponse(status="ok")


@r.post("/email", response_model=StatusResponse)
def save_email(req: EmailRequest, db: Session = Depends(get_db)):
    """Save email from the done screen beta signup."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.email = req.email
    db.commit()
    return StatusResponse(status="ok")


@r.get("/audio/{filename}")
def serve_audio(filename: str):
    """Serve a generated audio file (final mix or voice-only raw)."""
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
    """Export all sessions as CSV."""
    sessions = db.query(DemoSession).order_by(DemoSession.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "session_id",
        "user_id",
        # Questions
        "q1_low_voice",
        "q2_chills",
        "q3_first_call",
        "q4_unseen",
        # AI generation
        "speech_format",
        "speech_text",
        "voice_id",
        "music_track",
        "audio_filename",
        # Chills data
        "chills_count",
        "chills_timestamps",
        # Feedback
        "feedback_text",
        # Email
        "email",
        # Timing
        "generation_time_seconds",
        "created_at",
        "completed_at",
    ])

    for s in sessions:
        writer.writerow([
            s.session_id,
            s.user_id,
            s.q1_low_voice,
            s.q2_chills,
            s.q3_first_call,
            s.q4_unseen,
            s.speech_format,
            s.speech_text,
            s.voice_id,
            s.music_track,
            s.audio_filename,
            s.chills_count,
            s.chills_timestamps_json,
            s.feedback_text,
            s.email,
            s.generation_time_seconds,
            s.created_at,
            s.completed_at,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jolter_sessions.csv"},
    )
