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

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from pydub import AudioSegment

from app.core.config import cfg
from app.db import get_db, SessionLocal, engine
from app.models import DemoSession, User, AuditLog, StickyNote, PushSubscription
from app.services.prompt import build_user_prompt, build_question_prompt
from app.services.llm import generate_speech, call_claude
from app.services.tts import synth
from app.services.mix import mix as mix_audio
from app.services.music_selector import select_track
from app.utils.audio import load_audio, duration_ms, content_duration_ms
from app.utils.encryption import encrypt_field, decrypt_field, encrypt_file, decrypt_file_to_bytes
from app.routes.auth import get_current_user_required, get_current_user_optional

# Module-level import: happens once at server start, not during generation
try:
    from pywebpush import webpush
    _HAS_WEBPUSH = True
except ImportError:
    _HAS_WEBPUSH = False

r = APIRouter(prefix="/api/demo", tags=["demo"])

MUSIC_FADEIN_MS = 10     # Joaquin's new mix: near-zero fade-in, music enters immediately
TAIL_BUFFER_MS = 4000    # Music plays 4 seconds after voice ends


# --- Auto-migration: ensure job tracking columns exist on existing tables ---

def _ensure_job_columns():
    """Add stage/progress/gen_error columns to demo_sessions if missing.
    Runs once on startup. Existing completed rows get stage='done', progress=100."""
    from sqlalchemy import inspect, text
    try:
        insp = inspect(engine)
        existing = {c['name'] for c in insp.get_columns('demo_sessions')}
        with engine.begin() as conn:
            if 'stage' not in existing:
                conn.execute(text(
                    "ALTER TABLE demo_sessions ADD COLUMN stage VARCHAR(40) NOT NULL DEFAULT 'done'"
                ))
                print("[Demo] Added 'stage' column to demo_sessions")
            if 'progress' not in existing:
                conn.execute(text(
                    "ALTER TABLE demo_sessions ADD COLUMN progress INTEGER NOT NULL DEFAULT 100"
                ))
                print("[Demo] Added 'progress' column to demo_sessions")
            if 'gen_error' not in existing:
                conn.execute(text(
                    "ALTER TABLE demo_sessions ADD COLUMN gen_error TEXT"
                ))
                print("[Demo] Added 'gen_error' column to demo_sessions")
    except Exception as e:
        print(f"[Demo] Auto-migration note: {e}")

_ensure_job_columns()


def _ensure_note_columns():
    """Add prompt_question column to sticky_notes if missing.
    Runs once on startup."""
    from sqlalchemy import inspect, text
    try:
        insp = inspect(engine)
        existing = {c['name'] for c in insp.get_columns('sticky_notes')}
        with engine.begin() as conn:
            if 'prompt_question' not in existing:
                conn.execute(text(
                    "ALTER TABLE sticky_notes ADD COLUMN prompt_question TEXT"
                ))
                print("[Demo] Added 'prompt_question' column to sticky_notes")
    except Exception as e:
        print(f"[Demo] Auto-migration (sticky_notes) note: {e}")

_ensure_note_columns()


# --- DB-backed job status updates (replaces in-memory _jobs dict) ---

def _update_job_db(session_id: str, **kwargs):
    """Update generation job status in the database."""
    db = SessionLocal()
    try:
        row = db.query(DemoSession).filter(DemoSession.session_id == session_id).first()
        if row:
            for key, value in kwargs.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            db.commit()
    except Exception as e:
        print(f"[Demo] Job status update error: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


# --- Push notifications ---

def _send_push_notifications(user_id: int):
    """Send 'Your Jolt is ready' push. Runs in its own thread, never blocks generation."""
    if not _HAS_WEBPUSH or not user_id or not cfg.VAPID_PRIVATE_KEY:
        return
    db = SessionLocal()
    try:
        subs = db.query(PushSubscription).filter(PushSubscription.user_id == user_id).all()
        if not subs:
            return
        payload = json.dumps({"title": "ReWire", "body": "Your Jolt is ready"})
        for sub in subs:
            try:
                webpush(
                    subscription_info={
                        "endpoint": sub.endpoint,
                        "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                    },
                    data=payload,
                    vapid_private_key=cfg.VAPID_PRIVATE_KEY,
                    vapid_claims={"sub": cfg.VAPID_CLAIMS_EMAIL},
                )
            except Exception as e:
                if "410" in str(e) or "404" in str(e):
                    db.delete(sub)
                    db.commit()
    except Exception:
        pass
    finally:
        db.close()


# --- Sticky note question generation helper ---

def _generate_note_question(user_id: int, q1: str, q2: str, q3: str, q4: str, db_session=None):
    """Generate a personalized journal question for a new sticky note.
    Uses the user's original Q1-Q4 answers and their last 3 jolts.
    Returns the question string, or empty string if generation fails."""
    try:
        # Get recent jolt speech texts for context
        recent_jolts_text = "none yet"
        own_db = False
        db = db_session
        if db is None:
            db = SessionLocal()
            own_db = True
        try:
            recent_sessions = (
                db.query(DemoSession)
                .filter(DemoSession.user_id == user_id, DemoSession.speech_text.isnot(None))
                .order_by(DemoSession.created_at.desc())
                .limit(3)
                .all()
            )
            if recent_sessions:
                summaries = []
                for s in recent_sessions:
                    text = decrypt_field(s.speech_text) or ""
                    # Take first 80 words as a summary of each jolt
                    words = text.split()[:80]
                    if words:
                        summaries.append(" ".join(words) + "...")
                if summaries:
                    recent_jolts_text = " | ".join(summaries)
        finally:
            if own_db:
                db.close()

        # Build the prompt with user context
        system_prompt = build_question_prompt(
            q1_low_voice=q1,
            q2_chills=q2,
            q3_first_call=q3,
            q4_unseen=q4,
            recent_jolts=recent_jolts_text,
        )

        # Call Claude for a short question
        question = call_claude(system_prompt=system_prompt, user_message="Generate.", max_tokens=100)
        print(f"[Demo] Generated note question: {question}")
        return question
    except Exception as e:
        print(f"[Demo] Note question generation error: {e}")
        return ""


# --- Request / Response schemas ---

class GenerateRequest(BaseModel):
    q0_wish_easier: str = ""
    q1_low_voice: str
    q2_chills: str
    q3_first_call: str
    q4_unseen: str = ""
    note_id: Optional[int] = None
    place: Optional[str] = None


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

def _run_generate(job_id: str, q0: str, q1: str, q2: str, q3: str, q4: str, track_name: str, user_id: int = None, note_id: int = None, place: str = None):
    """Run the full generation pipeline in a background thread."""
    try:
        start = time.time()
        session_id = job_id  # reuse job_id as session_id

        _update_job_db(session_id, stage="writing", progress=5)

        # 0. Get track info (track already selected in /generate endpoint)
        track = cfg.get_track(track_name)
        print(f"[Demo] Using track: {track['name']} ({track['description']})")

        # 1. Get music duration info
        music_path = track["file"]
        if not music_path.exists():
            _update_job_db(session_id, stage="error", gen_error="Music file not found in assets")
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

        _update_job_db(session_id, stage="writing", progress=10)

        # 3. Generate speech text via Claude
        print(f"[Demo] Generating speech for session {session_id}, target_words={target_words}")
        speech_text = generate_speech(user_prompt)
        print(f"[Demo] Speech generated, {_word_count(speech_text)} words")

        _update_job_db(session_id, stage="synthesizing", progress=35)

        # 4. TTS via ElevenLabs
        print(f"[Demo] Synthesizing TTS...")
        voice_id = track["voice_id"]
        voice_wav_path = synth(
            text=speech_text,
            voice_id=voice_id,
            key=cfg.ELEVENLABS_API_KEY,
        )
        print(f"[Demo] TTS done: {voice_wav_path}")

        _update_job_db(session_id, stage="refining", progress=60)

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

            _update_job_db(session_id, progress=60 + (attempt + 1) * 4)

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

        _update_job_db(session_id, stage="mixing", progress=78)

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

        # Encrypt raw audio files at rest
        encrypt_file(str(cfg.out_dir_path / raw_voice_name))
        encrypt_file(str(cfg.out_dir_path / raw_music_name))

        # 8. Mix with music (Joaquin's DSP pipeline, untouched)
        print(f"[Demo] Mixing with music...")
        audio_filename = f"{session_id}.mp3"
        out_path = cfg.out_dir_path / audio_filename

        _update_job_db(session_id, stage="mixing", progress=82)

        mix_audio(
            voice_path=voice_wav_path,
            music_path=music_mix_path,
            out_path=str(out_path),
            sync_mode="retime_music_to_voice",
            music_fadein_ms=MUSIC_FADEIN_MS,
            music_premix_gain_db=-3.5,
            ffmpeg_bin=cfg.FFMPEG_BIN,
        )
        print(f"[Demo] Mix done: {out_path}")

        # Encrypt final mix audio at rest
        encrypt_file(str(out_path))

        elapsed = round(time.time() - start, 2)

        _update_job_db(session_id, stage="saving", progress=92)

        # 9. Save generation results to DB (update existing row created by /generate endpoint)
        db = SessionLocal()
        try:
            row = db.query(DemoSession).filter(DemoSession.session_id == session_id).first()
            if row:
                row.speech_format = speech_format
                row.speech_text = encrypt_field(speech_text)
                row.voice_id = voice_id
                row.music_track = track["name"]
                row.audio_filename = audio_filename
                row.generation_time_seconds = elapsed
                row.stage = "done"
                row.progress = 100
            db.commit()

            # 10. Handle sticky note creation/update
            if user_id:
                try:
                    if note_id:
                        # Jolt from sticky notes screen: update the existing note
                        existing_note = db.query(StickyNote).filter(
                            StickyNote.id == note_id,
                            StickyNote.user_id == user_id,
                        ).first()
                        if existing_note:
                            existing_note.session_id = session_id
                            existing_note.state = "ready"
                            existing_note.updated_at = datetime.now(timezone.utc)
                            print(f"[Demo] Updated existing note {note_id} -> ready, session_id={session_id}")
                        else:
                            print(f"[Demo] Note {note_id} not found for user {user_id}, skipping note update")
                    else:
                        # First jolt from questions flow: create 4 answer notes
                        # + 1 reflection note with a personalized question.
                        # Created in order so that newest-first ordering puts
                        # the reflection note on top, then q4, q3, q2, q1 below.
                        #
                        # Issue 3: Notes are IDLE (joltable), not watched.
                        # Each note includes the question context as a prefix.
                        answer_labels = [
                            ("q1", "When you are at your lowest, the voice in your head says: ", q1),
                            ("q2", "The last time you got chills or goosebumps: ", q2),
                            ("q3", "If something beautiful happened, the first person you'd call: ", q3),
                            ("q4", "Something true about you that nobody sees: ", q4),
                        ]
                        for label, prefix, answer_text in answer_labels:
                            if answer_text and answer_text.strip():
                                full_text = prefix + answer_text
                                note = StickyNote(
                                    user_id=user_id,
                                    text=encrypt_field(full_text),
                                    state="idle",
                                    session_id=None,
                                    place=place,
                                )
                                db.add(note)
                        db.flush()
                        print(f"[Demo] Created 4 answer notes (idle, joltable) for user {user_id}")

                        # Create reflection note with personalized question (newest = top of stack)
                        question = _generate_note_question(user_id, q1, q2, q3, q4, db_session=db)
                        blank_note = StickyNote(
                            user_id=user_id,
                            text="",
                            state="idle",
                            session_id=None,
                            place=place,
                            prompt_question=question if question else None,
                        )
                        db.add(blank_note)
                        print(f"[Demo] Created reflection note for user {user_id} (question: {question[:50] if question else 'none'})")

                    # Mark first jolt completed on user
                    user = db.query(User).filter(User.id == user_id).first()
                    if user and not user.has_completed_first_jolt:
                        user.has_completed_first_jolt = True

                    db.commit()
                except Exception as e:
                    print(f"[Demo] Sticky note creation/update error: {e}")
                    # Don't fail the whole session if note handling fails
        finally:
            db.close()

        print(f"[Demo] Session {session_id} complete in {elapsed}s")

        # Send push notification in its own thread (never blocks generation)
        if user_id:
            threading.Thread(target=_send_push_notifications, args=(user_id,), daemon=True).start()

    except Exception as e:
        print(f"[Demo] Generation error for job {job_id}: {e}")
        _update_job_db(job_id, stage="error", gen_error=str(e))


# --- Endpoints ---

@r.post("/generate", response_model=GenerateJobResponse)
def generate(req: GenerateRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user_optional)):
    """
    Start generation in background. Returns a job_id immediately,
    along with the meditation audio URL so the frontend can start
    playing meditation while generation runs.
    Poll /generate/status/{job_id} for progress.
    """
    job_id = uuid.uuid4().hex[:16]

    # Select track (single-track mode, always returns ad_infinitum)
    track_name = select_track(
        q1_low_voice=req.q1_low_voice,
        q2_chills=req.q2_chills,
        q3_first_call=req.q3_first_call,
        q4_unseen=req.q4_unseen,
    )

    # Meditation URL: raw unmixed audio served from /assets/ static mount
    base = cfg.PUBLIC_BASE_URL.rstrip("/") if cfg.PUBLIC_BASE_URL else ""
    meditation_url = f"{base}/assets/{cfg.MEDITATION_FILE}"

    # Extract user_id if logged in (passed to background thread)
    uid = user.id if user else None

    # Create DemoSession row immediately so status polling always finds it
    session = DemoSession(
        session_id=job_id,
        user_id=uid,
        q0_wish_easier=encrypt_field(req.q0_wish_easier),
        q1_low_voice=encrypt_field(req.q1_low_voice),
        q2_chills=encrypt_field(req.q2_chills),
        q3_first_call=encrypt_field(req.q3_first_call),
        q4_unseen=encrypt_field(req.q4_unseen),
        stage="queued",
        progress=0,
    )
    db.add(session)
    db.commit()

    t = threading.Thread(
        target=_run_generate,
        args=(job_id, req.q0_wish_easier, req.q1_low_voice, req.q2_chills, req.q3_first_call, req.q4_unseen, track_name),
        kwargs={"user_id": uid, "note_id": req.note_id, "place": req.place},
        daemon=True,
    )
    t.start()

    return GenerateJobResponse(job_id=job_id, meditation_url=meditation_url)


@r.get("/generate/status/{job_id}", response_model=GenerateStatusResponse)
def generate_status(job_id: str, db: Session = Depends(get_db)):
    """Poll this endpoint for generation progress."""
    session = db.query(DemoSession).filter(DemoSession.session_id == job_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Job not found")

    # Build audio URL from filename if generation is complete
    audio_url = None
    if session.audio_filename:
        base = cfg.PUBLIC_BASE_URL.rstrip("/") if cfg.PUBLIC_BASE_URL else ""
        audio_url = f"{base}/api/demo/audio/{session.audio_filename}"

    return GenerateStatusResponse(
        job_id=job_id,
        stage=session.stage,
        progress=session.progress,
        session_id=session.session_id if session.stage == "done" else None,
        audio_url=audio_url,
        error=session.gen_error,
    )


@r.post("/feedback", response_model=StatusResponse)
def save_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Save chills data and post-experience feedback."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.chills_count = req.chills_count or 0
    session.chills_timestamps_json = req.chills_timestamps_json
    session.feedback_text = encrypt_field(req.feedback_text)
    session.completed_at = datetime.now(timezone.utc)

    db.commit()
    return StatusResponse(status="ok")


@r.post("/email", response_model=StatusResponse)
def save_email(req: EmailRequest, db: Session = Depends(get_db)):
    """Save email from the done screen beta signup."""
    session = db.query(DemoSession).filter(DemoSession.session_id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.email = encrypt_field(req.email)
    db.commit()
    return StatusResponse(status="ok")


@r.get("/audio/{filename}")
def serve_audio(filename: str, request: Request, db: Session = Depends(get_db)):
    """Serve a generated audio file (final mix or voice-only raw). Decrypts on-the-fly."""
    filepath = cfg.out_dir_path / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    # Audit log audio access
    try:
        log = AuditLog(
            action="audio_access",
            target=filename,
            ip_address=request.client.host if request.client else None,
        )
        db.add(log)
        db.commit()
    except Exception:
        pass  # Don't block audio serving if logging fails

    # Decrypt file and return with Content-Length so the browser knows
    # the full size, can report accurate duration, and fires onended.
    audio_bytes = decrypt_file_to_bytes(str(filepath))
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Accept-Ranges": "bytes",
        },
    )


@r.get("/export/csv")
def export_csv(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user_required)):
    """Export all sessions as CSV. Requires admin authentication."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Audit log the export
    try:
        log = AuditLog(
            user_id=user.id,
            user_email=user.email,
            action="csv_export",
            target="all_sessions",
            ip_address=request.client.host if request.client else None,
        )
        db.add(log)
        db.commit()
    except Exception:
        pass  # Don't block export if logging fails

    sessions = db.query(DemoSession).order_by(DemoSession.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "session_id",
        "user_id",
        # Questions
        "q0_wish_easier",
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
            decrypt_field(s.q0_wish_easier),
            decrypt_field(s.q1_low_voice),
            decrypt_field(s.q2_chills),
            decrypt_field(s.q3_first_call),
            decrypt_field(s.q4_unseen),
            s.speech_format,
            decrypt_field(s.speech_text),
            s.voice_id,
            s.music_track,
            s.audio_filename,
            s.chills_count,
            s.chills_timestamps_json,
            decrypt_field(s.feedback_text),
            decrypt_field(s.email),
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
