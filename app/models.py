from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
)
from datetime import datetime, timezone
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    password_hash = Column(String(300), nullable=True)  # null for Google-auth users
    name = Column(String(200), nullable=True)
    auth_provider = Column(String(20), nullable=False, default="local")  # "local" or "google"
    google_id = Column(String(200), nullable=True, unique=True)
    is_admin = Column(Boolean, default=False, nullable=False, server_default="false")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DemoSession(Base):
    __tablename__ = "demo_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)

    # --- Link to user (nullable for backward compat during transition) ---
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # --- 4 Questions (new order matching Felix's UI) ---
    q1_low_voice = Column(Text)       # "What does the voice in your head say at your lowest?"
    q2_chills = Column(Text)          # "When was the last time you got chills or goosebumps?"
    q3_first_call = Column(Text)      # "If something beautiful happened, who's the first person you'd call?"
    q4_unseen = Column(Text)          # "What's something true about you that nobody sees?"

    # --- AI Generation ---
    speech_format = Column(String(40))
    speech_text = Column(Text)
    voice_id = Column(String(60))
    music_track = Column(String(120))
    audio_filename = Column(String(200))

    # --- Chills Data (from stimulus tap-to-log) ---
    chills_count = Column(Integer, default=0)
    chills_timestamps_json = Column(Text)     # JSON array of timestamps in seconds

    # --- Post-Stimulus Feedback (single text field) ---
    feedback_text = Column(Text)

    # --- Email (from done screen beta signup) ---
    email = Column(String(200))

    # --- Timing ---
    generation_time_seconds = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user_email = Column(String(200), nullable=True)
    action = Column(String(100), nullable=False)       # e.g. "csv_export", "audio_access"
    target = Column(String(300), nullable=True)         # e.g. session_id or filename
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))