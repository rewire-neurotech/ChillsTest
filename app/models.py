from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime
)
from datetime import datetime, timezone
from app.db import Base


class DemoSession(Base):
    __tablename__ = "demo_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)

    # --- 4 Questions ---
    q1_wound = Column(Text)
    q2_chills_trigger = Column(Text)
    q3_hidden_truth = Column(Text)
    q4_first_tell = Column(Text)

    # --- Demographics (filled during wait) ---
    age = Column(String(20))
    gender = Column(String(40))
    ethnicity = Column(String(80))

    # --- AI Generation ---
    speech_format = Column(String(40))
    speech_text = Column(Text)
    voice_id = Column(String(60))
    music_track = Column(String(120))
    audio_filename = Column(String(200))

    # --- Chills Feedback (original - kept for backward compat) ---
    felt_chills = Column(Boolean)
    chills_count = Column(Integer, default=0)
    chills_timestamps_json = Column(Text)
    experience_driver = Column(String(40))
    feedback_note = Column(Text)

    # --- Universal Questions (everyone answers) ---
    crying_response = Column(String(40))          # "full_tears" / "eyes_watered" / "lump_in_throat" / "no"
    eyes_open_closed = Column(String(20))         # "open" / "closed" / "both"
    inspired_to_do = Column(Text)                 # free text

    # --- Chills YES branch ---
    chills_intensity = Column(Integer)             # 0-10 slider
    chills_trigger_json = Column(Text)             # JSON array: ["music","voice","speech_text","specific_phrase","crescendo","other"]
    chills_trigger_other = Column(Text)            # free text if "other" selected
    chills_body_location_json = Column(Text)       # JSON array: ["scalp","back_of_neck","spine","arms","chest","full_body"]
    chills_peak_timing = Column(String(40))        # "beginning" / "middle" / "end" / "specific_phrase"
    chills_reflection = Column(Text)               # "What gave you chills? Did this make you think of anything?"

    # --- Chills NO branch ---
    no_chills_barriers_json = Column(Text)         # JSON array of barriers
    no_chills_closeness = Column(String(60))       # "not_at_all" / "flicker_faded" / "almost_there" / "felt_something"
    no_chills_emotional_shift = Column(Boolean)    # did they experience emotional shift anyway?
    no_chills_emotional_describe = Column(Text)    # free text if yes

    # --- Emotional Breakthrough Inventory (1-7 Likert) ---
    ebi_faced_difficult = Column(Integer)          # "I faced emotionally difficult feelings..."
    ebi_resolution = Column(Integer)               # "I experienced a resolution..."
    ebi_explore = Column(Integer)                  # "I felt able to explore challenging emotions..."
    ebi_breakthrough = Column(Integer)             # "I had an emotional breakthrough"
    ebi_closure = Column(Integer)                  # "I was able to get a sense of closure..."
    ebi_release = Column(Integer)                  # "I achieved an emotional release..."
    ebi_stuck_resisting = Column(Integer)          # "I was resisting and avoiding..." (reverse-scored)
    ebi_stuck_throughout = Column(Integer)         # "I felt emotionally stuck..." (reverse-scored)

    # --- Content Quality (everyone) ---
    content_relevance = Column(Integer)            # 1-5 (Generic -> Deeply personal)
    content_inauthentic = Column(Boolean)          # did any line feel off?
    content_inauthentic_detail = Column(Text)      # free text if yes
    content_music_match = Column(String(40))       # "perfect" / "good_not_ideal" / "distracting" / "wrong_mood"
    content_pacing = Column(String(20))            # "too_fast" / "just_right" / "too_slow"

    # --- Open-ended (everyone) ---
    open_improve = Column(Text)                    # "What's one thing that would make this hit harder?"
    open_final_thoughts = Column(Text)             # "Any thoughts or feelings after that?"

    # --- Optional ---
    prolific_id = Column(String(100))
    email = Column(String(200))

    # --- Journal Notes (auto-saved, JSON array of note objects) ---
    notes_json = Column(Text)

    # --- Timing ---
    generation_time_seconds = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
