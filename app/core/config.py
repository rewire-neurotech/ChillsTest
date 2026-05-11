import os
from pathlib import Path


class Config:
    """Minimal config for Jolter backend."""

    # --- API Keys ---
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # --- ElevenLabs ---
    ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

    # --- Claude ---
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    # --- FFmpeg ---
    FFMPEG_BIN: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    FFPROBE_BIN: str = os.getenv("FFPROBE_BIN", "ffprobe")

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    OUT_DIR: str = os.getenv("OUT_DIR", "/tmp/rewire-demo-output")

    @property
    def out_dir_path(self) -> Path:
        p = Path(self.OUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- Track Registry ---
    TRACKS = {
        "ad_infinitum": {
            "file": "ad_infinitum.mpeg",
            "voice_id": "lMILJ9d29MrRXy9BIgcz",
            "description": "Dark, atmospheric, cinematic - deep emotional intensity",
        },
        # Legacy track kept for backward compatibility with existing sessions
        "a_thousand_hearts": {
            "file": "a_thousand_hearts.mpeg",
            "voice_id": "lMILJ9d29MrRXy9BIgcz",
            "description": "Gentle, emotional, intimate - warmth and tenderness",
        },
    }

    DEFAULT_TRACK: str = "ad_infinitum"

    def get_track(self, track_name: str = None) -> dict:
        """Return track info dict with resolved file paths."""
        name = track_name or self.DEFAULT_TRACK
        track = self.TRACKS.get(name)
        if not track:
            track = self.TRACKS[self.DEFAULT_TRACK]
            name = self.DEFAULT_TRACK
        return {
            "name": name,
            "file": self.ASSETS_DIR / track["file"],
            "voice_id": track["voice_id"],
            "description": track["description"],
        }

    # --- Meditation ---
    # Raw unmixed meditation audio served directly from assets
    MEDITATION_FILE: str = "christian_meditation.mpeg"

    @property
    def meditation_path(self) -> Path:
        return self.ASSETS_DIR / self.MEDITATION_FILE

    # --- Backward compatibility ---
    @property
    def MUSIC_FILE(self) -> Path:
        return self.ASSETS_DIR / self.TRACKS[self.DEFAULT_TRACK]["file"]

    @property
    def ELEVENLABS_VOICE_ID(self) -> str:
        return self.TRACKS[self.DEFAULT_TRACK]["voice_id"]

    # --- DB ---
    DB_URL: str = os.getenv("DB_URL", "sqlite:///./demo.db")

    # --- Auth ---
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    # --- CORS ---
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", "*"
    ).split(",")

    # --- Public URL for serving audio ---
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "")

    # --- Encryption (AES-256 via Fernet for sensitive fields) ---
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")

    # --- Web Push (VAPID keys for push notifications) ---
    VAPID_PUBLIC_KEY: str = os.getenv("VAPID_PUBLIC_KEY", "")
    VAPID_PRIVATE_KEY: str = os.getenv("VAPID_PRIVATE_KEY", "")
    VAPID_CLAIMS_EMAIL: str = os.getenv("VAPID_CLAIMS_EMAIL", "mailto:hello@rewire.bio")


cfg = Config()
