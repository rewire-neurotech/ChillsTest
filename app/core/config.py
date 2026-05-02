import os
from pathlib import Path


class Config:
    """Minimal config for ReWire Demo backend."""

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
    # Each track: music file, voice ID, meditation mix, and description
    TRACKS = {
        "heroes_wwii": {
            "file": "heroes_wwii.mp3",
            "voice_id": "0yXkuUWXDHdmdQJugJLb",
            "meditation": "meditation_heroes_wwii.mp3",
            "description": "Cinematic WW2 orchestral - intense, triumphant, heavy",
        },
        "a_thousand_hearts": {
            "file": "a_thousand_hearts.mpeg",
            "voice_id": "lMILJ9d29MrRXy9BIgcz",
            "meditation": "meditation_a_thousand_hearts.mp3",
            "description": "Gentle, emotional, intimate - warmth and tenderness",
        },
    }

    DEFAULT_TRACK: str = "heroes_wwii"

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
            "meditation": self.ASSETS_DIR / track["meditation"],
            "description": track["description"],
        }

    # --- Backward compatibility ---
    @property
    def MUSIC_FILE(self) -> Path:
        return self.ASSETS_DIR / self.TRACKS[self.DEFAULT_TRACK]["file"]

    @property
    def ELEVENLABS_VOICE_ID(self) -> str:
        return self.TRACKS[self.DEFAULT_TRACK]["voice_id"]

    # --- DB ---
    DB_URL: str = os.getenv("DB_URL", "sqlite:///./demo.db")

    # --- CORS ---
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", "*"
    ).split(",")

    # --- Public URL for serving audio ---
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "")


cfg = Config()
