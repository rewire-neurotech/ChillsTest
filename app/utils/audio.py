# app/utils/audio.py
from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Tuple
from pydub import AudioSegment


_STAGE_DIR_RE = re.compile(
    r"^\s*(?:\[.*?\]|\(.*?\)|soft\s+instrumental\s+music\s+(?:begins|starts|fades)|background\s+music.*)$",
    re.IGNORECASE,
)

def clean_script(s: str) -> str:
    lines = []
    for raw in s.splitlines():
        t = raw.strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]"):
            continue
        if t.startswith("(") and t.endswith(")"):
            continue
        if _STAGE_DIR_RE.search(t):
            continue
        lines.append(t)
    text = " ".join(lines)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def load_audio(path: str | Path) -> AudioSegment:
    p = str(path)
    
    return AudioSegment.from_file(p)

def normalize_dbfs(seg: AudioSegment, target_dbfs: float = -1.0) -> AudioSegment:
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)

def loop_to(seg: AudioSegment, target_ms: int) -> AudioSegment:
    if len(seg) >= target_ms:
        return seg[:target_ms]
    reps = target_ms // len(seg) + 1
    out = seg * reps
    return out[:target_ms]

def make_stereo(seg: AudioSegment) -> AudioSegment:
    
    if seg.channels == 2:
        return seg
    return seg.set_channels(2)

def export_mp3(seg: AudioSegment, out_path: str | Path, bitrate: str = "192k") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(out_path), format="mp3", bitrate=bitrate)

def duration_ms(seg: AudioSegment) -> int:
    return int(len(seg))


def content_duration_ms(
    seg: AudioSegment,
    silence_threshold_dbfs: float = -45.0,
    window_ms: int = 500,
    min_silent_tail_ms: int = 2000,
) -> int:
    """
    Return the duration in ms of the actual audio content, stripping
    trailing silence from the end.

    Scans backward from the end in windows of `window_ms`. Once it finds
    a window whose RMS is above `silence_threshold_dbfs`, that position
    is treated as the content end.

    If the trailing silence is shorter than `min_silent_tail_ms`, the
    full duration is returned (i.e. there's no meaningful silence to strip).

    Falls back to full duration if anything goes wrong.
    """
    total_ms = int(len(seg))
    if total_ms <= 0:
        return 0

    try:
        # Scan backward from the end
        pos = total_ms
        while pos > 0:
            chunk_start = max(0, pos - window_ms)
            chunk = seg[chunk_start:pos]

            # Calculate RMS in dBFS for this chunk
            if chunk.rms > 0:
                full_scale = float(1 << (8 * chunk.sample_width - 1))
                rms_dbfs = 20.0 * math.log10(chunk.rms / full_scale)
            else:
                rms_dbfs = -120.0

            if rms_dbfs > silence_threshold_dbfs:
                # Found actual content here — content ends at `pos`
                content_end = pos
                silent_tail = total_ms - content_end

                if silent_tail < min_silent_tail_ms:
                    # Trailing silence is negligible, return full duration
                    return total_ms

                print(
                    f"[audio] Content ends at {content_end}ms, "
                    f"trailing silence: {silent_tail}ms (stripped)"
                )
                return content_end

            pos -= window_ms

        # Entire file is silence — return full duration as fallback
        return total_ms

    except Exception as e:
        print(f"[audio] content_duration_ms error: {e}, returning full duration")
        return total_ms
