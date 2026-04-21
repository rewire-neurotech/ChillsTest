from __future__ import annotations
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydub import AudioSegment

from ..utils.audio import (
    load_audio,
    normalize_dbfs,
    make_stereo,
    duration_ms,
)


def _ffmpeg_bin(custom_bin: str | None) -> str:
    return custom_bin or shutil.which("ffmpeg") or "ffmpeg"


def _ffmpeg_has(ffmpeg_path: str, needle: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=True,
        )
        return needle in (out.stdout + out.stderr)
    except Exception:
        return False


def _atempo_chain(factor: float) -> str:
    if factor <= 0:
        return "atempo=1.0"
    parts = []
    f = float(factor)
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f *= 2.0
    parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def _retime_with_ffmpeg(src_wav: str, target_ms: int, ffmpeg_path: str) -> str:
    seg = AudioSegment.from_file(src_wav)
    cur_ms = len(seg)
    if cur_ms <= 0 or target_ms <= 0:
        out_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        seg.export(out_raw.name, format="wav")
        return out_raw.name

    max_delta_ratio = 0.15
    lo = int(target_ms * (1 - max_delta_ratio))
    hi = int(target_ms * (1 + max_delta_ratio))
    if cur_ms < lo:
        pad = AudioSegment.silent(duration=target_ms - cur_ms, frame_rate=seg.frame_rate)
        seg = seg + pad
    elif cur_ms > hi:
        seg = seg[:target_ms]

    mid_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    seg.export(mid_wav.name, format="wav")

    cur_ms2 = len(AudioSegment.from_file(mid_wav.name))
    factor = max(1e-6, cur_ms2 / float(target_ms))  # >1 => faster/shorter, <1 => slower/longer

    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        mid_wav.name,
        "-filter:a",
        _atempo_chain(factor),
        out_wav.name,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav.name


def _peak_dbfs(seg: AudioSegment) -> float:
    sample_peak = seg.max
    full_scale = float(1 << (8 * seg.sample_width - 1))
    if sample_peak <= 0:
        return -120.0
    return 20.0 * math.log10(sample_peak / full_scale)


def _apply_peak_guard(seg: AudioSegment, ceiling_dbfs: float = -1.0) -> AudioSegment:
    pk = _peak_dbfs(seg)
    headroom = ceiling_dbfs - pk
    if headroom < 0:
        seg = seg.apply_gain(headroom)
    return seg


def _hard_fit(seg: AudioSegment, target_ms: int) -> AudioSegment:
    if len(seg) < target_ms:
        return seg + AudioSegment.silent(duration=target_ms - len(seg), frame_rate=seg.frame_rate)
    elif len(seg) > target_ms:
        return seg[:target_ms]
    return seg


def _hard_fit_samples(seg: AudioSegment, target_samples_per_ch: int) -> AudioSegment:
    ch = seg.channels
    sw = seg.sample_width
    frame_bytes = ch * sw

    raw = seg.raw_data
    total_frames = len(raw) // frame_bytes
    if total_frames == target_samples_per_ch:
        return seg

    if total_frames > target_samples_per_ch:
        new_bytes = target_samples_per_ch * frame_bytes
        raw = raw[:new_bytes]
        return seg._spawn(raw)
    else:
        need_frames = target_samples_per_ch - total_frames
        pad_ms = int(1000 * need_frames / seg.frame_rate)
        silence = AudioSegment.silent(duration=pad_ms, frame_rate=seg.frame_rate).set_channels(ch).set_sample_width(sw)
        out = seg + silence
        raw2 = out.raw_data[: target_samples_per_ch * frame_bytes]
        return out._spawn(raw2)


def _decode_samples(path: Path) -> tuple[int, int, int]:
    seg = AudioSegment.from_file(str(path))
    ch = seg.channels
    sw = seg.sample_width
    frame_bytes = ch * sw
    total_frames = len(seg.raw_data) // frame_bytes
    return total_frames, seg.frame_rate, ch


def _rms_dbfs(chunk: AudioSegment) -> float:
    if chunk.rms <= 1:
        return -120.0
    return 20.0 * math.log10(chunk.rms / float(1 << (8 * chunk.sample_width - 1)))


def analyze_music(music_path: str | Path, frame_ms: int = 200) -> dict:
    seg = make_stereo(load_audio(music_path).set_frame_rate(44100))
    if len(seg) <= 0:
        return {"drop_ms": None, "frame_ms": frame_ms}

    win = max(50, frame_ms)
    energies: list[float] = []
    for i in range(0, len(seg), win):
        chunk = seg[i: i + win]
        energies.append(_rms_dbfs(chunk))

    if not energies:
        return {"drop_ms": None, "frame_ms": win}

    k = 4
    smoothed: list[float] = []
    for i in range(len(energies)):
        lo = max(0, i - k)
        hi = min(len(energies), i + k + 1)
        smoothed.append(sum(energies[lo:hi]) / (hi - lo))

    diffs = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]
    if not diffs:
        return {"drop_ms": None, "frame_ms": win}

    search_len = max(1, int(0.8 * len(diffs)))
    search_diffs = diffs[:search_len]

    max_idx = max(range(len(search_diffs)), key=lambda i: search_diffs[i])
    drop_ms = max_idx * win

    return {"drop_ms": drop_ms, "frame_ms": win}


def _duck_music_to_voice(
    music: AudioSegment,
    voice: AudioSegment,
    floor_boost_db: float = 3.0,
    max_duck_db: float = -1.0,
    attack_ms: int = 180,
    release_ms: int = 650,
    win_ms: int = 60,
    lookahead_ms: int = 500,
    gap_hold_ms: int = 2600,
) -> AudioSegment:

    win = max(20, win_ms)
    step = win
    out = AudioSegment.silent(duration=0, frame_rate=music.frame_rate)
    prev_gain = 0.0

    silence_threshold_db = -45.0

    in_voice_region = False
    silence_run_ms = 0

    for i in range(0, len(music), step):
        i_ms = i

        m_chunk = music[i_ms: i_ms + win]

        # Lookahead: check voice energy slightly ahead
        start_v_la = i_ms + lookahead_ms
        end_v_la = start_v_la + win
        v_chunk_la = voice[start_v_la:end_v_la]
        v_db_la = _rms_dbfs(v_chunk_la)

        # Current voice energy
        v_now = voice[i_ms: i_ms + win]
        v_now_db = _rms_dbfs(v_now)
        voice_now = v_now_db > silence_threshold_db

        if voice_now:
            in_voice_region = True
            silence_run_ms = 0
        else:
            if in_voice_region:
                silence_run_ms += step
                if silence_run_ms >= gap_hold_ms:
                    in_voice_region = False
            else:
                silence_run_ms = 0

        # Calculate target gain based on voice energy
        if v_db_la <= -48.0:
            target = floor_boost_db
        elif v_db_la >= -26.0:
            target = max_duck_db
        else:
            t = (v_db_la + 48.0) / 22.0  # 0..1
            target = max_duck_db * t + floor_boost_db * (1 - t)

        # Hold ducking during short gaps in voice
        short_gap = (not voice_now) and in_voice_region and (silence_run_ms < gap_hold_ms)
        if short_gap:
            hold_level = max(max_duck_db + 1.5, -1.5)
            target = min(target, hold_level)

        # Smooth gain transitions
        if target < prev_gain:
            alpha = min(1.0, step / float(max(1, attack_ms)))
        else:
            alpha = min(1.0, step / float(max(1, release_ms)))
        gain = prev_gain + alpha * (target - prev_gain)
        prev_gain = gain

        out += m_chunk.apply_gain(gain)

    return out


def mix(
    voice_path: str | Path,
    music_path: str | Path,
    out_path: str | Path,
    duck_db: float = 4.0,
    sync_mode: Literal["retime_voice_to_music", "retime_music_to_voice", "no_retime_trim_pad"] = "retime_voice_to_music",
    voice_target_dbfs: float = -13.0,
    music_target_dbfs: float = -14.5,
    final_peak_dbfs: float = -1.0,
    music_fadein_ms: int = 3000,
    ffmpeg_bin: str | None = None,
    **_ignored,
) -> int:

    ffmpeg_path = _ffmpeg_bin(ffmpeg_bin)

    # Load and normalize music
    music = make_stereo(load_audio(music_path).set_frame_rate(44100))
    music = normalize_dbfs(music, music_target_dbfs)
    if len(music) <= 0:
        raise ValueError("Music stem is empty or unreadable.")

    # Apply fade-in to music (voice starts first, music enters gradually)
    if music_fadein_ms > 0:
        music = music.fade_in(music_fadein_ms)

    # EQ: gentle bass reduction to avoid muddiness
    if _ffmpeg_has(ffmpeg_path, "equalizer"):
        tmp_m_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_m_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        music.export(tmp_m_in.name, format="wav")
        try:
            af = "equalizer=f=50:t=h:w=2:g=-2,equalizer=f=80:t=h:w=2:g=-1"
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    tmp_m_in.name,
                    "-af",
                    af,
                    tmp_m_out.name,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            music = AudioSegment.from_file(tmp_m_out.name)
        except Exception:
            pass

    # Light compression to even out dynamics without saturating
    if _ffmpeg_has(ffmpeg_path, "acompressor") or _ffmpeg_has(ffmpeg_path, "dynaudnorm"):
        tmp_m2_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_m2_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        music.export(tmp_m2_in.name, format="wav")
        try:
            if _ffmpeg_has(ffmpeg_path, "acompressor"):
                af = "acompressor=threshold=-20dB:ratio=2:attack=18:release=280:makeup=1.5"
            else:
                af = "dynaudnorm=f=125:s=8"
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_m2_in.name, "-af", af, tmp_m2_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            music = AudioSegment.from_file(tmp_m2_out.name)
        except Exception:
            pass

    # Load and normalize voice
    voice = make_stereo(load_audio(voice_path).set_frame_rate(44100))
    voice = normalize_dbfs(voice, voice_target_dbfs)
    if len(voice) <= 0:
        raise ValueError("Voice stem is empty or unreadable.")

    # Voice EQ: clarity boost
    if _ffmpeg_has(ffmpeg_path, "equalizer") or _ffmpeg_has(ffmpeg_path, "highpass"):
        tmp_v_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_v_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice.export(tmp_v_in.name, format="wav")

        vf = "highpass=f=70,equalizer=f=150:t=h:w=1.5:g=2,equalizer=f=3800:t=h:w=2:g=-1.5"
        try:
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_v_in.name, "-af", vf, tmp_v_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            voice = AudioSegment.from_file(tmp_v_out.name).set_frame_rate(44100).set_channels(2)
        except Exception:
            pass

    # Sync lengths
    ch = music.channels
    sw = music.sample_width
    frame_bytes = ch * sw

    target_samples_per_ch = len(music.raw_data) // frame_bytes
    target_ms = int(round(1000 * target_samples_per_ch / music.frame_rate))

    if sync_mode == "retime_voice_to_music":
        tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice.export(tmp_v.name, format="wav")
        v_wav = _retime_with_ffmpeg(tmp_v.name, target_ms, ffmpeg_path)
        voice_exact = AudioSegment.from_file(v_wav).set_frame_rate(44100).set_channels(ch)
        music_exact = music

    elif sync_mode == "retime_music_to_voice":
        # Here we retime music to match the voice duration.
        voice_frames = len(voice.raw_data) // frame_bytes
        target_samples_per_ch = voice_frames
        target_ms = int(round(1000 * target_samples_per_ch / voice.frame_rate))

        tmp_m = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        music.export(tmp_m.name, format="wav")
        m_wav = _retime_with_ffmpeg(tmp_m.name, target_ms, ffmpeg_path)
        music_exact = AudioSegment.from_file(m_wav).set_frame_rate(44100).set_channels(ch)
        voice_exact = voice

    else:
        voice_frames = len(voice.raw_data) // frame_bytes
        target_samples_per_ch = voice_frames
        target_ms = int(round(1000 * target_samples_per_ch / voice.frame_rate))

        voice_exact = _hard_fit(voice, target_ms)
        music_exact = _hard_fit(music, target_ms)

    voice_exact = _hard_fit_samples(voice_exact, target_samples_per_ch)
    music_exact = _hard_fit_samples(music_exact, target_samples_per_ch)

    # Duck music around voice with softer settings
    music_adapt = _duck_music_to_voice(
        music_exact,
        voice_exact,
        floor_boost_db=3.0,
        max_duck_db=-1.0,
        attack_ms=180,
        release_ms=650,
        win_ms=60,
        lookahead_ms=500,
        gap_hold_ms=2600,
    )
    music_adapt = _hard_fit_samples(music_adapt, target_samples_per_ch)

    final_mix = music_adapt.overlay(voice_exact)

    # Peak guard
    final_mix = _apply_peak_guard(final_mix, ceiling_dbfs=final_peak_dbfs)
    final_mix = _hard_fit_samples(final_mix, target_samples_per_ch)

    tail_ms = min(900, max(350, target_ms // 18))
    final_mix = final_mix.fade_out(tail_ms)

    # Final loudness normalization
    tmp_wav_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    final_mix.export(tmp_wav_in.name, format="wav")

    tmp_wav_polished = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if _ffmpeg_has(ffmpeg_path, "loudnorm"):
        af = "loudnorm=I=-16.0:TP=-1.0:LRA=11:linear=1"
    else:
        af = "dynaudnorm=f=125:s=12,volume=-0.6dB"

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            tmp_wav_in.name,
            "-af",
            af,
            "-ar",
            "44100",
            "-ac",
            str(ch),
            tmp_wav_polished.name,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    polished = AudioSegment.from_file(tmp_wav_polished.name).set_channels(ch).set_frame_rate(44100)
    polished = _hard_fit_samples(polished, target_samples_per_ch)

    tmp_wav_exact = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    polished.export(tmp_wav_exact.name, format="wav")

    t_sec = f"{target_samples_per_ch / polished.frame_rate:.6f}"
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            tmp_wav_exact.name,
            "-t",
            t_sec,
            "-shortest",
            "-ar",
            "44100",
            "-ac",
            str(ch),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "256k",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    out_frames, out_sr, out_ch = _decode_samples(Path(out_path))

    SAMPLE_TOL = 64

    ok_by_samples = (
        out_sr == 44100
        and out_ch == ch
        and abs(out_frames - target_samples_per_ch) <= SAMPLE_TOL
    )

    if not ok_by_samples:
        target_ms_exact = int(round(1000 * target_samples_per_ch / 44100.0))
        actual_ms_exact = int(round(1000 * out_frames / 44100.0))
        MS_TOL = 3
        if abs(actual_ms_exact - target_ms_exact) > MS_TOL:
            raise RuntimeError(
                f"Final length drift: {actual_ms_exact} ms vs {target_ms_exact} ms "
                f"({out_frames} vs {target_samples_per_ch} samples)."
            )

    return int(round(1000 * target_samples_per_ch / polished.frame_rate))
