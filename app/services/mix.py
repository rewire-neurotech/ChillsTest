from __future__ import annotations
import array as _array
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pyloudnorm as pyln
from pedalboard import HighpassFilter, Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment

from ..utils.audio import (
    load_audio,
    normalize_dbfs,
    make_stereo,
    duration_ms,
)


def _apply_reverb(seg: AudioSegment, room_size: float = 0.85, damping: float = 0.55,
                  wet_level: float = 0.15, dry_level: float = 0.85,
                  pre_gain_db: float = -6.0, hpf_hz: float = 500.0,
                  wet_boost_db: float = 0.0) -> AudioSegment:
    # Reverb send: HPF → full-wet reverb (parallel to dry, not in series)
    tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    seg.apply_gain(pre_gain_db).export(tmp_in.name, format="wav")
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=hpf_hz),
        Reverb(room_size=room_size, damping=damping, wet_level=1.0, dry_level=0.0),
    ])
    with AudioFile(tmp_in.name) as f_in:
        with AudioFile(tmp_out.name, "w", f_in.samplerate, f_in.num_channels) as f_out:
            chunk_size = f_in.samplerate
            while f_in.tell() < f_in.frames:
                chunk = f_in.read(chunk_size)
                f_out.write(board(chunk, f_in.samplerate, reset=False))
    wet = AudioSegment.from_file(tmp_out.name)
    # Dry: original signal untouched; mix at their respective levels
    dry_db = 20.0 * math.log10(max(dry_level, 1e-6))
    wet_db = 20.0 * math.log10(max(wet_level, 1e-6)) + wet_boost_db
    return seg.apply_gain(dry_db).overlay(wet.apply_gain(wet_db))


def _normalize_peak(seg: AudioSegment, target_dbfs: float) -> AudioSegment:
    peak = seg.max_dBFS
    if peak == float("-inf"):
        return seg
    return seg.apply_gain(target_dbfs - peak)


def _normalize_lufs(seg: AudioSegment, target_lufs: float) -> AudioSegment:
    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float64)
    raw /= 32768.0
    data = raw.reshape(-1, seg.channels)
    meter = pyln.Meter(seg.frame_rate)
    loudness = meter.integrated_loudness(data)
    if not np.isfinite(loudness) or loudness < -70.0:
        return seg
    gain_db = target_lufs - loudness
    return seg.apply_gain(gain_db)


def _level_envelope(
    seg: AudioSegment,
    frame_ms: int = 1000,
    smooth_sec: float = 30.0,
    target_rms_db: float = -20.0,
    max_gain_db: float = 10.0,
) -> AudioSegment:
    import wave as _wave
    sr = seg.frame_rate
    ch = seg.channels

    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float64) / 32768.0
    data = raw.reshape(-1, ch)
    mono = data.mean(axis=1)

    frame_samples = int(sr * frame_ms / 1000)
    n_frames = max(1, math.ceil(len(mono) / frame_samples))

    meter = pyln.Meter(sr)
    rms_db = np.full(n_frames, -60.0)
    for i in range(n_frames):
        chunk = data[i * frame_samples : (i + 1) * frame_samples]
        if len(chunk) >= frame_samples // 2:
            lufs = meter.integrated_loudness(chunk)
            if np.isfinite(lufs) and lufs > -70.0:
                rms_db[i] = lufs

    sigma_frames = max(1.0, smooth_sec * 1000.0 / frame_ms)
    half_w = int(4 * sigma_frames)
    x = np.arange(-half_w, half_w + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_frames) ** 2)
    kernel /= kernel.sum()
    rms_smoothed = np.convolve(rms_db, kernel, mode="same")

    gain_db_frames = np.clip(target_rms_db - rms_smoothed, -max_gain_db, max_gain_db)

    frame_centers = (np.arange(n_frames) + 0.5) * frame_samples
    gain_db_samples = np.interp(np.arange(len(mono), dtype=np.float64), frame_centers, gain_db_frames)
    gain_lin = 10.0 ** (gain_db_samples / 20.0)

    result = np.clip(data * gain_lin[:, np.newaxis], -1.0, 1.0)
    pcm = np.ascontiguousarray((result * 32767.0).astype(np.int16))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with _wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return AudioSegment.from_file(tmp.name)


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
    factor = max(1e-6, cur_ms2 / float(target_ms))

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


def _rms_dbfs_from_samples(samples, sample_width: int) -> float:
    n = len(samples)
    if n == 0:
        return -120.0
    sum_sq = 0.0
    for s in samples:
        sum_sq += s * s
    rms = math.sqrt(sum_sq / n)
    if rms <= 0:
        return -120.0
    full_scale = float(1 << (8 * sample_width - 1))
    return 20.0 * math.log10(rms / full_scale)


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
    sw = music.sample_width
    channels = music.channels
    sr = music.frame_rate

    if sw == 1:
        typecode = "b"
        maxv = 127
    elif sw == 2:
        typecode = "h"
        maxv = 32767
    elif sw == 4:
        typecode = "i"
        maxv = 2147483647
    else:
        typecode = "h"
        maxv = 32767

    minv = -(maxv + 1)

    m_samples = _array.array(typecode, music.raw_data)
    v_samples = _array.array(typecode, voice.raw_data)

    frames_per_win = int(sr * win / 1000)
    spw = frames_per_win * channels
    total_samples = len(m_samples)
    total_wins = max(1, (total_samples + spw - 1) // spw)

    lookahead_wins = max(1, int(round(lookahead_ms / win)))

    voice_rms: list[float] = []
    for w in range(total_wins):
        s0 = min(w * spw, len(v_samples))
        s1 = min(s0 + spw, len(v_samples))
        if s1 > s0:
            voice_rms.append(_rms_dbfs_from_samples(v_samples[s0:s1], sw))
        else:
            voice_rms.append(-120.0)

    silence_threshold_db = -45.0
    in_voice_region = False
    silence_run_ms = 0
    prev_gain = 0.0
    gains_db: list[float] = []

    for w in range(total_wins):
        v_now_db = voice_rms[w]
        voice_now = v_now_db > silence_threshold_db

        la_w = min(w + lookahead_wins, len(voice_rms) - 1)
        v_la_db = voice_rms[la_w]

        if voice_now:
            in_voice_region = True
            silence_run_ms = 0
        else:
            if in_voice_region:
                silence_run_ms += win
                if silence_run_ms >= gap_hold_ms:
                    in_voice_region = False
            else:
                silence_run_ms = 0

        if v_la_db <= -48.0:
            target = floor_boost_db
        elif v_la_db >= -26.0:
            target = max_duck_db
        else:
            t = (v_la_db + 48.0) / 22.0
            target = max_duck_db * t + floor_boost_db * (1 - t)

        short_gap = (not voice_now) and in_voice_region and (silence_run_ms < gap_hold_ms)
        if short_gap:
            hold_level = max(max_duck_db + 1.5, -1.5)
            target = min(target, hold_level)

        if target < prev_gain:
            alpha = min(1.0, win / float(max(1, attack_ms)))
        else:
            alpha = min(1.0, win / float(max(1, release_ms)))
        gain = prev_gain + alpha * (target - prev_gain)
        prev_gain = gain

        gains_db.append(gain)

    gains_lin: list[float] = [10.0 ** (g / 20.0) for g in gains_db]

    for w in range(total_wins):
        s0 = w * spw
        s1 = min(s0 + spw, total_samples)
        n = s1 - s0
        if n <= 0:
            continue

        g0 = gains_lin[w]
        g1 = gains_lin[min(w + 1, len(gains_lin) - 1)]

        if abs(g0 - g1) < 1e-9:
            if abs(g0 - 1.0) < 1e-9:
                continue
            for j in range(n):
                val = int(m_samples[s0 + j] * g0)
                if val > maxv:
                    val = maxv
                elif val < minv:
                    val = minv
                m_samples[s0 + j] = val
        else:
            inv_n = 1.0 / n
            for j in range(n):
                g = g0 + (g1 - g0) * (j * inv_n)
                val = int(m_samples[s0 + j] * g)
                if val > maxv:
                    val = maxv
                elif val < minv:
                    val = minv
                m_samples[s0 + j] = val

    return music._spawn(m_samples.tobytes())


def mix(
    voice_path: str | Path,
    music_path: str | Path,
    out_path: str | Path,
    duck_db: float = 4.0,
    sync_mode: Literal["retime_voice_to_music", "retime_music_to_voice", "no_retime_trim_pad"] = "retime_voice_to_music",
    voice_target_dbfs: float = -13.0,
    music_target_dbfs: float = -14.5,
    final_peak_dbfs: float = -1.0,
    music_fadein_ms: int = 10,
    music_premix_gain_db: float = -0.5,
    ffmpeg_bin: str | None = None,
    stems_dir: str | Path | None = None,
    **_ignored,
) -> int:

    ffmpeg_path = _ffmpeg_bin(ffmpeg_bin)

    music = make_stereo(load_audio(music_path).set_frame_rate(44100))
    music = _normalize_lufs(music, -10.0)
    music = _level_envelope(music, target_rms_db=-15.0, max_gain_db=50.0, smooth_sec=5.0)
    if len(music) <= 0:
        raise ValueError("Music stem is empty or unreadable.")

    if stems_dir is not None:
        Path(stems_dir).mkdir(parents=True, exist_ok=True)
        music.export(str(Path(stems_dir) / "stem_music_after_envelope.mp3"), format="mp3", bitrate="256k")

    if music_fadein_ms > 0:
        music = music.fade_in(music_fadein_ms)

    if _ffmpeg_has(ffmpeg_path, "dynaudnorm") or _ffmpeg_has(ffmpeg_path, "acompressor"):
        tmp_m2_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_m2_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        music.export(tmp_m2_in.name, format="wav")
        try:
            if _ffmpeg_has(ffmpeg_path, "dynaudnorm") and _ffmpeg_has(ffmpeg_path, "acompressor"):
                af = (
                    "dynaudnorm=f=3000:g=31:p=0.5:m=15,"
                    "acompressor=threshold=-20dB:ratio=3:attack=300:release=3000:makeup=5,"
                    "alimiter=level_in=1:level_out=0.89:limit=0.89:attack=5:release=50"
                )
            elif _ffmpeg_has(ffmpeg_path, "dynaudnorm"):
                af = "dynaudnorm=f=3000:g=31:p=0.5:m=15,alimiter=level_in=1:level_out=0.89:limit=0.89:attack=5:release=50"
            else:
                af = "acompressor=threshold=-20dB:ratio=3:attack=300:release=3000:makeup=5,alimiter=level_in=1:level_out=0.89:limit=0.89:attack=5:release=50"
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_m2_in.name, "-af", af, tmp_m2_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            music = AudioSegment.from_file(tmp_m2_out.name)
        except Exception:
            pass

    music = _apply_reverb(music, wet_boost_db=8.0)
    music = _normalize_lufs(music, -15.0)

    if stems_dir is not None:
        Path(stems_dir).mkdir(parents=True, exist_ok=True)
        music.export(str(Path(stems_dir) / "stem_music.mp3"), format="mp3", bitrate="256k")

    voice = make_stereo(load_audio(voice_path).set_frame_rate(44100))

    if _ffmpeg_has(ffmpeg_path, "dynaudnorm"):
        tmp_vl_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_vl_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice.export(tmp_vl_in.name, format="wav")
        try:
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_vl_in.name, "-af",
                 "dynaudnorm=f=800:g=15:p=0.5:m=10",
                 tmp_vl_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            voice = AudioSegment.from_file(tmp_vl_out.name).set_frame_rate(44100).set_channels(2)
        except Exception:
            pass

    voice = _normalize_lufs(voice, -15.0)
    if len(voice) <= 0:
        raise ValueError("Voice stem is empty or unreadable.")

    if _ffmpeg_has(ffmpeg_path, "acompressor") or _ffmpeg_has(ffmpeg_path, "dynaudnorm"):
        tmp_v_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_v_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice.export(tmp_v_in.name, format="wav")
        try:
            if _ffmpeg_has(ffmpeg_path, "acompressor"):
                vf = "acompressor=threshold=-15dB:ratio=4:attack=10:release=150:makeup=1"
            else:
                vf = "dynaudnorm=f=125:s=12"
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_v_in.name, "-af", vf, tmp_v_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            voice = AudioSegment.from_file(tmp_v_out.name).set_frame_rate(44100).set_channels(2)
        except Exception:
            pass

    if _ffmpeg_has(ffmpeg_path, "equalizer") and _ffmpeg_has(ffmpeg_path, "highpass"):
        tmp_veq_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_veq_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        voice.export(tmp_veq_in.name, format="wav")
        try:
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_veq_in.name, "-af",
                 "highpass=f=80:p=2,lowshelf=f=250:g=10,equalizer=f=770:t=q:w=0.5:g=-3",
                 tmp_veq_out.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            voice = AudioSegment.from_file(tmp_veq_out.name).set_frame_rate(44100).set_channels(2)
        except Exception:
            pass

    if stems_dir is not None:
        voice.export(str(Path(stems_dir) / "stem_voice.mp3"), format="mp3", bitrate="256k")

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

    music_exact = music_exact.apply_gain(music_premix_gain_db)

    music_adapt = _duck_music_to_voice(
        music_exact,
        voice_exact,
        floor_boost_db=0.0,
        max_duck_db=-4.0,
        attack_ms=30,
        release_ms=1500,
        win_ms=30,
        lookahead_ms=150,
        gap_hold_ms=1500,
    )

    final_mix = music_adapt.overlay(voice_exact)

    final_mix = _apply_peak_guard(final_mix, ceiling_dbfs=final_peak_dbfs)
    final_mix = _hard_fit_samples(final_mix, target_samples_per_ch)

    tail_ms = min(900, max(350, target_ms // 18))
    final_mix = final_mix.fade_out(tail_ms)

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
