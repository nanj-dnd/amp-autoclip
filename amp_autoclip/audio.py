"""Audio transient channel — corroboration only, never a sole trigger.

Bat-ball contact and ball-on-pitch are sharp broadband transients. Audio
samples at ~44kHz against video at 30-60fps, so an aligned transient pins
impact timing far more precisely than any frame-based estimate. It fails
in multi-lane nets (other lanes' impacts), wind, and shouting — which is
exactly why it only ever adjusts confidence and snaps timing on events the
pose channel already proposed.
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal as sp_signal

from .config import EngineConfig


@dataclass
class AudioTransient:
    t: float           # seconds from video start
    strength: float    # peak onset strength / threshold (>= 1.0)


def extract_pcm(video_path: str, sr: int) -> Optional[np.ndarray]:
    """Mono float32 PCM via ffmpeg; None when the video has no audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", video_path,
             "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", tmp.name],
            capture_output=True)
        if proc.returncode != 0:
            return None
        try:
            with wave.open(tmp.name, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                width = wf.getsampwidth()
        except (wave.Error, EOFError):
            return None
    if width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        return None
    return pcm if len(pcm) > sr // 10 else None


def detect_transients(video_path: str, cfg: EngineConfig) -> List[AudioTransient]:
    """Band-limited spectral-flux onsets, thresholded at median + k*MAD.
    The adaptive threshold keeps this usable across quiet backyards and
    loud academies without per-venue tuning."""
    acfg = cfg.audio
    pcm = extract_pcm(video_path, acfg.sample_rate)
    if pcm is None:
        return []
    lo, hi = acfg.band_hz
    nyq = acfg.sample_rate / 2
    b, a = sp_signal.butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
    band = sp_signal.filtfilt(b, a, pcm)

    hop = acfg.sample_rate // 200          # 5 ms hop
    win = hop * 4
    n = (len(band) - win) // hop
    if n <= 10:
        return []
    env = np.empty(n)
    for i in range(n):
        env[i] = np.sqrt(np.mean(band[i * hop:i * hop + win] ** 2))
    flux = np.maximum(np.diff(env, prepend=env[0]), 0.0)

    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    thresh = med + acfg.onset_mad_k * mad
    peaks, _ = sp_signal.find_peaks(flux, height=thresh,
                                    distance=max(1, int(0.1 / (hop / acfg.sample_rate))))
    return [AudioTransient(t=float(p * hop / acfg.sample_rate),
                           strength=float(flux[p] / thresh)) for p in peaks]


def nearest_transient(transients: List[AudioTransient], t: float,
                      window_ms: int) -> Optional[AudioTransient]:
    best, best_dt = None, window_ms / 1000.0
    for tr in transients:
        dt = abs(tr.t - t)
        if dt <= best_dt:
            best, best_dt = tr, dt
    return best
