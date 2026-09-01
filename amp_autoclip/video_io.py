"""Video probing, decoding, and cheap motion gating.

Stage 1 of the pipeline: frame-difference motion energy finds candidate
active windows so the pose model only runs where something is happening.
Gating is deliberately loose (padded, low percentile) — its job is to skip
minutes of dead net time, never to make boundary decisions.
"""
from __future__ import annotations

import json
import subprocess
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from .config import EngineConfig
from .types import VideoInfo


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    vstream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in meta["streams"])
    num, den = vstream.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(meta["format"].get("duration", 0.0))
    n_frames = int(vstream.get("nb_frames", 0)) or int(duration * fps)
    return VideoInfo(path=path, fps=fps, duration_s=duration,
                     width=int(vstream["width"]), height=int(vstream["height"]),
                     n_frames=n_frames, has_audio=has_audio)


def iter_frames(path: str, target_fps: float,
                downscale_to: int = 0) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_s, frame_bgr) at ~target_fps, optionally downscaled
    so the long side is `downscale_to` px."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if downscale_to:
                    h, w = frame.shape[:2]
                    scale = downscale_to / max(h, w)
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                yield idx / src_fps, frame
            idx += 1
    finally:
        cap.release()


def motion_energy(path: str, sample_fps: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample motion energy (mean abs frame difference on downscaled gray).
    Returns (timestamps, energy)."""
    ts, energy = [], []
    prev = None
    for t, frame in iter_frames(path, sample_fps, downscale_to=160):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev is not None:
            ts.append(t)
            energy.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    return np.asarray(ts), np.asarray(energy)


def detect_scene_cuts(path: str, cfg: EngineConfig,
                      sample_fps: float = 12.0) -> List[float]:
    """Hard camera cuts (edited/broadcast footage). A raw phone session has
    none; when cuts exist, clip windows must never span one — the frames on
    the far side belong to a different shot."""
    ts, energy = motion_energy(path, sample_fps)
    if len(energy) < 10:
        return []
    med = np.median(energy)
    mad = np.median(np.abs(energy - med)) + 1e-9
    thresh = med + cfg.segmenter.scene_cut_mad_k * mad
    cuts: List[float] = []
    for t, e in zip(ts, energy):
        if e > thresh and (not cuts or t - cuts[-1] > 0.5):
            cuts.append(float(t))
    return cuts


def active_windows(path: str, cfg: EngineConfig,
                   duration_s: float) -> List[Tuple[float, float]]:
    """Candidate windows worth running pose on, padded outward and merged."""
    ts, energy = motion_energy(path)
    if len(energy) == 0:
        return [(0.0, duration_s)]
    thresh = max(np.percentile(energy, cfg.segmenter.motion_gate_percentile),
                 cfg.segmenter.motion_gate_abs_floor)
    pad = cfg.segmenter.motion_gate_pad_s
    windows: List[Tuple[float, float]] = []
    for t, e in zip(ts, energy):
        if e <= thresh:
            continue
        lo, hi = max(0.0, t - pad), min(duration_s, t + pad)
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], hi)
        else:
            windows.append((lo, hi))
    return windows or [(0.0, duration_s)]
