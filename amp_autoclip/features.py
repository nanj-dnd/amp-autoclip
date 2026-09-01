"""Kinematic feature series derived from a pose track.

Everything downstream (event detection, the temporal model, confidence)
consumes these series rather than raw keypoints. All speeds are normalized
by torso length so thresholds transfer across camera distance; the frame
axis is resampled to a uniform grid so derivatives are well-behaved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .pose import Track
from .types import KP


@dataclass
class FeatureSeries:
    t: np.ndarray                 # (N,) uniform time grid, seconds
    fps: float                    # grid rate
    kpts: np.ndarray              # (N, 17, 2) interpolated pixel coords
    conf: np.ndarray              # (N, 17)
    torso_len: float              # median shoulder-hip distance, px (scale unit)
    # derived series, all (N,):
    pelvis_x: np.ndarray          # hip midpoint x, px
    pelvis_y: np.ndarray          # hip midpoint y, px
    pelvis_speed_x: np.ndarray    # signed, torso-lengths/s (+ = image right)
    pelvis_speed_y: np.ndarray    # signed, torso-lengths/s (+ = image down)
    torso_scale: np.ndarray       # per-frame torso length, px (smoothed) —
                                  # grows/shrinks as the subject approaches/
                                  # recedes from the camera
    scale_rate: np.ndarray        # d(torso_scale)/dt / torso_len, 1/s
    centroid_speed: np.ndarray    # unsigned pelvis speed, torso-lengths/s
    wrist_speed_l: np.ndarray     # torso-lengths/s
    wrist_speed_r: np.ndarray
    wrist_above_shoulder_l: np.ndarray   # bool-ish 0/1
    wrist_above_shoulder_r: np.ndarray
    ankle_y_l: np.ndarray         # px (image y grows downward)
    ankle_y_r: np.ndarray
    ankle_vy_l: np.ndarray        # px/s, + = moving down
    ankle_vy_r: np.ndarray
    ankle_still_std: np.ndarray   # rolling std of both ankles, frac of torso
    trunk_lean: np.ndarray        # |angle of shoulder-mid->hip-mid from vertical|, rad
    trunk_rot_vel: np.ndarray     # |d/dt shoulder-line angle|, rad/s
    motion_energy: np.ndarray     # mean keypoint speed, torso-lengths/s


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win < 3 or len(x) < win:
        return x
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < win:
        return np.full_like(x, np.std(x))
    out = np.empty_like(x)
    half = win // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = np.std(x[lo:hi])
    return out


def compute_features(track: Track, grid_fps: float) -> Optional[FeatureSeries]:
    """Resample a track to a uniform grid and derive kinematic series.
    Returns None when the track is too sparse to be usable."""
    frames = track.frames
    if len(frames) < 8:
        return None
    ts = np.array([f.t for f in frames])
    K = np.stack([f.kpts for f in frames])       # (M, 17, 2)
    C = np.stack([f.conf for f in frames])       # (M, 17)

    grid = np.arange(ts[0], ts[-1], 1.0 / grid_fps)
    if len(grid) < 8:
        return None
    kpts = np.empty((len(grid), 17, 2))
    conf = np.empty((len(grid), 17))
    for j in range(17):
        for d in range(2):
            kpts[:, j, d] = np.interp(grid, ts, K[:, j, d])
        conf[:, j] = np.interp(grid, ts, C[:, j])

    def mid(a: str, b: str) -> np.ndarray:
        return (kpts[:, KP[a]] + kpts[:, KP[b]]) / 2.0

    sh_mid, hip_mid = mid("l_shoulder", "r_shoulder"), mid("l_hip", "r_hip")
    torso = float(np.median(np.linalg.norm(sh_mid - hip_mid, axis=1)))
    if torso < 1e-3:
        return None

    def speed(pts: np.ndarray, smooth_win: int = 5) -> np.ndarray:
        sx = _smooth(pts[:, 0], smooth_win)
        sy = _smooth(pts[:, 1], smooth_win)
        v = np.hypot(np.gradient(sx, grid), np.gradient(sy, grid))
        return v / torso

    pelvis_x = _smooth(hip_mid[:, 0], 5)
    pelvis_y = _smooth(hip_mid[:, 1], 5)
    pelvis_speed_x = np.gradient(pelvis_x, grid) / torso
    pelvis_speed_y = np.gradient(pelvis_y, grid) / torso
    torso_scale = _smooth(np.linalg.norm(sh_mid - hip_mid, axis=1),
                          int(grid_fps * 0.7) | 1)
    scale_rate = np.gradient(torso_scale, grid) / torso
    centroid_speed = speed(hip_mid)

    wl, wr = kpts[:, KP["l_wrist"]], kpts[:, KP["r_wrist"]]
    sl, sr = kpts[:, KP["l_shoulder"]], kpts[:, KP["r_shoulder"]]
    al, ar = kpts[:, KP["l_ankle"]], kpts[:, KP["r_ankle"]]

    ankle_all = np.concatenate([al, ar], axis=1)          # (N, 4)
    ankle_dev = np.zeros(len(grid))
    for c in range(4):
        ankle_dev += _rolling_std(ankle_all[:, c], win=int(grid_fps * 0.5)) ** 2
    ankle_still_std = np.sqrt(ankle_dev) / torso

    trunk_vec = hip_mid - sh_mid
    trunk_lean = np.abs(np.arctan2(trunk_vec[:, 0], trunk_vec[:, 1] + 1e-9))
    sh_line = sl - sr
    sh_angle = np.unwrap(np.arctan2(sh_line[:, 1], sh_line[:, 0] + 1e-9))
    trunk_rot_vel = np.abs(np.gradient(_smooth(sh_angle, 5), grid))

    all_speed = np.zeros(len(grid))
    for j in range(17):
        all_speed += speed(kpts[:, j])
    motion_energy = all_speed / 17.0

    return FeatureSeries(
        t=grid, fps=grid_fps, kpts=kpts, conf=conf, torso_len=torso,
        pelvis_x=pelvis_x, pelvis_y=pelvis_y,
        pelvis_speed_x=pelvis_speed_x, pelvis_speed_y=pelvis_speed_y,
        torso_scale=torso_scale, scale_rate=scale_rate,
        centroid_speed=centroid_speed,
        wrist_speed_l=speed(wl), wrist_speed_r=speed(wr),
        wrist_above_shoulder_l=(wl[:, 1] < sl[:, 1]).astype(float),
        wrist_above_shoulder_r=(wr[:, 1] < sr[:, 1]).astype(float),
        ankle_y_l=al[:, 1], ankle_y_r=ar[:, 1],
        ankle_vy_l=np.gradient(_smooth(al[:, 1], 5), grid),
        ankle_vy_r=np.gradient(_smooth(ar[:, 1], 5), grid),
        ankle_still_std=ankle_still_std,
        trunk_lean=trunk_lean, trunk_rot_vel=trunk_rot_vel,
        motion_energy=motion_energy,
    )


def model_feature_matrix(fs: FeatureSeries) -> np.ndarray:
    """(N, F) per-frame feature matrix for the temporal model: normalized
    keypoint positions (pelvis-centered, torso-scaled), their velocities,
    and keypoint confidences. F = 17*2 + 17*2 + 17 = 85."""
    hip_mid = (fs.kpts[:, KP["l_hip"]] + fs.kpts[:, KP["r_hip"]]) / 2.0
    rel = (fs.kpts - hip_mid[:, None, :]) / fs.torso_len          # (N,17,2)
    vel = np.gradient(rel, fs.t, axis=0)
    return np.concatenate(
        [rel.reshape(len(fs.t), -1), vel.reshape(len(fs.t), -1), fs.conf],
        axis=1)
