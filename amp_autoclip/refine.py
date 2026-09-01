"""Fine pass: native-fps anchor refinement around each detected delivery.

The coarse pass finds deliveries at ~20fps, which is plenty for detection but
coarse for the anchors the KPI model scores off (release/impact are sub-100ms
events). This pass re-runs pose at native fps in a small window around each
kept delivery's key anchor and re-times the anchors from the dense series.
On a 1-hour session this touches a few minutes of footage instead of sixty.

Audio-derived impact times are never overwritten — a clean transient is more
precise than any frame-based estimate at any fps.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import EngineConfig
from .events_bowling import _foot_contacts
from .features import compute_features
from .pose import Track, extract_tracks
from .types import SessionResult

log = logging.getLogger("amp_autoclip.refine")


def _best_track(tracks, t: float) -> Optional[Track]:
    covering = [tr for tr in tracks if tr.t0 <= t <= tr.t1]
    if not covering:
        return None
    return max(covering, key=lambda tr: len(tr.frames))


def refine_anchors(video_path: str, result: SessionResult,
                   cfg: EngineConfig) -> None:
    """Mutates anchor times (and marks them refined) in place."""
    video = result.video
    fine_fps = min(video.fps, cfg.pose.max_process_fps)
    if fine_fps <= cfg.pose.detect_fps + 1:
        return   # coarse pass already ran at native rate
    for d in result.deliveries:
        if d.status == "discard":
            continue
        key_name = "ball_release" if d.type == "bowling" else "impact"
        key = d.anchor(key_name)
        if key is None:
            continue
        lo = max(0.0, key.t - cfg.pose.refine_pre_s)
        hi = min(video.duration_s, key.t + cfg.pose.refine_post_s)
        tracks = extract_tracks(video_path, [(lo, hi)], cfg, video.fps,
                                (video.width, video.height),
                                detect_fps=fine_fps, min_track_frames=3)
        track = _best_track(tracks, key.t)
        if track is None:
            continue
        fs = compute_features(track, fine_fps)
        if fs is None:
            continue
        speed = np.maximum(fs.wrist_speed_l, fs.wrist_speed_r)
        # re-time the key anchor at the dense peak nearest the coarse estimate
        near = np.abs(fs.t - key.t) <= 0.4
        if not np.any(near):
            continue
        idx = np.where(near)[0]
        peak_i = idx[int(np.argmax(speed[idx]))]
        refined_t = float(fs.t[peak_i])
        coarse_t = key.t
        if key.source == "pose":         # audio-pinned times stay
            key.t = refined_t
        key.source = key.source + "+refined"
        if d.type == "bowling":
            bfc_t, ffc_t = _foot_contacts(fs, int(peak_i), cfg)
            for name, t in (("back_foot_contact", bfc_t),
                            ("front_foot_contact", ffc_t)):
                a = d.anchor(name)
                if a is not None and t is not None:
                    a.t = t
                    a.source = "pose+refined"
        d.notes.append("anchors_refined")
        log.info("refined %s anchor %.3f -> %.3f", d.type, coarse_t, refined_t)
