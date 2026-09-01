"""Engine entrypoint: video file in, SessionResult + clips out.

run_session() is the single function the job worker calls. It is pure
post-hoc processing — the video is assumed fully uploaded and sitting on
disk (the worker downloads from cloud storage first). No streaming, no
on-device constraints.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from . import audio as audio_mod
from . import video_io
from .config import DEFAULT, EngineConfig
from .pose import extract_tracks
from .segmenter import segment_session
from .types import SessionResult

log = logging.getLogger("amp_autoclip")

ProgressFn = Callable[[str, float], None]   # (stage_name, fraction 0..1)


def run_session(video_path: str, mode: str = "auto",
                cfg: EngineConfig = DEFAULT,
                progress: Optional[ProgressFn] = None) -> SessionResult:
    def report(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)
        log.info("[%s] %.0f%%", stage, frac * 100)

    report("probe", 0.02)
    video = video_io.probe(video_path)

    report("motion_gate", 0.08)
    windows = video_io.active_windows(video_path, cfg, video.duration_s)
    scene_cuts = video_io.detect_scene_cuts(video_path, cfg)

    report("audio", 0.15)
    transients = (audio_mod.detect_transients(video_path, cfg)
                  if video.has_audio else [])

    report("pose", 0.25)
    tracks = extract_tracks(video_path, windows, cfg, video.fps,
                            (video.width, video.height))

    report("segment", 0.80)
    out = segment_session(video, tracks, transients, cfg, mode=mode,
                          scene_cuts=scene_cuts)
    result = SessionResult(video=video, deliveries=out.deliveries,
                           session_flags=out.flags)

    if cfg.pose.refine:
        report("refine", 0.88)
        from .refine import refine_anchors
        refine_anchors(video_path, result, cfg)

    report("done", 1.0)
    return result
