"""ffmpeg clip cutting + manifest writing.

Output contract: mp4/h264 clips, one per delivery, identical container to
what the existing frame-extraction step already consumes — the downstream
pipeline's input contract is unchanged. Re-encode (not stream copy) because
stream copy snaps the cut to the previous keyframe and can shift the start
by a whole GOP, which would silently violate the padding guarantees.
Timestamps live in the manifest so clips can be re-cut with different
padding without re-running detection.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import List

from .config import EngineConfig
from .types import Delivery, SessionResult


def cut_clip(src: str, dst: str, start_t: float, end_t: float,
             cfg: EngineConfig) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", "%.3f" % start_t, "-to", "%.3f" % end_t, "-i", src]
    if cfg.clip.reencode:
        cmd += ["-c:v", "libx264", "-preset", cfg.clip.preset,
                "-crf", str(cfg.clip.crf), "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += (["-c:a", "aac"] if cfg.clip.keep_audio else ["-an"])
    cmd += ["-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def write_outputs(result: SessionResult, out_dir: str,
                  cfg: EngineConfig, cut_files: bool = True) -> dict:
    """Cut clips for non-discarded deliveries and write manifest.json.
    Review-status deliveries ARE cut (the review UI shows the candidate clip)
    but are marked so they never flow to KPI scoring unreviewed.
    Returns the manifest dict (with clip_file fields) that was written."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = result.to_manifest()
    n = 0
    for d, md in zip(result.deliveries, manifest["deliveries"]):
        if d.status == "discard":
            md["clip_file"] = None
            continue
        n += 1
        name = "delivery_%02d_%s.mp4" % (n, d.type)
        md["clip_file"] = name
        if cut_files:
            cut_clip(result.video.path, os.path.join(out_dir, name),
                     d.start_t, d.end_t, cfg)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
