"""Label space for the temporal action segmentation model.

Per-frame PHASE classification plus per-frame EVENT heatmaps. Phases carry
the segmentation; near-instant anchor events (BFC/FFC/release/impact) are
ill-posed as phases, so they get their own regression channels with
Gaussian targets around the labeled frame — the same way spatial keypoint
detectors localize joints, applied to time.

This file is also the labeling rubric's machine-readable half: the labeler
tool and the dataset builder both import these constants, so labels can
only ever be produced against exactly this space.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# ---- phase label spaces (per-frame classification) ----
BOWLING_PHASES: List[str] = ["idle", "runup", "delivery", "followthrough"]
# "delivery" = gather/bound through BFC, FFC, release
BATTING_PHASES: List[str] = ["idle", "stance", "shot", "followthrough"]
# "stance" = set stance through trigger; "shot" = backlift through impact

# legal phase successors — enforced by the decoder, so the model can never
# emit a grammatically impossible sequence (e.g. followthrough -> delivery)
BOWLING_GRAMMAR: Dict[int, List[int]] = {0: [0, 1], 1: [1, 2, 0], 2: [2, 3],
                                         3: [3, 0]}
BATTING_GRAMMAR: Dict[int, List[int]] = {0: [0, 1], 1: [1, 2, 0], 2: [2, 3],
                                         3: [3, 0]}

# ---- event channels (per-frame heatmap regression) ----
BOWLING_EVENTS: List[str] = ["back_foot_contact", "front_foot_contact",
                             "ball_release"]
BATTING_EVENTS: List[str] = ["stance_set", "impact"]

EVENT_SIGMA_FRAMES = 2.0   # Gaussian target width around the labeled frame


def phases_for(role: str) -> List[str]:
    return BOWLING_PHASES if role == "bowling" else BATTING_PHASES


def events_for(role: str) -> List[str]:
    return BOWLING_EVENTS if role == "bowling" else BATTING_EVENTS


def grammar_for(role: str) -> Dict[int, List[int]]:
    return BOWLING_GRAMMAR if role == "bowling" else BATTING_GRAMMAR


def event_heatmap(n_frames: int, event_frames: Dict[str, int],
                  role: str) -> np.ndarray:
    """(n_events, N) Gaussian heatmaps; all-zero rows for unlabeled events."""
    names = events_for(role)
    hm = np.zeros((len(names), n_frames), dtype=np.float32)
    idx = np.arange(n_frames)
    for k, name in enumerate(names):
        if name in event_frames and event_frames[name] is not None:
            f = event_frames[name]
            hm[k] = np.exp(-0.5 * ((idx - f) / EVENT_SIGMA_FRAMES) ** 2)
    return hm
