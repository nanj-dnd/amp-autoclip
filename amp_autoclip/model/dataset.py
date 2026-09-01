"""Dataset assembly: labeler JSONL + cached pose features -> training tensors.

Layout expected under a dataset root:
    features/<video_id>.npz      # from cache_features(): X (N,85), t (N,), fps
    labels/<video_id>.jsonl      # one labeled delivery per line (labeler output)

Label line schema (written by labeler/, validated against model.labels):
    {"video_id": ..., "role": "bowling"|"batting",
     "start_t": float, "end_t": float,            # true boundaries, unpadded
     "phases": [{"phase": "runup", "start_t": ..., "end_t": ...}, ...],
     "anchors": {"ball_release": t, "back_foot_contact": t, ...},
     "shot_type": "swing"|"leave"|null, "verified_by": "hardik", ...}

Splits are BY SESSION VIDEO, never by clip — clips from one session share
lighting/camera/players and would leak across splits.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..features import FeatureSeries, model_feature_matrix
from . import labels as L


def cache_features(fs: FeatureSeries, video_id: str, root: str) -> str:
    os.makedirs(os.path.join(root, "features"), exist_ok=True)
    path = os.path.join(root, "features", video_id + ".npz")
    np.savez_compressed(path, X=model_feature_matrix(fs).astype(np.float32),
                        t=fs.t.astype(np.float64), fps=fs.fps)
    return path


@dataclass
class Sample:
    video_id: str
    role: str
    X: np.ndarray            # (N, 85)
    phase_targets: np.ndarray  # (N,) int, -100 = unlabeled
    event_targets: np.ndarray  # (n_events, N)


def _phase_index(role: str, name: str) -> int:
    return L.phases_for(role).index(name)


def load_samples(root: str, role: str) -> List[Sample]:
    samples: List[Sample] = []
    labels_dir = os.path.join(root, "labels")
    if not os.path.isdir(labels_dir):
        return samples
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".jsonl"):
            continue
        video_id = fn[:-6]
        feat_path = os.path.join(root, "features", video_id + ".npz")
        if not os.path.exists(feat_path):
            continue
        data = np.load(feat_path)
        X, t = data["X"], data["t"]
        n = len(t)
        phase = np.full(n, -100, dtype=np.int64)
        event_frames: Dict[str, Optional[int]] = {}
        any_label = False
        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("role") != role:
                    continue
                any_label = True
                # frames outside any labeled delivery but inside the labeled
                # region default to idle
                lo = np.searchsorted(t, rec["start_t"] - 3.0)
                hi = np.searchsorted(t, rec["end_t"] + 3.0)
                phase[lo:hi] = _phase_index(role, "idle")
                for ph in rec.get("phases", []):
                    a = np.searchsorted(t, ph["start_t"])
                    b = np.searchsorted(t, ph["end_t"])
                    phase[a:b] = _phase_index(role, ph["phase"])
                for name, at in (rec.get("anchors") or {}).items():
                    if at is not None and name in L.events_for(role):
                        event_frames[name] = int(np.searchsorted(t, at))
        if not any_label:
            continue
        samples.append(Sample(
            video_id=video_id, role=role, X=X, phase_targets=phase,
            event_targets=L.event_heatmap(n, event_frames, role)))
    return samples


def session_key(video_id: str) -> str:
    """Session a sequence belongs to: "part_3__t12" -> "part_3"."""
    return video_id.rsplit("__t", 1)[0]


def session_split(samples: List[Sample], val_frac: float = 0.15,
                  test_frac: float = 0.15) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Deterministic session-level split, so the split is stable as new labels
    arrive and no session ever straddles splits.

    Hashes the SESSION key, not video_id: both dataset builders emit one
    sequence per (video, track) as "<stem>__t<track>", so hashing video_id
    would scatter the tracks of a single session across train/val/test —
    the exact clip-level leak this function exists to prevent."""
    train, val, test = [], [], []
    for s in samples:
        h = int(hashlib.sha1(session_key(s.video_id).encode()).hexdigest(), 16) % 100
        if h < test_frac * 100:
            test.append(s)
        elif h < (test_frac + val_frac) * 100:
            val.append(s)
        else:
            train.append(s)
    return train, val, test
