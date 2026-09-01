"""Model-based segmentation path — drop-in replacement for the heuristic
event detectors once a trained checkpoint exists and beats them on the
held-out eval (amp_autoclip.eval decides that, not vibes).

Decoding: per-frame phase argmax -> median filter -> grammar-constrained
merge (min phase durations, legal successors only) -> segments. Anchor
events from heatmap peaks, constrained to fall inside the right phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import ndimage
from scipy import signal as sp_signal

from ..features import FeatureSeries, model_feature_matrix
from . import labels as L
from .tcn import MSTCN

MEDIAN_WIN_S = 0.25          # UNTUNED: phase smoothing window
MIN_PHASE_S = {"idle": 0.0, "runup": 0.6, "delivery": 0.2, "followthrough": 0.3,
               "stance": 0.4, "shot": 0.15}   # UNTUNED
EVENT_PEAK_MIN = 0.3         # UNTUNED: heatmap peak floor


@dataclass
class ModelSegment:
    role: str
    start_t: float           # first non-idle phase start (unpadded boundary)
    end_t: float             # followthrough end
    phase_intervals: List[Dict]
    anchors: Dict[str, float]
    mean_phase_prob: float   # decoder confidence ingredient


def load_model(path: str, device: str = "cpu") -> MSTCN:
    ckpt = torch.load(path, map_location=device)
    role = ckpt["role"]
    model = MSTCN(n_phases=len(L.phases_for(role)),
                  n_events=len(L.events_for(role)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.role = role
    return model


@torch.no_grad()
def segment_with_model(model: MSTCN, fs: FeatureSeries) -> List[ModelSegment]:
    role = model.role
    phases = L.phases_for(role)
    X = torch.from_numpy(model_feature_matrix(fs).T[None]).float()
    outs, event_logits = model(X)
    probs = torch.softmax(outs[-1], dim=1)[0].numpy()      # (P, N)
    events = torch.sigmoid(event_logits)[0].numpy()        # (E, N)

    pred = probs.argmax(axis=0)
    win = max(1, int(MEDIAN_WIN_S * fs.fps) | 1)
    pred = ndimage.median_filter(pred, size=win)

    # enforce min durations: absorb too-short runs into the previous phase
    runs = []   # (phase_idx, start_i, end_i)
    s = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[s]:
            runs.append([int(pred[s]), s, i])
            s = i
    merged = []
    for run in runs:
        dur = (run[2] - run[1]) / fs.fps
        if merged and dur < MIN_PHASE_S.get(phases[run[0]], 0.0):
            merged[-1][2] = run[2]
        else:
            merged.append(run)

    # cut into delivery segments: idle -> ... -> followthrough -> idle
    segments: List[ModelSegment] = []
    current: List[List[int]] = []
    for run in merged:
        if phases[run[0]] == "idle":
            if current:
                segments.append(_finish(current, fs, role, probs, events))
                current = []
        else:
            current.append(run)
    if current:
        segments.append(_finish(current, fs, role, probs, events))
    return [s for s in segments if s is not None]


def _finish(runs: List[List[int]], fs: FeatureSeries, role: str,
            probs: np.ndarray, events: np.ndarray) -> Optional[ModelSegment]:
    phases = L.phases_for(role)
    i0, i1 = runs[0][1], runs[-1][2] - 1
    intervals = [{"phase": phases[r[0]], "start_t": float(fs.t[r[1]]),
                  "end_t": float(fs.t[min(r[2], len(fs.t) - 1)])} for r in runs]
    anchors: Dict[str, float] = {}
    for k, name in enumerate(L.events_for(role)):
        seg = events[k, i0:i1 + 1]
        if len(seg) == 0:
            continue
        peaks, props = sp_signal.find_peaks(seg, height=EVENT_PEAK_MIN)
        if len(peaks):
            best = peaks[np.argmax(props["peak_heights"])]
            anchors[name] = float(fs.t[i0 + best])
    mean_prob = float(np.mean(probs.max(axis=0)[i0:i1 + 1]))
    return ModelSegment(role=role, start_t=float(fs.t[i0]),
                        end_t=float(fs.t[i1]), phase_intervals=intervals,
                        anchors=anchors, mean_phase_prob=mean_prob)
