"""Bowling boundary detection: run-up initiation -> follow-through resolution.

Anchor-first strategy per the spec: detect ball release (the strongest
signal), locate BFC/FFC just before it, then extend outward — backward
along the pelvis trajectory to find run-up onset, forward through the
stillness conditions to find follow-through resolution. Anchors let us
place the clip reliably even when the run-up start is ambiguous.

Phase model:
idle -> run-up -> gather -> BFC -> FFC -> release -> follow-through -> idle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal as sp_signal

from .config import EngineConfig
from .features import FeatureSeries
from .types import Anchor


@dataclass
class BowlingEvent:
    release_t: float
    release_conf: float
    arm: str                        # "l" | "r"
    bfc_t: Optional[float]
    ffc_t: Optional[float]
    raw_start_t: float              # run-up onset (unpadded)
    raw_end_t: float                # follow-through resolution (unpadded)
    start_source: str               # "runup" | "anchor_fallback"
    end_source: str                 # "resolution" | "capped"
    notes: List[str]

    def anchors(self) -> List[Anchor]:
        out = [Anchor("ball_release", self.release_t, self.release_conf)]
        if self.bfc_t is not None:
            out.append(Anchor("back_foot_contact", self.bfc_t, 0.5))
        if self.ffc_t is not None:
            out.append(Anchor("front_foot_contact", self.ffc_t, 0.5))
        return out


def _release_candidates(fs: FeatureSeries, cfg: EngineConfig) -> List[dict]:
    """Wrist-speed peaks with the arm raised above the shoulder — the
    approximation of maximum arm extension at the top of the arc."""
    bcfg = cfg.bowling
    min_gap = int(cfg.segmenter.min_event_gap_s * fs.fps)
    cands = []
    for arm, speed, raised in (("l", fs.wrist_speed_l, fs.wrist_above_shoulder_l),
                               ("r", fs.wrist_speed_r, fs.wrist_above_shoulder_r)):
        gated = speed * (raised > 0.5)
        peaks, props = sp_signal.find_peaks(
            gated, height=bcfg.release_min_wrist_speed,
            distance=max(1, min_gap))
        for p, h in zip(peaks, props["peak_heights"]):
            cands.append({"i": int(p), "t": float(fs.t[p]), "arm": arm,
                          "height": float(h)})
    # two arms can fire on the same action — keep the stronger within min gap
    cands.sort(key=lambda c: c["t"])
    merged: List[dict] = []
    for c in cands:
        if merged and c["t"] - merged[-1]["t"] < cfg.segmenter.min_event_gap_s:
            if c["height"] > merged[-1]["height"]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _foot_contacts(fs: FeatureSeries, release_i: int,
                   cfg: EngineConfig) -> (Optional[float], Optional[float]):
    """BFC and FFC: ankle ground plants in the delivery stride. A plant is a
    local maximum of ankle image-y (foot at its lowest) where vertical
    velocity settles. FFC is the last plant before release; BFC the last
    plant of the *other* foot before FFC."""
    win = int(cfg.bowling.foot_contact_window_s * fs.fps)
    lo = max(0, release_i - win)
    plants = []   # (t, foot)
    for foot, y in (("l", fs.ankle_y_l), ("r", fs.ankle_y_r)):
        seg = y[lo:release_i + 1]
        if len(seg) < 5:
            continue
        peaks, _ = sp_signal.find_peaks(seg, distance=max(1, int(fs.fps * 0.08)),
                                        prominence=fs.torso_len * 0.02)
        for p in peaks:
            plants.append((float(fs.t[lo + p]), foot))
    plants.sort()
    if not plants:
        return None, None
    ffc_t, ffc_foot = plants[-1]
    bfc_t = None
    for t, foot in reversed(plants[:-1]):
        # left/right ankles blur together at distance/low fps — a same-frame
        # "opposite foot" plant is keypoint noise, not a real BFC
        if foot != ffc_foot and t < ffc_t - 0.03:
            bfc_t = t
            break
    return bfc_t, ffc_t


def _approach_pos(fs: FeatureSeries, cfg: EngineConfig) -> np.ndarray:
    """(N, 3) image-space 'position' whose displacement measures approach:
    pelvis x, pelvis y (torso units) and integrated relative scale (weighted).
    A side-on run-up moves x; a front-on run-up grows scale; diagonals mix."""
    gain = cfg.bowling.approach_scale_gain
    dt = np.gradient(fs.t)
    return np.stack([
        fs.pelvis_x / fs.torso_len,
        fs.pelvis_y / fs.torso_len,
        np.cumsum(fs.scale_rate * dt) * gain,
    ], axis=1)


def _bowling_direction(pos: np.ndarray, fs: FeatureSeries,
                       anchor_i: int) -> Optional[np.ndarray]:
    """Unit vector of 'toward the crease' in approach space, taken from the
    late run-up just before the delivery-stride anchor — the bowler is
    guaranteed to be closing on the crease there, and the trunk is still
    upright. (The stride itself is unusable: trunk flexion at delivery
    shrinks the shoulder-hip scale proxy and corrupts the depth component.)
    Defining direction from the data rather than a camera assumption is what
    lets walking-back register as negative at any camera angle."""
    a = max(0, anchor_i - int(fs.fps * 2.0))
    b = max(0, anchor_i - int(fs.fps * 0.3))
    if b - a < 3:
        return None
    d = pos[b] - pos[a]
    n = float(np.linalg.norm(d))
    return d / n if n > 0.05 else None


def _runup_onset(fs: FeatureSeries, pos: np.ndarray, dir3: Optional[np.ndarray],
                 anchor_i: int, cfg: EngineConfig) -> Optional[float]:
    """Walk backward from the delivery-stride anchor while locomotion is
    sustained (stride cadence = whole-body motion energy, with gap tolerance
    across the gather), then validate that the candidate run-up's NET
    displacement projects onto the bowling direction. Shuffling at the mark
    fails the projection test; walking back projects negative and shrinks
    the window until it's rejected."""
    bcfg = cfg.bowling
    if dir3 is None:
        return None
    loco = fs.motion_energy > bcfg.runup_loco_energy
    gap_frames = int(bcfg.runup_gap_tolerance_s * fs.fps)
    onset_i = anchor_i
    gap = 0
    for i in range(anchor_i, -1, -1):
        if loco[i]:
            onset_i = i
            gap = 0
        else:
            gap += 1
            if gap > gap_frames:
                break
    # shrink from the far end until the net displacement points at the crease
    while onset_i < anchor_i:
        approach = float(np.dot(pos[anchor_i] - pos[onset_i], dir3))
        if approach >= bcfg.runup_min_approach:
            break
        onset_i += 1
    sustained = fs.t[anchor_i] - fs.t[onset_i]
    if sustained < bcfg.runup_sustain_s:
        return None
    if float(np.dot(pos[anchor_i] - pos[onset_i], dir3)) < bcfg.runup_min_approach:
        return None
    return float(fs.t[onset_i])


def _followthrough_resolution(fs: FeatureSeries, pos: np.ndarray,
                              dir3: Optional[np.ndarray], release_i: int,
                              arm: str, cfg: EngineConfig) -> (float, str):
    """Earliest time after release where ALL dissipation conditions hold for
    the whole stability window — arm velocity decayed, trunk back toward
    upright, and momentum resolved. Momentum resolves two ways per the spec:
    near-stationary, OR clearly broken out of the action (sustained movement
    *away* from the crease = walking back to the mark). Requiring the
    conditions simultaneously implements 'whichever resolves last governs
    the boundary'."""
    bcfg = cfg.bowling
    wrist = fs.wrist_speed_l if arm == "l" else fs.wrist_speed_r
    hold = max(1, int(bcfg.resolution_hold_s * fs.fps))
    max_i = min(len(fs.t) - 1,
                release_i + int(bcfg.resolution_max_after_release_s * fs.fps))
    stationary = fs.centroid_speed < bcfg.resolution_centroid_speed
    if dir3 is not None:
        win = max(3, int(fs.fps * 0.7) | 1)
        toward_v = np.gradient(pos @ dir3, fs.t)
        kernel = np.ones(win) / win
        toward_v = np.convolve(np.pad(toward_v, win // 2, mode="edge"),
                               kernel, mode="valid")[:len(fs.t)]
        walking_back = toward_v < -bcfg.breakout_reverse_speed
    else:
        walking_back = np.zeros(len(fs.t), dtype=bool)
    ok = ((wrist < bcfg.resolution_wrist_speed)
          & (fs.trunk_lean < bcfg.resolution_trunk_lean_rad)
          & (stationary | walking_back))
    for i in range(release_i + int(bcfg.min_followthrough_s * fs.fps),
                   max_i - hold):
        if np.all(ok[i:i + hold]):
            return float(fs.t[i]), "resolution"
    return float(fs.t[max_i]), "capped"


def detect_bowling_events(fs: FeatureSeries, cfg: EngineConfig) -> List[BowlingEvent]:
    events: List[BowlingEvent] = []
    pos = _approach_pos(fs, cfg)
    for cand in _release_candidates(fs, cfg):
        notes: List[str] = []
        release_i, release_t, arm = cand["i"], cand["t"], cand["arm"]
        bfc_t, ffc_t = _foot_contacts(fs, release_i, cfg)
        if bfc_t is None:
            notes.append("bfc_not_found")
        if ffc_t is None:
            notes.append("ffc_not_found")

        anchor_t = bfc_t if bfc_t is not None else release_t
        anchor_i = int(np.searchsorted(fs.t, anchor_t))
        dir3 = _bowling_direction(pos, fs, anchor_i)
        onset_t = _runup_onset(fs, pos, dir3, anchor_i, cfg)
        if onset_t is not None:
            raw_start, start_source = onset_t, "runup"
        else:
            # anchor fallback: a fixed lookback from BFC is explicitly NOT the
            # spec's definition of the start — flag it so confidence drops and
            # the clip can land in review rather than ship with a guessed start.
            raw_start = anchor_t - 2.0
            start_source = "anchor_fallback"
            notes.append("runup_onset_not_found")

        raw_end, end_source = _followthrough_resolution(fs, pos, dir3,
                                                        release_i, arm, cfg)
        if end_source == "capped":
            notes.append("followthrough_capped")

        rel_conf = min(1.0, cand["height"] / (cfg.bowling.release_min_wrist_speed * 2))
        events.append(BowlingEvent(
            release_t=release_t, release_conf=rel_conf, arm=arm,
            bfc_t=bfc_t, ffc_t=ffc_t,
            raw_start_t=max(raw_start, float(fs.t[0])),
            raw_end_t=raw_end, start_source=start_source,
            end_source=end_source, notes=notes))
    _separate_neighbors(events, fs)
    return events


def _separate_neighbors(events: List[BowlingEvent], fs: FeatureSeries) -> None:
    """Delivery N's window must never contain delivery N+1's action, however
    close together they are. When raw windows collide, cut at the motion
    minimum between the two releases (the walk-back lull) and flag both."""
    events.sort(key=lambda e: e.release_t)
    for a, b in zip(events, events[1:]):
        if a.raw_end_t <= b.raw_start_t:
            continue
        i0 = int(np.searchsorted(fs.t, a.release_t + 0.3))
        i1 = int(np.searchsorted(fs.t, b.release_t - 0.3))
        if i1 <= i0:
            cut = (a.release_t + b.release_t) / 2.0
        else:
            cut = float(fs.t[i0 + int(np.argmin(fs.motion_energy[i0:i1]))])
        if a.raw_end_t > cut:
            a.raw_end_t = cut
            a.notes.append("end_clamped_to_next_delivery")
        if b.raw_start_t < cut:
            b.raw_start_t = cut
            b.notes.append("start_clamped_to_prev_delivery")
