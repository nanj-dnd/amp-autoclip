"""Batting boundary detection: stance set -> follow-through resolution.

Anchor-first: detect the swing/impact peak, snap it to an audio transient
when one aligns, walk backward to find the stable stance window, forward
through motion decay to find follow-through resolution. Handles the
shot-type variance the spec calls out: full swings, fast-decaying
defensive shots (min post-impact dwell), leaves (emitted, tagged), and
aborted deliveries (discarded).

Phase model:
idle -> stance set -> trigger -> backlift -> downswing -> impact ->
follow-through -> idle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from .audio import AudioTransient, nearest_transient
from .config import EngineConfig
from .features import FeatureSeries
from .types import Anchor


@dataclass
class BattingEvent:
    impact_t: Optional[float]       # None for a leave
    impact_conf: float
    impact_source: str              # "pose" | "pose+audio" | "audio-corroborated"
    stance_t: Optional[float]       # stance-set onset (start of stable window)
    trigger_t: Optional[float]      # end of stillness = trigger/backlift onset
    raw_start_t: float
    raw_end_t: float
    shot_type: str                  # "swing" | "leave"
    aborted: bool
    notes: List[str]

    def anchors(self) -> List[Anchor]:
        out: List[Anchor] = []
        if self.stance_t is not None:
            out.append(Anchor("stance_set", self.stance_t, 0.5))
        if self.trigger_t is not None:
            out.append(Anchor("trigger_onset", self.trigger_t, 0.4))
        if self.impact_t is not None:
            out.append(Anchor("impact", self.impact_t, self.impact_conf,
                              source=self.impact_source))
        return out


def _swing_peaks(fs: FeatureSeries, cfg: EngineConfig) -> List[dict]:
    speed = np.maximum(fs.wrist_speed_l, fs.wrist_speed_r)
    min_gap = max(1, int(cfg.segmenter.min_event_gap_s * fs.fps))
    peaks, props = sp_signal.find_peaks(
        speed, height=cfg.batting.impact_min_wrist_speed, distance=min_gap)
    return [{"i": int(p), "t": float(fs.t[p]), "height": float(h)}
            for p, h in zip(peaks, props["peak_heights"])]


def _stance_window(fs: FeatureSeries, before_i: int,
                   cfg: EngineConfig) -> Tuple[Optional[float], Optional[float]]:
    """Latest sustained ankle-stillness window ending before `before_i`.
    Returns (stance_start_t, stillness_end_t). Stillness end doubles as the
    trigger/backlift onset estimate."""
    bcfg = cfg.batting
    lookback = int(bcfg.stance_max_lookback_s * fs.fps)
    lo = max(0, before_i - lookback)
    still = fs.ankle_still_std[lo:before_i] < bcfg.stance_still_std
    if len(still) == 0:
        return None, None
    min_frames = max(1, int(bcfg.stance_min_s * fs.fps))
    # scan from the end: latest run of stillness long enough to be a set stance
    run_end = None
    run_len = 0
    for i in range(len(still) - 1, -1, -1):
        if still[i]:
            if run_end is None:
                run_end = i
            run_len += 1
        else:
            if run_end is not None and run_len >= min_frames:
                break
            run_end, run_len = None, 0
    if run_end is None or run_len < min_frames:
        return None, None
    start_i = lo + run_end - run_len + 1
    end_i = lo + run_end
    # cap how much stance footage is kept — a batter standing set for ten
    # seconds between balls is idle, not part of this delivery
    start_t = max(float(fs.t[start_i]),
                  float(fs.t[end_i]) - bcfg.stance_max_include_s)
    return start_t, float(fs.t[end_i])


def _followthrough_resolution(fs: FeatureSeries, impact_i: int,
                              cfg: EngineConfig) -> Tuple[float, str]:
    """Wrist velocity decayed AND trunk rotation back to baseline, sustained.
    The min_post_impact dwell keeps fast-decaying defensive shots from being
    truncated before the batter settles back to neutral."""
    bcfg = cfg.batting
    wrist = np.maximum(fs.wrist_speed_l, fs.wrist_speed_r)
    hold = max(1, int(bcfg.resolution_hold_s * fs.fps))
    earliest = impact_i + int(bcfg.min_post_impact_s * fs.fps)
    max_i = min(len(fs.t) - 1,
                impact_i + int(bcfg.resolution_max_after_impact_s * fs.fps))
    ok = ((wrist < bcfg.resolution_wrist_speed)
          & (fs.trunk_rot_vel < bcfg.resolution_trunk_rot))
    for i in range(earliest, max_i - hold):
        if np.all(ok[i:i + hold]):
            return float(fs.t[i]), "resolution"
    return float(fs.t[max_i]), "capped"


def _is_aborted(fs: FeatureSeries, stillness_end_t: float,
                cfg: EngineConfig) -> bool:
    """Batter pulled out: after the stance broke, sustained walking (centroid
    translation) instead of a swing."""
    i0 = int(np.searchsorted(fs.t, stillness_end_t))
    i1 = min(len(fs.t), i0 + int(cfg.batting.trigger_to_impact_max_s * fs.fps))
    if i1 - i0 < 3:
        return False
    walking = fs.centroid_speed[i0:i1] > 0.5   # torso-lengths/s, sustained walk
    return bool(np.mean(walking) > 0.6)


def detect_batting_events(fs: FeatureSeries, cfg: EngineConfig,
                          transients: List[AudioTransient]) -> List[BattingEvent]:
    bcfg = cfg.batting
    events: List[BattingEvent] = []
    used_transients: List[float] = []

    for cand in _swing_peaks(fs, cfg):
        notes: List[str] = []
        impact_i, impact_t = cand["i"], cand["t"]
        impact_conf = min(1.0, cand["height"] / (bcfg.impact_min_wrist_speed * 2))
        impact_source = "pose"

        tr = nearest_transient(transients, impact_t, cfg.audio.align_window_ms)
        if tr is not None:
            # audio pins sub-frame timing; corroboration raises confidence
            impact_t = tr.t
            impact_conf = min(1.0, impact_conf + cfg.audio.corroboration_boost)
            impact_source = "pose+audio"
            used_transients.append(tr.t)
        elif transients:
            # audio track is live but silent here — mild penalty only, the
            # swing may be an air-shot or the mic may have missed it
            impact_conf = max(0.0, impact_conf - cfg.audio.missing_audio_penalty)
            notes.append("no_audio_transient")

        stance_t, trigger_t = _stance_window(fs, impact_i, cfg)
        if stance_t is None:
            notes.append("stance_set_not_found")
            raw_start = impact_t - 2.5   # fallback lookback; review-bound
        else:
            if trigger_t is not None and impact_t - trigger_t > bcfg.trigger_to_impact_max_s:
                notes.append("stance_far_from_impact")
            raw_start = stance_t

        raw_end, end_source = _followthrough_resolution(fs, impact_i, cfg)
        if end_source == "capped":
            notes.append("followthrough_capped")

        events.append(BattingEvent(
            impact_t=impact_t, impact_conf=impact_conf,
            impact_source=impact_source, stance_t=stance_t,
            trigger_t=trigger_t, raw_start_t=max(raw_start, float(fs.t[0])),
            raw_end_t=raw_end, shot_type="swing", aborted=False, notes=notes))

    if bcfg.emit_leaves:
        events.extend(_detect_leaves(fs, cfg, transients, used_transients,
                                     [e.impact_t for e in events if e.impact_t]))

    events.sort(key=lambda e: e.raw_start_t)
    return events


def _detect_leaves(fs: FeatureSeries, cfg: EngineConfig,
                   transients: List[AudioTransient],
                   used: List[float], swing_times: List[float]) -> List[BattingEvent]:
    """Leaves: a delivery arrived (ball-on-pitch transient) while the batter
    was set, but no swing followed. Stance/trigger mechanics are still
    scoreable, so these are emitted tagged shot_type='leave'. v1 honesty:
    without ball tracking this only fires when audio catches the bounce, so
    leaves on silent/noisy footage will be missed — a known gap, see DESIGN.md."""
    bcfg = cfg.batting
    out: List[BattingEvent] = []
    for tr in transients:
        if any(abs(tr.t - u) < 1.0 for u in used):
            continue
        if any(abs(tr.t - s) < cfg.segmenter.min_event_gap_s for s in swing_times):
            continue
        i = int(np.searchsorted(fs.t, tr.t))
        if i <= 0 or i >= len(fs.t):
            continue
        # batter must be set (recent stillness) and NOT swinging
        stance_t, trigger_t = _stance_window(fs, min(i + int(fs.fps), len(fs.t) - 1), cfg)
        if stance_t is None:
            continue
        wrist = np.maximum(fs.wrist_speed_l, fs.wrist_speed_r)
        win = wrist[max(0, i - int(fs.fps * 0.3)):i + int(fs.fps * 0.5)]
        if len(win) and float(np.max(win)) > bcfg.impact_min_wrist_speed * 0.7:
            continue   # that's a swing the peak-picker missed, not a leave
        if trigger_t is not None and _is_aborted(fs, trigger_t, cfg):
            continue   # backed out — discard, don't emit
        raw_end = min(float(fs.t[-1]), tr.t + 1.5)
        out.append(BattingEvent(
            impact_t=None, impact_conf=0.0, impact_source="audio-corroborated",
            stance_t=stance_t, trigger_t=trigger_t,
            raw_start_t=stance_t, raw_end_t=raw_end,
            shot_type="leave", aborted=False,
            notes=["leave_from_audio_bounce"]))
    return out
