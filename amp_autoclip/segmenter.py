"""Session-level segmentation: tracks -> deliveries.

Pulls together subject selection, per-role event detection, outward
padding, plausibility gating, confidence scoring, and bowling/batting
pairing. This is the v1 (heuristic, spec-encoded) segmentation path; the
trained temporal model in amp_autoclip.model slots in as a replacement for
the event-detection step behind the same Delivery interface.
"""
from __future__ import annotations

from typing import List, Optional

from .audio import AudioTransient
from .config import EngineConfig
from .events_batting import BattingEvent, detect_batting_events
from .events_bowling import BowlingEvent, detect_bowling_events
from .features import FeatureSeries, compute_features
from .pose import RoleScores, Track, score_roles
from .types import Delivery, VideoInfo

# UNTUNED evidence weights for the confidence score (see config.py header)
_NOTE_PENALTIES = {
    "runup_onset_not_found": 0.15,
    "followthrough_capped": 0.10,
    "bfc_not_found": 0.05,
    "ffc_not_found": 0.05,
    "stance_set_not_found": 0.20,
    "stance_far_from_impact": 0.10,
    "no_audio_transient": 0.0,     # already applied in the detector
    "leave_from_audio_bounce": 0.0,
    "end_clamped_to_next_delivery": 0.05,
    "start_clamped_to_prev_delivery": 0.05,
    "duplicate_track_detection": 0.0,
    "scene_cut_clamped": 0.10,
}
_ROLE_BONUS_WEIGHT = 0.20
_LEAVE_BASE_CONF = 0.45


def _confidence(base: float, role_val: float, notes: List[str],
                video_fps: float, cfg: EngineConfig) -> float:
    conf = base
    conf += max(0.0, role_val - 0.3) * _ROLE_BONUS_WEIGHT
    for n in notes:
        conf -= _NOTE_PENALTIES.get(n, 0.0)
    if video_fps < cfg.segmenter.low_fps_threshold:
        conf -= cfg.segmenter.low_fps_penalty
    return max(0.0, min(1.0, conf))


def _status(conf: float, cfg: EngineConfig) -> str:
    if conf >= cfg.segmenter.confidence_auto:
        return "auto"
    if conf >= cfg.segmenter.confidence_review:
        return "review"
    return "discard"


def _pad_and_clamp(raw_start: float, raw_end: float, pre_ms: int, post_ms: int,
                   duration: float):
    return (max(0.0, raw_start - pre_ms / 1000.0),
            min(duration, raw_end + post_ms / 1000.0))


def _bowling_delivery(ev: BowlingEvent, role: RoleScores, track_id: int,
                      video: VideoInfo, cfg: EngineConfig) -> Delivery:
    start, end = _pad_and_clamp(ev.raw_start_t, ev.raw_end_t,
                                cfg.bowling.pre_roll_ms, cfg.bowling.post_roll_ms,
                                video.duration_s)
    notes = list(ev.notes)
    conf = _confidence(ev.release_conf, role.bowler, notes, video.fps, cfg)
    d = Delivery(start_t=start, end_t=end, type="bowling", confidence=conf,
                 anchors=ev.anchors(), raw_start_t=ev.raw_start_t,
                 raw_end_t=ev.raw_end_t, subject_track_id=track_id, notes=notes)
    dur = end - start
    if not (cfg.bowling.min_clip_s <= dur <= cfg.bowling.max_clip_s):
        d.notes.append("implausible_duration_%.1fs" % dur)
        d.status = "discard"
        d.confidence = min(d.confidence, cfg.segmenter.confidence_review - 0.01)
    else:
        d.status = _status(conf, cfg)
    return d


def _batting_delivery(ev: BattingEvent, role: RoleScores, track_id: int,
                      video: VideoInfo, cfg: EngineConfig) -> Delivery:
    start, end = _pad_and_clamp(ev.raw_start_t, ev.raw_end_t,
                                cfg.batting.pre_roll_ms, cfg.batting.post_roll_ms,
                                video.duration_s)
    notes = list(ev.notes)
    base = ev.impact_conf if ev.shot_type == "swing" else _LEAVE_BASE_CONF
    conf = _confidence(base, role.batter, notes, video.fps, cfg)
    d = Delivery(start_t=start, end_t=end, type="batting", confidence=conf,
                 anchors=ev.anchors(), raw_start_t=ev.raw_start_t,
                 raw_end_t=ev.raw_end_t, shot_type=ev.shot_type,
                 subject_track_id=track_id, notes=notes)
    dur = end - start
    if not (cfg.batting.min_clip_s <= dur <= cfg.batting.max_clip_s):
        d.notes.append("implausible_duration_%.1fs" % dur)
        d.status = "discard"
        d.confidence = min(d.confidence, cfg.segmenter.confidence_review - 0.01)
    else:
        d.status = _status(conf, cfg)
    return d


def _clamp_padding_to_neighbors(deliveries: List[Delivery]) -> None:
    """Outward padding is biased generous, but a padded window must never
    reach a neighboring same-type delivery's key anchor — a clip containing
    two releases would mis-anchor downstream scoring."""
    for a, b in zip(deliveries, deliveries[1:]):
        if a.type != b.type:
            continue
        key = "ball_release" if a.type == "bowling" else "impact"
        b_anchor = b.anchor(key)
        if b_anchor is not None:                # neighbor's anchor is a hard bound
            a.end_t = min(a.end_t, b_anchor.t - 0.3)
        a_anchor = a.anchor(key)
        if a_anchor is not None:
            b.start_t = max(b.start_t, a_anchor.t + 0.3)


def _resolve_role_conflicts(deliveries: List[Delivery]) -> None:
    """The same person detected as both bowler and batter for the same
    action is one event seen through two detectors, not two players — a
    batter's high-wrist pull can read as a 'release'. Same action = the key
    anchors (release/impact are both wrist-speed peaks) land at nearly the
    same instant. Keep the reading whose evidence resolved better."""
    for a in deliveries:
        for b in deliveries:
            if (a is b or a.type != "bowling" or b.type != "batting"
                    or a.status == "discard" or b.status == "discard"):
                continue
            if a.subject_track_id is None or a.subject_track_id != b.subject_track_id:
                continue
            rel, imp = a.anchor("ball_release"), b.anchor("impact")
            if rel is None or imp is None or abs(rel.t - imp.t) > 1.0:
                continue
            loser = a if a.confidence < b.confidence else b
            loser.status = "discard"
            loser.notes.append("role_conflict_same_subject")


def _pair_overlaps(deliveries: List[Delivery]) -> None:
    """One net-session video can contain the bowler AND the batter of the
    same delivery — two different people. Emit both clips (role-specific KPI
    pipelines consume one subject each) linked by a shared pair_id."""
    bowling = [d for d in deliveries if d.type == "bowling" and d.status != "discard"]
    batting = [d for d in deliveries if d.type == "batting" and d.status != "discard"]
    n = 0
    for bo in bowling:
        for ba in batting:
            if (bo.subject_track_id is not None
                    and bo.subject_track_id == ba.subject_track_id):
                continue
            overlap = min(bo.end_t, ba.end_t) - max(bo.start_t, ba.start_t)
            if overlap > 0.5 and ba.pair_id is None and bo.pair_id is None:
                n += 1
                pid = "pair_%02d" % n
                bo.pair_id = pid
                ba.pair_id = pid
                break


def segment_session(video: VideoInfo, tracks: List[Track],
                    transients: List[AudioTransient], cfg: EngineConfig,
                    mode: str = "auto",
                    scene_cuts: Optional[List[float]] = None) -> "SessionOutput":
    """mode: 'bowling' | 'batting' | 'both' | 'auto'.

    Detection runs over EVERY viable track, not a single chosen subject:
    tracks fragment whenever a player leaves frame between deliveries (and on
    every cut in edited footage), so one delivery may live in each fragment.
    Duplicate detections of the same action across tracks are merged
    afterward; role scores gate which detector sees which track and feed the
    confidence."""
    deliveries: List[Delivery] = []
    flags: List[str] = []
    grid_fps = min(video.fps, cfg.pose.max_process_fps)

    want_bowling = mode in ("bowling", "both", "auto")
    want_batting = mode in ("batting", "both", "auto")

    for track in tracks:
        if track.t1 - track.t0 < cfg.segmenter.min_track_s:
            continue
        role = score_roles(track, video.width, video.height)
        if max(role.bowler, role.batter) < cfg.segmenter.role_gate:
            continue
        fs = compute_features(track, grid_fps)
        if fs is None:
            continue
        if want_bowling and role.bowler >= cfg.segmenter.role_gate:
            for ev in detect_bowling_events(fs, cfg):
                deliveries.append(_bowling_delivery(ev, role, track.track_id,
                                                    video, cfg))
        if want_batting and role.batter >= cfg.segmenter.role_gate:
            for ev in detect_batting_events(fs, cfg, transients):
                deliveries.append(_batting_delivery(ev, role, track.track_id,
                                                    video, cfg))

    deliveries.sort(key=lambda d: d.start_t)
    _dedupe_same_action(deliveries, cfg)
    if scene_cuts:
        flags.append("edited_footage_suspected_%d_cuts" % len(scene_cuts))
        _clamp_to_scene_cuts(deliveries, scene_cuts)
    _recheck_plausibility(deliveries, cfg)
    if mode == "auto":
        _suppress_weak_role(deliveries)
    _clamp_padding_to_neighbors([d for d in deliveries if d.status != "discard"])
    _resolve_role_conflicts(deliveries)
    _refresh_status(deliveries, cfg)
    _pair_overlaps(deliveries)

    kept = [d for d in deliveries if d.status != "discard"]
    if deliveries and len(kept) / len(deliveries) < (1 - cfg.segmenter.max_discard_fraction):
        flags.append("high_discard_rate_review_session")
    if not deliveries:
        flags.append("no_deliveries_detected")
        if mode == "bowling":
            flags.append("no_bowler_subject_found")
        if mode == "batting":
            flags.append("no_batter_subject_found")
    return SessionOutput(deliveries=deliveries, flags=flags)


def _key_anchor_t(d: Delivery) -> Optional[float]:
    a = d.anchor("ball_release" if d.type == "bowling" else "impact")
    return a.t if a else None


def _dedupe_same_action(deliveries: List[Delivery], cfg: EngineConfig) -> None:
    """Two tracks can both see the same person perform the same delivery
    (fragmented or crossing tracks). Same type + key anchors within the
    min event gap = one action; keep the stronger reading."""
    for a in deliveries:
        for b in deliveries:
            if (a is b or a.type != b.type
                    or a.status == "discard" or b.status == "discard"):
                continue
            ta, tb = _key_anchor_t(a), _key_anchor_t(b)
            if ta is not None and tb is not None:
                same = abs(ta - tb) < cfg.segmenter.min_event_gap_s
            else:   # leaves: fall back to window overlap
                overlap = min(a.end_t, b.end_t) - max(a.start_t, b.start_t)
                same = overlap > 0.5 * min(a.end_t - a.start_t,
                                           b.end_t - b.start_t)
            if not same:
                continue
            loser = a if a.confidence < b.confidence else b
            loser.status = "discard"
            loser.notes.append("duplicate_track_detection")


def _recheck_plausibility(deliveries: List[Delivery], cfg: EngineConfig) -> None:
    """Clamping (scene cuts, neighbors) can shrink a window after the
    construction-time plausibility check. A clip far below its role's minimum
    duration holds no scoreable phases — discard; moderately short survives,
    but never auto-ships."""
    for d in deliveries:
        if d.status == "discard":
            continue
        min_s = cfg.bowling.min_clip_s if d.type == "bowling" else cfg.batting.min_clip_s
        dur = d.end_t - d.start_t
        if dur < min_s * cfg.segmenter.fragment_discard_frac:
            d.status = "discard"
            d.notes.append("fragment_too_short_%.1fs" % dur)
        elif dur < min_s:
            d.notes.append("short_fragment")
            if d.status == "auto":
                d.status = "review"


def _refresh_status(deliveries: List[Delivery], cfg: EngineConfig) -> None:
    """Penalties applied after construction (clamps, audio, fps) lower the
    confidence but never re-derived the status — a 0.34 clip could survive as
    'review'. Re-derive, demoting only (deliberate demotions stay demoted)."""
    order = {"discard": 0, "review": 1, "auto": 2}
    for d in deliveries:
        if d.status == "discard":
            continue
        s = _status(d.confidence, cfg)
        if order[s] < order[d.status]:
            d.status = s
            if s == "discard":
                d.notes.append("confidence_below_review_after_penalties")


def _is_complete(d: Delivery) -> bool:
    """Structurally complete = carries the anchors that define its role, not
    just a wrist-speed peak."""
    if d.type == "bowling":
        return (d.anchor("back_foot_contact") is not None
                or d.anchor("front_foot_contact") is not None)
    return d.anchor("stance_set") is not None


def _suppress_weak_role(deliveries: List[Delivery]) -> None:
    """Auto mode only: when one role shows repeated structurally-complete
    detections and the other role never once produces its defining anchor,
    the weak role is misfiring on the strong role's motion (a bowler's arm
    swing reads as a 'swing', a batter's pull as a 'release'). Suppress it
    for the session."""
    complete = {"bowling": 0, "batting": 0}
    total = {"bowling": 0, "batting": 0}
    for d in deliveries:
        if d.status == "discard":
            continue
        total[d.type] += 1
        complete[d.type] += int(_is_complete(d))
    for weak, strong in (("batting", "bowling"), ("bowling", "batting")):
        if complete[strong] >= 2 and total[weak] > 0 and complete[weak] == 0:
            for d in deliveries:
                if d.type == weak and d.status != "discard":
                    d.status = "discard"
                    d.notes.append("role_suppressed_no_defining_anchor")


def _clamp_to_scene_cuts(deliveries: List[Delivery],
                         cuts: List[float]) -> None:
    """A clip must never span a hard camera cut — frames past the cut belong
    to a different shot. Clamp both raw and padded boundaries to the cut
    segment containing the key anchor."""
    for d in deliveries:
        t = _key_anchor_t(d)
        if t is None:
            t = (d.raw_start_t + d.raw_end_t) / 2
        prev_cut = max([c for c in cuts if c <= t], default=None)
        next_cut = min([c for c in cuts if c > t], default=None)
        clamped = False
        if prev_cut is not None and d.start_t < prev_cut:
            d.start_t = prev_cut
            d.raw_start_t = max(d.raw_start_t, prev_cut)
            clamped = True
        if next_cut is not None and d.end_t > next_cut:
            d.end_t = next_cut
            d.raw_end_t = min(d.raw_end_t, next_cut)
            clamped = True
        if clamped:
            d.notes.append("scene_cut_clamped")
            d.confidence = max(0.0, d.confidence
                               - _NOTE_PENALTIES["scene_cut_clamped"])
            if d.status == "auto":   # edited footage never auto-ships
                d.status = "review"


class SessionOutput:
    def __init__(self, deliveries: List[Delivery], flags: List[str]):
        self.deliveries = deliveries
        self.flags = flags
