"""Central configuration for the auto-clipping engine.

============================================================================
!! EVERY NUMERIC THRESHOLD IN THIS FILE IS AN UNVALIDATED STARTING POINT !!

None of these values have been calibrated against real Amp footage. They are
reasoned guesses chosen to be safe-side (over-inclusive) per the boundary
spec, and they MUST be tuned against labeled ground truth before the
confidence gates are trusted in production. Do not scatter magic numbers
elsewhere in the codebase — every invented constant lives here.

Expected most-sensitive knobs, tune these first (in order):
  1. RUNUP_ONSET_SPEED / RUNUP_SUSTAIN_S      — bowling clip start
  2. RESOLUTION_* stillness thresholds         — both clip ends
  3. CONFIDENCE_AUTO / CONFIDENCE_REVIEW       — what ships vs. what queues
  4. MIN_EVENT_GAP_S                           — delivery N vs N+1 separation
  5. STANCE_STILL_STD / STANCE_MIN_S           — batting stance-set false fires
============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PoseConfig:
    """Pose extraction + subject tracking."""
    model_weights: str = "yolo11n-pose.pt"
    device: str = "cpu"              # "mps" on Apple Silicon, "cuda" on GPU box
    imgsz: int = 640                 # inference long side, px
    min_kpt_conf: float = 0.30       # UNTUNED: keypoint confidence floor
    min_box_conf: float = 0.35       # UNTUNED: person detection floor
    max_process_fps: float = 60.0    # never process more than this many fps
    detect_fps: float = 20.0         # UNTUNED: coarse detection pass rate —
                                     # wrist-speed peaks survive 20fps; precise
                                     # anchor timing comes from the refine pass
    batch_size: int = 8              # frames per YOLO inference batch
    refine: bool = True              # fine pass at native fps around each
                                     # detected delivery's key anchor
    refine_pre_s: float = 1.5        # UNTUNED: fine window before key anchor
    refine_post_s: float = 1.0       # UNTUNED: fine window after key anchor
    track_max_gap_s: float = 0.7     # UNTUNED: track survives dropouts up to this
    track_match_dist: float = 0.12   # UNTUNED: centroid match radius, frac of frame diag
    stitch_gap_s: float = 1.2        # UNTUNED: re-join a player's track across
                                     # occlusions/exits up to this long
    stitch_dist: float = 0.20        # UNTUNED: max centroid jump across a
                                     # stitched gap, frac of frame diag


@dataclass
class BowlingConfig:
    """Boundary spec: run-up initiation -> follow-through resolution."""
    # -- run-up onset (START) --
    # The run-up is detected as sustained LOCOMOTION (stride cadence shows up
    # as whole-body motion energy) whose NET DISPLACEMENT projects onto the
    # bowling direction. The bowling direction is estimated in image space
    # from the delivery stride itself (x, y, and torso-scale change), which
    # makes the detector camera-angle-agnostic: side-on run-ups show as x
    # translation, front-on run-ups as scale growth.
    runup_loco_energy: float = 0.35   # UNTUNED: mean keypoint speed
                                      # (torso-lengths/s) counting as locomotion
    runup_min_approach: float = 0.15  # UNTUNED: net displacement toward the
                                      # crease over the run-up, torso-equivalent
                                      # units — rejects pacing/walking-back
    approach_scale_gain: float = 3.0  # UNTUNED: weight of relative torso-scale
                                      # growth vs translation in the projection
    runup_sustain_s: float = 0.60     # UNTUNED: locomotion must hold this long
                                      # (no single-frame noise fires)
    runup_gap_tolerance_s: float = 0.50  # UNTUNED: allowed dropout inside run-up
                                         # (must bridge the gather/bound)
    # -- anchor events --
    release_min_wrist_speed: float = 0.9   # UNTUNED: peak wrist speed floor,
                                           # torso-lengths per second
    release_arm_raised_margin: float = 0.0 # wrist must be above shoulder at peak
    foot_contact_window_s: float = 0.8     # UNTUNED: BFC/FFC searched this far
                                           # before release
    # -- follow-through resolution (END) --
    resolution_wrist_speed: float = 0.25   # UNTUNED: wrist angular speed "near zero"
    resolution_centroid_speed: float = 0.04  # UNTUNED: forward momentum decayed
    resolution_trunk_lean_rad: float = 0.35  # UNTUNED: trunk back toward upright
    breakout_reverse_speed: float = 0.06 # UNTUNED: sustained approach-projection
                                         # velocity AWAY from the crease that
                                         # counts as "broken out of the action"
                                         # (walking back to the mark) — an
                                         # alternative end condition to stillness
    resolution_hold_s: float = 0.50        # UNTUNED: stability window all
                                           # conditions must hold
    min_followthrough_s: float = 1.0       # UNTUNED: never end sooner than this
                                           # after release — the trail-leg
                                           # swing-through and 2-4 decelerating
                                           # strides take at least this long
    resolution_max_after_release_s: float = 6.0  # UNTUNED: hard cap on search
    # -- padding (bias outward; under-inclusion corrupts KPI scores) --
    pre_roll_ms: int = 750           # UNTUNED: before detected run-up onset
    post_roll_ms: int = 1000         # UNTUNED: after follow-through resolution
    # -- plausibility (a "delivery" outside this range is a detection failure) --
    min_clip_s: float = 3.0          # UNTUNED: 3-step spinner floor
    max_clip_s: float = 30.0         # UNTUNED: 20m+ run-up fast bowler ceiling


@dataclass
class BattingConfig:
    """Boundary spec: stance set -> follow-through resolution."""
    # -- stance set (START) --
    stance_still_std: float = 0.010  # UNTUNED: ankle keypoint positional std
                                     # (frac of frame diag) counted as "planted"
    stance_min_s: float = 0.40       # UNTUNED: stillness must hold this long
    stance_max_lookback_s: float = 4.0  # UNTUNED: how far before swing we search
    stance_max_include_s: float = 2.0   # UNTUNED: cap on stance footage kept
                                        # (long stands between balls are idle)
    trigger_to_impact_max_s: float = 3.5  # UNTUNED: stance must lead to delivery
                                          # within this bound to be valid
    # -- impact anchor --
    impact_min_wrist_speed: float = 0.8  # UNTUNED: torso-lengths/s swing peak floor
    # -- follow-through resolution (END) --
    resolution_wrist_speed: float = 0.20   # UNTUNED
    resolution_trunk_rot: float = 1.5      # UNTUNED: trunk rotation velocity, rad/s
    resolution_hold_s: float = 0.40        # UNTUNED
    min_post_impact_s: float = 0.8   # UNTUNED: never end sooner than this after
                                     # impact — protects short defensive shots
    resolution_max_after_impact_s: float = 4.0  # UNTUNED
    # -- padding --
    pre_roll_ms: int = 1000          # UNTUNED: stance frames are directly scored
                                     # (foot alignment, head position) — generous
    post_roll_ms: int = 750          # UNTUNED
    # -- plausibility --
    min_clip_s: float = 2.0          # UNTUNED
    max_clip_s: float = 15.0         # UNTUNED
    emit_leaves: bool = True         # leaves emitted, tagged shot_type="leave"


@dataclass
class AudioConfig:
    """Audio transient corroboration. Never a sole trigger."""
    sample_rate: int = 22050
    band_hz: Tuple[int, int] = (2000, 8000)  # bat crack / ball-on-pitch band
    onset_mad_k: float = 6.0         # UNTUNED: peak threshold = median + k*MAD
    align_window_ms: int = 150       # UNTUNED: transient within this of a video
                                     # impact estimate corroborates + snaps it
    corroboration_boost: float = 0.15   # UNTUNED: confidence delta when aligned
    missing_audio_penalty: float = 0.05 # UNTUNED: clean audio but no transient
                                        # on a claimed hit


@dataclass
class SegmenterConfig:
    """Session-level segmentation + confidence gating."""
    min_event_gap_s: float = 4.0     # UNTUNED: release/impact peaks closer than
                                     # this are one delivery, not two
    min_track_s: float = 1.0         # UNTUNED: tracks shorter than this carry
                                     # no delivery evidence and are skipped
    role_gate: float = 0.30          # UNTUNED: minimum role score for a track
                                     # to be fed to that role's detector
    scene_cut_mad_k: float = 8.0     # UNTUNED: frame-diff spike threshold
                                     # (median + k*MAD) that counts as a hard
                                     # camera cut; clips never span a cut
    fragment_discard_frac: float = 0.5  # UNTUNED: after cut-clamping, a clip
                                        # shorter than this fraction of its
                                        # role's min plausible duration holds
                                        # nothing scoreable and is discarded
    motion_gate_percentile: float = 60.0  # UNTUNED: frame-diff energy percentile
                                          # below which frames are skipped as idle
    motion_gate_abs_floor: float = 1.5    # UNTUNED: absolute frame-diff energy a
                                          # window must also exceed — on an
                                          # idle-heavy hour the percentile alone
                                          # would still mark 40% "active"
    motion_gate_pad_s: float = 3.0   # UNTUNED: candidate windows padded this much
                                     # before pose runs (never starve the model)
    # confidence gates (drive auto-ship vs review vs discard)
    confidence_auto: float = 0.75    # UNTUNED: >= ships without review
    confidence_review: float = 0.40  # UNTUNED: [review, auto) flags manual review;
                                     # below discards (session flagged if many)
    low_fps_threshold: float = 50.0  # UNTUNED: below this, anchor confidence is
                                     # penalized (release/impact are sub-100ms)
    low_fps_penalty: float = 0.10    # UNTUNED
    max_discard_fraction: float = 0.5  # UNTUNED: above this the whole session is
                                       # flagged rather than silently thinned


@dataclass
class ClipConfig:
    """Output clip encoding. Frame-extraction contract: same container/codec
    the pipeline already accepts (mp4/h264), so downstream is unchanged."""
    reencode: bool = True            # frame-accurate cuts (stream-copy snaps to
                                     # keyframes and can eat up to a GOP ~2s)
    crf: int = 18                    # visually lossless-ish; KPI model reads frames
    preset: str = "veryfast"
    keep_audio: bool = True


@dataclass
class EvalConfig:
    """Tolerances that define 'correct' against ground truth."""
    anchor_tolerance_ms: int = 66    # UNTUNED: BFC/FFC/release/impact — tight.
                                     # ±2 frames at 30fps; halves at 60fps.
    boundary_tolerance_ms: int = 250 # UNTUNED: outer clip boundaries — loose;
                                     # padding makes symmetric error uninteresting.
                                     # The gating metric is phase truncation, not
                                     # boundary offset.
    iou_match_threshold: float = 0.5 # segment counts as detected at this IoU


@dataclass
class EngineConfig:
    pose: PoseConfig = field(default_factory=PoseConfig)
    bowling: BowlingConfig = field(default_factory=BowlingConfig)
    batting: BattingConfig = field(default_factory=BattingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


DEFAULT = EngineConfig()
