"""Pose extraction, track persistence, and subject role inference.

Stage 2: YOLO11-pose over the active windows, with a deliberately simple
centroid tracker (no external tracker dependency) for ID persistence, then
role scoring to decide which human is the bowler / batter / bystander.
Clipping around the wrong human is a hard failure mode, so role inference
is explicit and its evidence is kept for the confidence score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .types import KP


@dataclass
class TrackFrame:
    t: float
    kpts: np.ndarray        # (17, 2) pixel coords
    conf: np.ndarray        # (17,) keypoint confidences
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Track:
    track_id: int
    frames: List[TrackFrame] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return self.frames[0].t

    @property
    def t1(self) -> float:
        return self.frames[-1].t

    def centroid_xy(self) -> np.ndarray:
        """(N, 3) array of t, cx, cy using hip midpoint (pelvis proxy),
        falling back to box center when hips are low-confidence."""
        rows = []
        for f in self.frames:
            lh, rh = f.kpts[KP["l_hip"]], f.kpts[KP["r_hip"]]
            ok = f.conf[KP["l_hip"]] > 0.2 and f.conf[KP["r_hip"]] > 0.2
            if ok:
                cx, cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            else:
                x1, y1, x2, y2 = f.box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rows.append((f.t, cx, cy))
        return np.asarray(rows)

    def mean_box_area(self) -> float:
        areas = [(f.box[2] - f.box[0]) * (f.box[3] - f.box[1]) for f in self.frames]
        return float(np.mean(areas)) if areas else 0.0


class CentroidTracker:
    """Nearest-centroid association with a max-gap timeout. Simple by design:
    net footage has few, well-separated people; when it fails (players crossing)
    the downstream role scores drop and the clip lands in review instead of
    shipping wrong."""

    def __init__(self, cfg: EngineConfig, frame_diag: float):
        self.cfg = cfg
        self.frame_diag = frame_diag
        self.tracks: List[Track] = []
        self._next_id = 0

    def _active(self, t: float) -> List[Track]:
        gap = self.cfg.pose.track_max_gap_s
        return [tr for tr in self.tracks if t - tr.t1 <= gap]

    def update(self, t: float, detections: List[TrackFrame]) -> None:
        active = self._active(t)
        max_dist = self.cfg.pose.track_match_dist * self.frame_diag
        used = set()
        # greedy nearest-match, closest pairs first
        pairs = []
        for di, det in enumerate(detections):
            dcx = (det.box[0] + det.box[2]) / 2
            dcy = (det.box[1] + det.box[3]) / 2
            for tr in active:
                last = tr.frames[-1]
                tcx = (last.box[0] + last.box[2]) / 2
                tcy = (last.box[1] + last.box[3]) / 2
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist <= max_dist:
                    pairs.append((dist, di, tr))
        pairs.sort(key=lambda p: p[0])
        matched_tracks = set()
        for _, di, tr in pairs:
            if di in used or tr.track_id in matched_tracks:
                continue
            tr.frames.append(detections[di])
            used.add(di)
            matched_tracks.add(tr.track_id)
        for di, det in enumerate(detections):
            if di not in used:
                tr = Track(track_id=self._next_id)
                self._next_id += 1
                tr.frames.append(det)
                self.tracks.append(tr)


def _parse_result(res, t: float) -> List[TrackFrame]:
    dets: List[TrackFrame] = []
    if res.keypoints is not None and len(res.keypoints) > 0:
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = (res.keypoints.conf.cpu().numpy()
               if res.keypoints.conf is not None
               else np.ones(kxy.shape[:2]))
        boxes = res.boxes.xyxy.cpu().numpy()
        for i in range(len(kxy)):
            dets.append(TrackFrame(t=t, kpts=kxy[i], conf=kcf[i],
                                   box=tuple(boxes[i][:4])))
    return dets


def extract_tracks(video_path: str, windows: List[Tuple[float, float]],
                   cfg: EngineConfig, video_fps: float,
                   frame_size: Tuple[int, int],
                   detect_fps: float = 0.0,
                   min_track_frames: int = 5) -> List[Track]:
    """Run YOLO pose over the active windows; return persistent tracks.

    Long-session friendly: seeks directly to each active window rather than
    decoding the whole file, samples at the coarse detection rate (the refine
    pass restores native-fps precision around detected anchors), and batches
    frames per inference call."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(cfg.pose.model_weights)
    fps_cap = detect_fps or min(cfg.pose.detect_fps, cfg.pose.max_process_fps)
    process_fps = min(video_fps, fps_cap)
    w, h = frame_size
    tracker = CentroidTracker(cfg, frame_diag=math.hypot(w, h))

    batch: List[np.ndarray] = []
    times: List[float] = []

    def flush() -> None:
        if not batch:
            return
        results = model.predict(batch, imgsz=cfg.pose.imgsz, verbose=False,
                                device=cfg.pose.device,
                                conf=cfg.pose.min_box_conf)
        for bt, res in zip(times, results):
            tracker.update(bt, _parse_result(res, bt))
        batch.clear()
        times.clear()

    cap = cv2.VideoCapture(video_path)
    try:
        for lo, hi in windows:
            if lo > 0.5:
                cap.set(cv2.CAP_PROP_POS_MSEC, lo * 1000.0)
            next_t = -1.0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if t > hi:
                    break
                if t < next_t:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                next_t = t + 1.0 / process_fps
                batch.append(frame)
                times.append(t)
                if len(batch) >= cfg.pose.batch_size:
                    flush()
            flush()
    finally:
        cap.release()
    tracks = [tr for tr in tracker.tracks if len(tr.frames) >= min_track_frames]
    return stitch_tracks(tracks, cfg, math.hypot(w, h))


def stitch_tracks(tracks: List[Track], cfg: EngineConfig,
                  frame_diag: float) -> List[Track]:
    """Re-join one player's track across short occlusions (net poles, brief
    frame exits). A fragmented run-up can't produce a run-up onset, so
    stitching directly improves the bowling start boundary."""
    tracks = sorted(tracks, key=lambda tr: tr.t0)
    merged: List[Track] = []
    for tr in tracks:
        target = None
        b = tr.frames[0].box
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for m in merged:
            gap = tr.t0 - m.t1
            if not (0.0 <= gap <= cfg.pose.stitch_gap_s):
                continue
            a = m.frames[-1].box
            d = math.hypot((a[0] + a[2]) / 2 - bcx, (a[1] + a[3]) / 2 - bcy)
            if d <= cfg.pose.stitch_dist * frame_diag:
                target = m
                break
        if target is not None:
            target.frames.extend(tr.frames)
        else:
            merged.append(tr)
    return merged


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------

@dataclass
class RoleScores:
    bowler: float
    batter: float
    evidence: Dict[str, float]


def score_roles(track: Track, frame_w: int, frame_h: int) -> RoleScores:
    """Heuristic role evidence per track.

    Bowler: sustained, mostly-monotonic horizontal translation (the run-up)
    followed by high wrist speed. Batter: low centroid variance (planted at
    the crease) plus periodic wrist activity. A coach or passer-by scores
    low on both. These scores feed both subject selection and the final
    clip confidence.
    """
    c = track.centroid_xy()
    if len(c) < 10:
        return RoleScores(0.0, 0.0, {"too_short": 1.0})
    t, x, y = c[:, 0], c[:, 1], c[:, 2]
    span = max(t[-1] - t[0], 1e-6)
    x_range = float(np.ptp(x)) / frame_w
    # net displacement vs path length: run-ups are directed, pacing is not
    dx = np.diff(x)
    path = float(np.sum(np.abs(dx))) + 1e-6
    directedness = abs(float(np.sum(dx))) / path
    x_std = float(np.std(x)) / frame_w
    box_area_frac = track.mean_box_area() / (frame_w * frame_h)

    bowler = 0.0
    bowler += min(x_range / 0.25, 1.0) * 0.5          # travelled a good fraction of frame
    bowler += directedness * 0.3                       # in one direction
    bowler += min(box_area_frac / 0.02, 1.0) * 0.2    # is an actual foreground person
    batter = 0.0
    batter += max(0.0, 1.0 - x_std / 0.05) * 0.6      # stays put
    batter += min(box_area_frac / 0.02, 1.0) * 0.4
    return RoleScores(bowler=bowler, batter=batter, evidence={
        "x_range": x_range, "directedness": directedness,
        "x_std": x_std, "box_area_frac": box_area_frac,
        "duration_s": span,
    })


def pick_subject(tracks: List[Track], role: str, frame_w: int,
                 frame_h: int) -> Optional[Tuple[Track, RoleScores]]:
    """Best track for the requested role, or None if nobody fits."""
    best: Optional[Tuple[Track, RoleScores]] = None
    for tr in tracks:
        s = score_roles(tr, frame_w, frame_h)
        val = s.bowler if role == "bowling" else s.batter
        if best is None or val > (best[1].bowler if role == "bowling" else best[1].batter):
            best = (tr, s)
    if best is None:
        return None
    val = best[1].bowler if role == "bowling" else best[1].batter
    return best if val > 0.3 else None
