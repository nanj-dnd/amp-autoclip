"""Core data types shared across the engine."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# COCO-17 keypoint indices (YOLO pose output order)
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}


@dataclass
class VideoInfo:
    path: str
    fps: float
    duration_s: float
    width: int
    height: int
    n_frames: int
    has_audio: bool


@dataclass
class Anchor:
    """A near-instant event the KPI model scores off (BFC, FFC, release,
    impact, stance_set, trigger). Timing in seconds from video start."""
    name: str
    t: float
    confidence: float
    source: str = "pose"   # "pose" | "audio" | "pose+audio"


@dataclass
class Delivery:
    """One detected delivery instance — the engine's unit of output."""
    start_t: float                 # padded clip start (s)
    end_t: float                   # padded clip end (s)
    type: str                      # "bowling" | "batting"
    confidence: float
    anchors: List[Anchor] = field(default_factory=list)
    raw_start_t: float = 0.0       # boundary before padding
    raw_end_t: float = 0.0
    shot_type: Optional[str] = None   # batting: "swing" | "leave" | None
    pair_id: Optional[str] = None  # links bowling+batting clips of one delivery
    subject_track_id: Optional[int] = None
    status: str = "auto"           # "auto" | "review" | "discard"
    notes: List[str] = field(default_factory=list)

    def anchor(self, name: str) -> Optional[Anchor]:
        for a in self.anchors:
            if a.name == name:
                return a
        return None

    def to_manifest(self) -> Dict:
        d = {
            "start_timestamp": round(self.start_t, 3),
            "end_timestamp": round(self.end_t, 3),
            "type": self.type,
            "confidence_score": round(self.confidence, 3),
            "status": self.status,
            "raw_start_timestamp": round(self.raw_start_t, 3),
            "raw_end_timestamp": round(self.raw_end_t, 3),
            "anchors": [
                {"name": a.name, "t": round(a.t, 3),
                 "confidence": round(a.confidence, 3), "source": a.source}
                for a in self.anchors
            ],
        }
        if self.shot_type:
            d["shot_type"] = self.shot_type
        if self.pair_id:
            d["pair_id"] = self.pair_id
        if self.subject_track_id is not None:
            d["subject_track_id"] = self.subject_track_id
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class SessionResult:
    video: VideoInfo
    deliveries: List[Delivery]
    session_flags: List[str] = field(default_factory=list)

    def to_manifest(self) -> Dict:
        return {
            "video": asdict(self.video),
            "session_flags": self.session_flags,
            "deliveries": [d.to_manifest() for d in self.deliveries],
        }
