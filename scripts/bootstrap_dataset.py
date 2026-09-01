"""Pseudo-label bootstrap: engine detections -> training dataset, no humans.

Runs the heuristic engine over a folder of videos in ONE role, keeps only
detections whose evidence is structurally complete (key anchor + a foot
contact for bowling; stance + impact for batting) above a confidence floor,
and writes them in the exact dataset layout model/train.py consumes:

    dataset/features/<video>__t<track>.npz
    dataset/labels/<video>__t<track>.jsonl

Each (video, track) pair becomes one training sequence. Pseudo-labels are
weaker than human labels — the model trained on them learns to imitate the
heuristic where the heuristic is confident, which is exactly what the
ensemble needs (agreement = confidence signal), and gets replaced by
review-corrected labels as those accumulate.

    python -m scripts.bootstrap_dataset --videos test_videos --role bowling \
        --out dataset --device mps
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from amp_autoclip import video_io
from amp_autoclip.audio import detect_transients
from amp_autoclip.config import DEFAULT as cfg
from amp_autoclip.features import compute_features
from amp_autoclip.model.dataset import cache_features
from amp_autoclip.pose import extract_tracks
from amp_autoclip.segmenter import segment_session

PSEUDO_CONF_FLOOR = 0.60   # UNTUNED: detections below this train nothing


def phases_for_delivery(d: dict, role: str) -> list:
    a = {x["name"]: x["t"] for x in d["anchors"]}
    s, e = d["raw_start_timestamp"], d["raw_end_timestamp"]
    if role == "bowling":
        rel = a["ball_release"]
        bfc = a.get("back_foot_contact") or a.get("front_foot_contact") or rel - 0.3
        return [{"phase": "runup", "start_t": s, "end_t": bfc},
                {"phase": "delivery", "start_t": bfc, "end_t": rel},
                {"phase": "followthrough", "start_t": rel, "end_t": e}]
    imp = a["impact"]
    trig = a.get("trigger_onset") or imp - 0.5
    return [{"phase": "stance", "start_t": s, "end_t": trig},
            {"phase": "shot", "start_t": trig, "end_t": imp},
            {"phase": "followthrough", "start_t": imp, "end_t": e}]


def is_complete(d: dict, role: str) -> bool:
    names = {x["name"] for x in d["anchors"]}
    if role == "bowling":
        return "ball_release" in names and (
            "back_foot_contact" in names or "front_foot_contact" in names)
    return "impact" in names and "stance_set" in names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--role", choices=["bowling", "batting"], required=True)
    p.add_argument("--out", default="dataset")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    cfg.pose.device = args.device

    n_seqs = n_labels = 0
    for path in sorted(glob.glob(os.path.join(args.videos, "*.mp4"))):
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", os.path.splitext(
            os.path.basename(path))[0])[:60]
        print("==", stem)
        video = video_io.probe(path)
        windows = video_io.active_windows(path, cfg, video.duration_s)
        cuts = video_io.detect_scene_cuts(path, cfg)
        transients = detect_transients(path, cfg) if video.has_audio else []
        tracks = extract_tracks(path, windows, cfg, video.fps,
                                (video.width, video.height))
        out = segment_session(video, tracks, transients, cfg, mode=args.role,
                              scene_cuts=cuts)
        grid_fps = min(video.fps, cfg.pose.max_process_fps)
        by_track: dict = {}
        for d in out.deliveries:
            md = d.to_manifest()
            if (d.status == "discard" or d.confidence < PSEUDO_CONF_FLOOR
                    or not is_complete(md, args.role)):
                continue
            by_track.setdefault(d.subject_track_id, []).append(md)
        for track in tracks:
            if track.track_id not in by_track:
                continue
            fs = compute_features(track, grid_fps)
            if fs is None:
                continue
            vid_id = "%s__t%d" % (stem, track.track_id)
            cache_features(fs, vid_id, args.out)
            os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
            with open(os.path.join(args.out, "labels", vid_id + ".jsonl"), "w") as f:
                for md in by_track[track.track_id]:
                    rec = {"video_id": vid_id, "role": args.role,
                           "start_t": md["raw_start_timestamp"],
                           "end_t": md["raw_end_timestamp"],
                           "phases": phases_for_delivery(md, args.role),
                           "anchors": {x["name"]: x["t"] for x in md["anchors"]},
                           "pseudo": True,
                           "source_confidence": md["confidence_score"]}
                    f.write(json.dumps(rec) + "\n")
                    n_labels += 1
            n_seqs += 1
    print("wrote %d sequences, %d pseudo-labeled deliveries" % (n_seqs, n_labels))


if __name__ == "__main__":
    main()
