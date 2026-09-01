"""Internal labeling tool: scrub footage, verify/correct engine pre-labels,
export ground truth in the exact rubric label space.

    .venv/bin/uvicorn labeler.app:app --port 8017
    open http://localhost:8017/?dir=/path/to/session/videos

Workflow (bootstrap-first, per the plan in DESIGN.md):
1. "Prelabel" runs the heuristic engine on the video — humans VERIFY and
   CORRECT candidate segments instead of labeling from scratch.
2. Keyboard-first boundary adjustment (frame stepping), anchor marking.
3. Save validates against the rubric (labels.py is the source of truth) and
   derives phase intervals mechanically from boundaries + anchors, so the
   stored labels always match the model's label space exactly.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from amp_autoclip.config import DEFAULT
from amp_autoclip.model import labels as L
from amp_autoclip.pipeline import run_session

if os.environ.get("AMP_POSE_DEVICE"):
    DEFAULT.pose.device = os.environ["AMP_POSE_DEVICE"]

app = FastAPI(title="Amp autoclip labeler")
STATIC = os.path.join(os.path.dirname(__file__), "static")


def _video_dir(d: str) -> str:
    if not os.path.isdir(d):
        raise HTTPException(404, "no such directory: %s" % d)
    return d


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/videos")
def list_videos(dir: str = Query(...)):
    d = _video_dir(dir)
    vids = sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".mp4", ".mov", ".m4v")))
    return {"videos": vids}


@app.get("/api/video")
def serve_video(dir: str = Query(...), name: str = Query(...)):
    path = os.path.join(_video_dir(dir), os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "no such video")
    return FileResponse(path)


@app.post("/api/upload")
async def upload_video(dir: str = Query(...), file: UploadFile = File(...)):
    """Upload a session video from the browser (for testing from another
    device). Streams to disk in chunks so large videos don't sit in memory."""
    d = _video_dir(dir)
    name = os.path.basename(file.filename or "upload.mp4")
    if not name.lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(422, "only .mp4 / .mov / .m4v videos")
    path = os.path.join(d, name)
    size = 0
    with open(path, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)
    if size < 1000:
        os.remove(path)
        raise HTTPException(422, "file looks empty")
    return {"saved": name, "bytes": size}


@app.get("/api/prelabel")
def prelabel(dir: str = Query(...), name: str = Query(...),
             mode: str = Query("auto")):
    """Heuristic pre-labels so humans only verify and correct. Cached — the
    engine takes a while on long sessions."""
    d = _video_dir(dir)
    path = os.path.join(d, os.path.basename(name))
    cache = path + ".prelabel.%s.json" % mode
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    result = run_session(path, mode=mode, cfg=DEFAULT)
    manifest = result.to_manifest()
    with open(cache, "w") as f:
        json.dump(manifest, f)
    return manifest


@app.get("/api/autoclip")
def autoclip(dir: str = Query(...), name: str = Query(...),
             mode: str = Query("auto")):
    """The actual product flow: session video -> cut per-delivery clips.
    Runs detection (cached after first run), then cuts one mp4 per
    non-discarded delivery next to the source video."""
    from amp_autoclip.clipper import cut_clip
    manifest = prelabel(dir=dir, name=name, mode=mode)
    d = _video_dir(dir)
    src = os.path.join(d, os.path.basename(name))
    vid = os.path.splitext(os.path.basename(name))[0]
    out_dir = os.path.join(d, vid + "_clips")
    os.makedirs(out_dir, exist_ok=True)
    clips, n = [], 0
    for md in manifest["deliveries"]:
        if md["status"] == "discard":
            continue
        n += 1
        fname = "delivery_%02d_%s.mp4" % (n, md["type"])
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            cut_clip(src, fpath, md["start_timestamp"], md["end_timestamp"],
                     DEFAULT)
        clips.append(dict(md, clip_file=fname))
    return {"video": manifest["video"],
            "session_flags": manifest["session_flags"],
            "n_detected": len(manifest["deliveries"]), "clips": clips}


@app.get("/api/clip")
def serve_clip(dir: str = Query(...), name: str = Query(...),
               file: str = Query(...)):
    vid = os.path.splitext(os.path.basename(name))[0]
    path = os.path.join(_video_dir(dir), vid + "_clips", os.path.basename(file))
    if not os.path.exists(path):
        raise HTTPException(404, "no such clip")
    return FileResponse(path, media_type="video/mp4",
                        filename=os.path.basename(file))


class SegmentLabel(BaseModel):
    role: str
    start_t: float
    end_t: float
    anchors: dict = {}
    shot_type: Optional[str] = None
    rejected: bool = False           # marked not-a-delivery
    verified_by: str = ""


class SaveBody(BaseModel):
    segments: List[SegmentLabel]


def _derive_phases(seg: SegmentLabel) -> List[dict]:
    """Mechanical phase intervals from boundaries + anchors, per the rubric.
    Keeps human effort on boundaries/anchors while the stored label still
    covers the model's full phase space."""
    a = seg.anchors
    ph = []
    if seg.role == "bowling":
        rel = a.get("ball_release")
        bfc = a.get("back_foot_contact") or (rel - 0.3 if rel else None)
        if rel is None:
            return []
        ph.append({"phase": "runup", "start_t": seg.start_t, "end_t": bfc})
        ph.append({"phase": "delivery", "start_t": bfc, "end_t": rel})
        ph.append({"phase": "followthrough", "start_t": rel, "end_t": seg.end_t})
    else:
        trigger = a.get("trigger_onset")
        impact = a.get("impact")
        shot_end = impact if impact is not None else \
            (seg.start_t + seg.end_t) / 2   # leave: nominal midpoint
        stance_end = trigger if trigger is not None else \
            max(seg.start_t, shot_end - 0.5)   # nominal backswing when unlabeled
        ph.append({"phase": "stance", "start_t": seg.start_t, "end_t": stance_end})
        ph.append({"phase": "shot", "start_t": stance_end, "end_t": shot_end})
        ph.append({"phase": "followthrough", "start_t": shot_end,
                   "end_t": seg.end_t})
    return ph


def _validate(seg: SegmentLabel) -> Optional[str]:
    if seg.rejected:
        return None
    if seg.role not in ("bowling", "batting"):
        return "role must be bowling or batting"
    if seg.end_t <= seg.start_t:
        return "end before start"
    if seg.role == "bowling" and seg.anchors.get("ball_release") is None:
        return "bowling segment needs a ball_release anchor"
    if seg.role == "batting":
        if seg.shot_type not in ("swing", "leave"):
            return "batting segment needs shot_type swing|leave"
        if seg.shot_type == "swing" and seg.anchors.get("impact") is None:
            return "swing needs an impact anchor"
        if seg.anchors.get("stance_set") is None:
            return "batting segment needs a stance_set anchor"
    for name in seg.anchors:
        if name not in L.BOWLING_EVENTS + L.BATTING_EVENTS + ["trigger_onset"]:
            return "unknown anchor: %s" % name
    return None


@app.post("/api/labels")
def save_labels(body: SaveBody, dir: str = Query(...), name: str = Query(...)):
    d = _video_dir(dir)
    video_id = os.path.splitext(os.path.basename(name))[0]
    errors = []
    lines = []
    for i, seg in enumerate(body.segments):
        err = _validate(seg)
        if err:
            errors.append({"segment": i, "error": err})
            continue
        if seg.rejected:
            rec = {"video_id": video_id, "role": seg.role, "rejected": True,
                   "start_t": seg.start_t, "end_t": seg.end_t,
                   "verified_by": seg.verified_by}
        else:
            rec = {"video_id": video_id, "role": seg.role,
                   "start_t": seg.start_t, "end_t": seg.end_t,
                   "phases": _derive_phases(seg), "anchors": seg.anchors,
                   "shot_type": seg.shot_type, "verified_by": seg.verified_by}
        lines.append(json.dumps(rec))
    if errors:
        raise HTTPException(422, detail=errors)
    out = os.path.join(d, video_id + ".jsonl")
    with open(out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return {"saved": len(lines), "path": out}


@app.get("/api/labels")
def load_labels(dir: str = Query(...), name: str = Query(...)):
    d = _video_dir(dir)
    video_id = os.path.splitext(os.path.basename(name))[0]
    path = os.path.join(d, video_id + ".jsonl")
    if not os.path.exists(path):
        return {"segments": []}
    segs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                segs.append(json.loads(line))
    return {"segments": segs}
