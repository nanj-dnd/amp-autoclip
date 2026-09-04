"""Calibrate padding and confidence against already-clipped deliveries.

Runs the heuristic engine over a folder of pre-cut single-delivery clips
(e.g. a ball-tracking system's fixed-window exports) where the answer is
known to be "exactly one delivery, filling the clip". Two uses:

1. CONFIDENCE CALIBRATION. Every clip is a guaranteed positive, so the
   detection rate and the confidence/status distribution measure how
   pessimistic the engine is on footage where it should be certain.

2. PADDING. The offset of the impact anchor inside each clip reports the
   pre/post-roll a working production system chose to keep, which is
   directly comparable to batting.pre_roll_ms / post_roll_ms (both UNTUNED).

    PYTHONPATH=. .venv/bin/python scripts/calibrate_from_clips.py \
        --clips '~/Documents/amp/drive_sessions/*/*.MP4'

CAVEAT, read before drawing conclusions: the "boundary hit the file edge"
counters do NOT measure whether the fixed window truncated the delivery.
The engine's follow-through resolution needs sustained stillness, which a
short clip rarely contains, so it runs to the last frame regardless. On the
84-clip set this fired for 65% of clips — but those clips had 2.34s of
footage after impact versus 2.41s for the ones that resolved cleanly, i.e.
no less room. Split by footage-after-impact before believing an edge count.
"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
from amp_autoclip.config import DEFAULT as cfg
from amp_autoclip import video_io
from amp_autoclip.pipeline import run_session

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True, help="glob of single-delivery clips")
ap.add_argument("--role", default="batting", choices=["batting", "bowling"])
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="clip_calibration.json")
args = ap.parse_args()
cfg.pose.device = args.device

EDGE = 0.10          # within 100ms of a file edge counts as "hit the edge"
rows = []
files = sorted(glob.glob(os.path.expanduser(args.clips)))
for i, p in enumerate(files, 1):
    dur = video_io.probe(p).duration_s
    try:
        res = run_session(p, mode=args.role, cfg=cfg)
        ds = [d for d in res.deliveries if d.status != "discard"]
    except Exception as e:
        rows.append({"file": p, "dur": dur, "error": str(e)[:80]}); continue
    if not ds:
        rows.append({"file": p, "dur": dur, "detected": 0}); continue
    d = max(ds, key=lambda x: x.confidence)
    imp = d.anchor("impact")
    rows.append({
        "file": p, "dur": round(dur, 3), "detected": len(ds),
        "raw_start": round(d.raw_start_t, 3), "raw_end": round(d.raw_end_t, 3),
        "impact": round(imp.t, 3) if imp else None,
        "conf": round(d.confidence, 3), "status": d.status,
        "start_at_edge": d.raw_start_t <= EDGE,
        "end_at_edge": d.raw_end_t >= dur - EDGE,
    })
    if i % 10 == 0:
        print("  ...%d/%d" % (i, len(files)), file=sys.stderr, flush=True)
json.dump(rows, open(args.out, "w"), indent=1)
ok = [r for r in rows if r.get("detected", 0) > 0]
imp = [r for r in ok if r.get("impact")]
print("clips %d | detected %d | no detection %d"
      % (len(rows), len(ok), sum(1 for r in rows if r.get("detected") == 0)))
if imp:
    pre = sorted(r["impact"] for r in imp)
    post = sorted(r["dur"] - r["impact"] for r in imp)
    mid = len(imp) // 2
    print("median %.2fs before impact, %.2fs after  (cf. %s.pre_roll_ms / "
          "post_roll_ms)" % (pre[mid], post[mid], args.role))
if ok:
    st = {}
    for r in ok:
        st[r["status"]] = st.get(r["status"], 0) + 1
    conf = sorted(r["conf"] for r in ok)
    print("status %s | median confidence %.2f" % (st, conf[len(conf) // 2]))
print("wrote %s" % args.out)
