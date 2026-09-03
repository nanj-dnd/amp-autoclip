"""External CSV labels -> training dataset, in the layout model/train.py reads.

Bridges the gap the labeler/ tool leaves open: human labels arrive as CSV
(amp-clip-segments-v1, or the long amp-labels draft-training-labels export),
but the trainer needs per-track cached pose features plus a jsonl keyed by
the SAME video_id as those features.

    python -m scripts.ingest_csv_labels --csv ~/Downloads/*clip-segments-v1.csv \
        --videos ~/Downloads ~/Documents/amp/clips --role batting \
        --out dataset --device mps

    # parse + report only, no videos needed:
    python -m scripts.ingest_csv_labels --csv ~/Downloads/*.csv --dry-run

Writes, per (video, track) that carries at least one label:
    dataset/features/<stem>__t<track>.npz
    dataset/labels/<stem>__t<track>.jsonl

Human labels give boundaries + one impact/contact anchor. Phase intervals and
the stance_set anchor are DERIVED (both flagged UNTUNED below) because no
annotation schema here carries them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from amp_autoclip.pose import Track

# UNTUNED: no schema carries a trigger/backlift onset, so the stance->shot
# boundary is placed this far before impact. Same fallback bootstrap_dataset
# uses. Re-derive from labeled trigger onsets as soon as any exist.
TRIGGER_LEAD_S = 0.50
# UNTUNED: same problem for the bowling runup->delivery boundary.
BFC_LEAD_S = 0.30
# A delivery is assigned to the track overlapping it by at least this much.
MIN_TRACK_OVERLAP = 0.50
# load_samples() fills idle for start-3s..end+3s; two labels closer than this
# would stomp each other's phases, so warn instead of silently corrupting.
IDLE_PAD_S = 3.0


@dataclass
class Seg:
    """One labeled segment, schema-independent."""
    filename: str
    role: str
    start_t: float
    end_t: float
    event_t: Optional[float]
    kind: str = "delivery"          # or "background"
    index: Optional[int] = None
    outcome: Optional[str] = None
    annotator: Optional[str] = None
    fps: Optional[float] = None
    source: str = ""


def _ms(v) -> Optional[float]:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v) / 1000.0
    except ValueError:
        return None


def _read_clip_segments(rows: List[dict], src: str) -> List[Seg]:
    """amp-clip-segments-v1: one row per segment, delivery or background."""
    out = []
    for r in rows:
        s, e = _ms(r.get("start_ms")), _ms(r.get("end_ms"))
        if s is None or e is None or e <= s:
            continue
        if (r.get("exclusion_reason") or "").strip():
            continue
        out.append(Seg(
            filename=r["filename"], role=r.get("discipline", ""),
            start_t=s, end_t=e, event_t=_ms(r.get("event_ms")),
            kind=r.get("segment_type", "delivery"),
            index=int(r["delivery_index"]) if (r.get("delivery_index") or "").strip() else None,
            outcome=(r.get("delivery_outcome") or "").strip() or None,
            annotator=(r.get("annotator") or "").strip() or None,
            fps=float(r["video_fps"]) if (r.get("video_fps") or "").strip() else None,
            source=src))
    return out


def _read_long_labels(rows: List[dict], src: str) -> List[Seg]:
    """draft-training-labels-v2: one row PER KPI, so many rows per delivery.
    Deduped on delivery_id; only rows the pipeline marked training-ready."""
    seen: Dict[str, Seg] = {}
    for r in rows:
        if r.get("target_scope") != "delivery":
            continue
        if r.get("training_row_status") != "ready_scored_label":
            continue
        did = r.get("delivery_id") or ""
        if did in seen:
            continue
        s, e = _ms(r.get("delivery_start_ms")), _ms(r.get("delivery_end_ms"))
        if s is None or e is None or e <= s:
            continue
        seen[did] = Seg(
            filename=r["filename"], role=r.get("discipline", ""),
            start_t=s, end_t=e, event_t=_ms(r.get("event_ms")),
            index=int(r["delivery_index"]) if (r.get("delivery_index") or "").strip() else None,
            outcome=(r.get("delivery_outcome") or "").strip() or None,
            annotator=(r.get("annotator_id") or "").strip() or None,
            fps=float(r["video_fps"]) if (r.get("video_fps") or "").strip() else None,
            source=src)
    return list(seen.values())


def read_csv_labels(paths: List[str]) -> List[Seg]:
    segs: List[Seg] = []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("!! %s: empty" % os.path.basename(p))
            continue
        src = os.path.basename(p)
        schema = rows[0].get("csv_schema_version", "")
        if "clip-segments" in schema or "segment_type" in rows[0]:
            got = _read_clip_segments(rows, src)
        elif "delivery_start_ms" in rows[0]:
            got = _read_long_labels(rows, src)
        else:
            print("!! %s: unrecognized schema %r — skipped" % (src, schema))
            continue
        print("   %-52s %3d rows -> %2d segments (%s)"
              % (src[:52], len(rows), len(got), schema or "inferred"))
        segs.extend(got)
    return segs


def derive_phases(seg: Seg, role: str) -> List[dict]:
    """Boundaries + impact anchor -> the phase intervals load_samples wants."""
    s, e = seg.start_t, seg.end_t
    ev = seg.event_t if seg.event_t is not None else (s + e) / 2.0
    if role == "bowling":
        bfc = max(s, ev - BFC_LEAD_S)
        return [{"phase": "runup", "start_t": s, "end_t": bfc},
                {"phase": "delivery", "start_t": bfc, "end_t": ev},
                {"phase": "followthrough", "start_t": ev, "end_t": e}]
    trig = max(s, ev - TRIGGER_LEAD_S)
    return [{"phase": "stance", "start_t": s, "end_t": trig},
            {"phase": "shot", "start_t": trig, "end_t": ev},
            {"phase": "followthrough", "start_t": ev, "end_t": e}]


def derive_anchors(seg: Seg, role: str, derive_stance_set: bool) -> Dict[str, float]:
    ev = seg.event_t
    if role == "bowling":
        return {"ball_release": ev} if ev is not None else {}
    a: Dict[str, float] = {}
    if ev is not None:
        a["impact"] = ev
    # The event loss is unmasked BCE, so an absent channel trains as "never
    # fires" rather than "unknown". Deriving stance_set from the annotator's
    # delivery start teaches their setup convention instead. UNTUNED.
    if derive_stance_set:
        a["stance_set"] = seg.start_t
    return a


def pick_track(tracks: List["Track"], seg: Seg, role: str, w: int, h: int):
    from amp_autoclip.pose import score_roles

    best, best_key = None, None
    for tr in tracks:
        lo, hi = max(tr.t0, seg.start_t), min(tr.t1, seg.end_t)
        cover = max(0.0, hi - lo) / max(seg.end_t - seg.start_t, 1e-6)
        if cover < MIN_TRACK_OVERLAP:
            continue
        rs = score_roles(tr, w, h)
        key = (rs.bowler if role == "bowling" else rs.batter, cover)
        if best_key is None or key > best_key:
            best, best_key = tr, key
    return best


def warn_crowding(segs: List[Seg], label: str) -> None:
    segs = sorted(segs, key=lambda s: s.start_t)
    for a, b in zip(segs, segs[1:]):
        if b.start_t - a.end_t < IDLE_PAD_S:
            print("   !! %s: labels %.2fs apart (%.2f->%.2f, %.2f->%.2f); "
                  "the idle pre-roll will overwrite the earlier phases"
                  % (label, b.start_t - a.end_t, a.start_t, a.end_t,
                     b.start_t, b.end_t))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--videos", nargs="*", default=[],
                   help="directories searched for each CSV's filename column")
    p.add_argument("--role", choices=["bowling", "batting"], default=None,
                   help="defaults to whatever discipline the CSVs carry")
    p.add_argument("--out", default="dataset")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-stance-set", action="store_true",
                   help="omit the derived stance_set anchor (see derive_anchors)")
    p.add_argument("--windows", choices=["labels", "motion"], default="labels",
                   help="'labels' runs pose over each labeled span plus "
                        "--window-margin of context (so the annotator's start "
                        "is inside the feature grid); 'motion' uses the "
                        "engine's motion gate, which starts tracks AFTER the "
                        "labeled start and makes start boundaries unlearnable")
    p.add_argument("--window-margin", type=float, default=3.0,
                   help="seconds of context kept either side of each label")
    p.add_argument("--dry-run", action="store_true",
                   help="parse and report only; no videos or pose needed")
    args = p.parse_args()

    paths: List[str] = []
    for c in args.csv:
        paths.extend(sorted(glob.glob(os.path.expanduser(c))))
    if not paths:
        raise SystemExit("no CSVs matched")

    print("Parsing %d CSV(s):" % len(paths))
    segs = read_csv_labels(paths)
    if not segs:
        raise SystemExit("no usable segments")

    roles = {s.role for s in segs}
    role = args.role or (roles.pop() if len(roles) == 1 else None)
    if role is None:
        raise SystemExit("mixed disciplines %s — pass --role" % sorted(roles))
    segs = [s for s in segs if s.role == role]

    by_file: Dict[str, List[Seg]] = {}
    for s in segs:
        by_file.setdefault(s.filename, []).append(s)

    print("\nRole %s — %d segment(s) across %d video(s):" % (role, len(segs), len(by_file)))
    resolved: Dict[str, Optional[str]] = {}
    for fn, group in sorted(by_file.items()):
        d = [g for g in group if g.kind == "delivery"]
        bg = [g for g in group if g.kind == "background"]
        path = None
        for root in args.videos:
            cand = os.path.join(os.path.expanduser(root), fn)
            if os.path.exists(cand):
                path = cand
                break
        resolved[fn] = path
        print("   %-46s %2d deliveries, %2d background  %s"
              % (fn[:46], len(d), len(bg), "OK" if path else "VIDEO NOT FOUND"))
        warn_crowding(d, fn[:30])

    missing = [f for f, p in resolved.items() if p is None]
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    from amp_autoclip import video_io
    from amp_autoclip.config import DEFAULT as cfg
    from amp_autoclip.features import compute_features
    from amp_autoclip.model.dataset import cache_features
    from amp_autoclip.pose import extract_tracks

    cfg.pose.device = args.device
    if missing:
        print("\n!! %d video(s) not found under %s; those labels are skipped."
              % (len(missing), args.videos or "(no --videos given)"))

    n_seq = n_lab = 0
    for fn, group in sorted(by_file.items()):
        path = resolved[fn]
        if not path:
            continue
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", os.path.splitext(fn)[0])[:60]
        print("\n== %s" % stem)
        video = video_io.probe(path)
        if args.windows == "labels":
            spans = sorted((max(0.0, g.start_t - args.window_margin),
                            min(video.duration_s, g.end_t + args.window_margin))
                           for g in group)
            windows = []
            for a, b in spans:
                if windows and a <= windows[-1][1]:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], b))
                else:
                    windows.append((a, b))
        else:
            windows = video_io.active_windows(path, cfg, video.duration_s)
        tracks = extract_tracks(path, windows, cfg, video.fps,
                                (video.width, video.height))
        grid_fps = min(video.fps, cfg.pose.max_process_fps)
        by_track: Dict[int, List[Seg]] = {}
        for seg in group:
            tr = pick_track(tracks, seg, role, video.width, video.height)
            if tr is None:
                print("   no track covers %.2f-%.2f (%s) — skipped"
                      % (seg.start_t, seg.end_t, seg.kind))
                continue
            by_track.setdefault(tr.track_id, []).append(seg)
        for tr in tracks:
            group_segs = by_track.get(tr.track_id)
            if not group_segs:
                continue
            fs = compute_features(tr, grid_fps)
            if fs is None:
                print("   track %d: features unavailable — skipped" % tr.track_id)
                continue
            vid_id = "%s__t%d" % (stem, tr.track_id)
            cache_features(fs, vid_id, args.out)
            os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
            # background first: a later record's idle pre-roll must not be the
            # thing that lands on a delivery's phases
            group_segs.sort(key=lambda s: (s.kind != "background", s.start_t))
            with open(os.path.join(args.out, "labels", vid_id + ".jsonl"), "w") as f:
                for seg in group_segs:
                    lo = max(seg.start_t, tr.t0)
                    hi = min(seg.end_t, tr.t1)
                    if hi - lo < 0.2:
                        continue
                    if seg.kind == "delivery" and (lo - seg.start_t > 0.05
                                                   or seg.end_t - hi > 0.05):
                        print("   !! track %d covers only %.2f-%.2f of labeled "
                              "%.2f-%.2f — boundary is outside the feature "
                              "grid and cannot be learned"
                              % (tr.track_id, lo, hi, seg.start_t, seg.end_t))
                    if seg.kind == "background":
                        rec = {"video_id": vid_id, "role": role,
                               "start_t": lo, "end_t": hi,
                               "phases": [{"phase": "idle", "start_t": lo,
                                           "end_t": hi}],
                               "anchors": {}, "background": True,
                               "source_csv": seg.source,
                               "verified_by": seg.annotator}
                    else:
                        rec = {"video_id": vid_id, "role": role,
                               "start_t": seg.start_t, "end_t": seg.end_t,
                               "phases": derive_phases(seg, role),
                               "anchors": derive_anchors(
                                   seg, role, not args.no_stance_set),
                               "delivery_index": seg.index,
                               "outcome": seg.outcome,
                               "derived_phases": True,
                               "source_csv": seg.source,
                               "verified_by": seg.annotator}
                        n_lab += 1
                    f.write(json.dumps(rec) + "\n")
            n_seq += 1
            print("   track %d -> %s (%d segments)"
                  % (tr.track_id, vid_id, len(group_segs)))

    print("\nwrote %d sequence(s), %d labeled deliveries into %s/"
          % (n_seq, n_lab, args.out))
    if n_lab:
        print("next: python -m amp_autoclip.model.train --data %s/ --role %s "
              "--out models/%s_v1.pt" % (args.out, role, role))


if __name__ == "__main__":
    main()
