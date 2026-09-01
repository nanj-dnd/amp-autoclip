"""Evaluation against labeled ground truth. Nothing ships until these
numbers exist on a held-out test split.

Metrics, in order of importance:
1. PHASE TRUNCATION RATE — % of matched clips where a predicted boundary
   cuts into a labeled scoreable phase (predicted start after true start,
   or predicted end before true end). This is THE gating metric: a
   truncated phase silently corrupts a KPI score downstream. Target ~0
   among auto-shipped clips.
2. Detection precision / recall — GT deliveries matched at IoU >= 0.5.
   A missed or hallucinated delivery is worse than a sloppy boundary.
3. Boundary errors — signed ms error per outer boundary, judged against the
   loose tolerance (outward misses are cheap, inward misses are counted by
   metric 1).
4. Anchor errors — BFC/FFC/release/impact timing vs the tight tolerance.
5. Confidence calibration — do 0.8-confidence clips succeed ~80% of the
   time? Decides whether the auto/review thresholds mean anything.

Usage:
    python -m amp_autoclip.eval.evaluate --pred clips/manifest.json \
        --truth labels/session01.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np

from ..config import DEFAULT, EngineConfig


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _match(preds: List[Dict], truths: List[Dict], iou_thresh: float):
    """Greedy IoU matching within same delivery type."""
    pairs, used_p, used_t = [], set(), set()
    cands = []
    for pi, p in enumerate(preds):
        for ti, t in enumerate(truths):
            if p["type"] != t.get("role", t.get("type")):
                continue
            iou = _iou(p["start_timestamp"], p["end_timestamp"],
                       t["start_t"], t["end_t"])
            if iou >= iou_thresh:
                cands.append((iou, pi, ti))
    for iou, pi, ti in sorted(cands, reverse=True):
        if pi in used_p or ti in used_t:
            continue
        pairs.append((pi, ti, iou))
        used_p.add(pi)
        used_t.add(ti)
    return pairs, used_p, used_t


def evaluate(preds: List[Dict], truths: List[Dict],
             cfg: EngineConfig = DEFAULT) -> Dict:
    kept = [p for p in preds if p.get("status") != "discard"]
    pairs, used_p, used_t = _match(kept, truths, cfg.eval.iou_match_threshold)

    n_trunc = 0
    start_err, end_err, anchor_errs = [], [], {}
    conf_outcomes = []   # (confidence, matched_and_untruncated)
    for pi, ti, iou in pairs:
        p, t = kept[pi], truths[ti]
        se = (p["start_timestamp"] - t["start_t"]) * 1000.0   # + = late start
        ee = (p["end_timestamp"] - t["end_t"]) * 1000.0       # - = early end
        start_err.append(se)
        end_err.append(ee)
        truncated = se > cfg.eval.boundary_tolerance_ms or \
            ee < -cfg.eval.boundary_tolerance_ms
        n_trunc += int(truncated)
        conf_outcomes.append((p["confidence_score"], not truncated))

        true_anchors = t.get("anchors") or {}
        pred_anchors = {a["name"]: a["t"] for a in p.get("anchors", [])}
        for name, tt in true_anchors.items():
            if tt is None or name not in pred_anchors:
                continue
            anchor_errs.setdefault(name, []).append(
                (pred_anchors[name] - tt) * 1000.0)

    for p in kept:
        if kept.index(p) not in used_p:
            conf_outcomes.append((p["confidence_score"], False))

    precision = len(pairs) / len(kept) if kept else 0.0
    recall = len(pairs) / len(truths) if truths else 0.0

    def stats(errs: List[float]) -> Optional[Dict]:
        if not errs:
            return None
        a = np.asarray(errs)
        return {"mean_ms": round(float(np.mean(a)), 1),
                "median_abs_ms": round(float(np.median(np.abs(a))), 1),
                "p95_abs_ms": round(float(np.percentile(np.abs(a), 95)), 1),
                "n": len(a)}

    anchors_within = {
        name: round(float(np.mean(np.abs(np.asarray(v))
                                  <= cfg.eval.anchor_tolerance_ms)), 3)
        for name, v in anchor_errs.items()}

    # calibration: mean success rate per confidence bin
    bins: Dict[str, Dict] = {}
    for lo in (0.0, 0.4, 0.6, 0.75, 0.9):
        hi = {0.0: 0.4, 0.4: 0.6, 0.6: 0.75, 0.75: 0.9, 0.9: 1.01}[lo]
        inb = [ok for c, ok in conf_outcomes if lo <= c < hi]
        if inb:
            bins["%.2f-%.2f" % (lo, hi)] = {
                "n": len(inb), "success_rate": round(float(np.mean(inb)), 3)}

    return {
        "n_truth": len(truths), "n_pred_kept": len(kept),
        "detection_precision": round(precision, 3),
        "detection_recall": round(recall, 3),
        "phase_truncation_rate": round(n_trunc / len(pairs), 3) if pairs else None,
        "start_boundary": stats(start_err),
        "end_boundary": stats(end_err),
        "anchor_errors_ms": {k: stats(v) for k, v in anchor_errs.items()},
        "anchor_within_tolerance": anchors_within,
        "confidence_calibration": bins,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="manifest.json from the engine")
    p.add_argument("--truth", required=True, help="labeler JSONL ground truth")
    args = p.parse_args()
    with open(args.pred) as f:
        preds = json.load(f)["deliveries"]
    truths = []
    with open(args.truth) as f:
        for line in f:
            if line.strip():
                truths.append(json.loads(line))
    print(json.dumps(evaluate(preds, truths), indent=2))


if __name__ == "__main__":
    main()
