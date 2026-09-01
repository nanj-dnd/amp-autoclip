from __future__ import annotations

import os

from amp_autoclip.eval.evaluate import evaluate
from amp_autoclip.service.jobs import MAX_ATTEMPTS, SQLiteJobStore


def _pred(start, end, typ="bowling", conf=0.8, status="auto", anchors=None):
    return {"start_timestamp": start, "end_timestamp": end, "type": typ,
            "confidence_score": conf, "status": status,
            "anchors": anchors or []}


def _truth(start, end, role="bowling", anchors=None):
    return {"start_t": start, "end_t": end, "role": role,
            "anchors": anchors or {}}


def test_eval_detection_and_truncation():
    preds = [
        _pred(1.0, 9.0, anchors=[{"name": "ball_release", "t": 6.02}]),
        _pred(20.5, 28.0),          # starts late: truncates the run-up
        _pred(50.0, 55.0),          # hallucinated
    ]
    truths = [
        _truth(1.2, 8.8, anchors={"ball_release": 6.0}),
        _truth(19.0, 28.2),
        _truth(35.0, 43.0),         # missed entirely
    ]
    r = evaluate(preds, truths)
    assert r["detection_recall"] < 1.0 and r["detection_precision"] < 1.0
    assert r["phase_truncation_rate"] == 0.5      # 1 of 2 matched clips truncates
    assert r["anchor_errors_ms"]["ball_release"]["median_abs_ms"] < 66
    assert r["anchor_within_tolerance"]["ball_release"] == 1.0


def test_eval_perfect():
    preds = [_pred(0.5, 10.0)]
    truths = [_truth(1.0, 9.5)]     # pred fully padded outward: fine
    r = evaluate(preds, truths)
    assert r["detection_precision"] == 1.0
    assert r["detection_recall"] == 1.0
    assert r["phase_truncation_rate"] == 0.0


def test_job_lifecycle_and_retry(tmp_path):
    store = SQLiteJobStore(os.path.join(str(tmp_path), "jobs.db"))
    job_id = store.enqueue("/x/session.mp4", "bowling")

    job = store.claim()
    assert job is not None and job.id == job_id and job.status == "processing"
    assert store.claim() is None                 # nothing else queued

    store.fail(job_id, "boom")                   # retry 1: back to queued
    j = store.get(job_id)
    assert j.status == "queued" and j.not_before > 0

    for _ in range(MAX_ATTEMPTS - 1):            # exhaust attempts
        j = store.get(job_id)
        store.db.execute("UPDATE clip_jobs SET not_before=0 WHERE id=?", (job_id,))
        store.db.commit()
        assert store.claim() is not None
        store.fail(job_id, "boom again")
    j = store.get(job_id)
    assert j.status == "failed" and "boom" in j.error   # surfaced, not swallowed

    job2 = store.enqueue("/y/other.mp4")
    store.claim()
    store.finish(job2, {"deliveries": [{"status": "review"}]}, needs_review=True)
    assert store.get(job2).status == "review_needed"
    assert [x.id for x in store.list_review_queue()] == [job2]
