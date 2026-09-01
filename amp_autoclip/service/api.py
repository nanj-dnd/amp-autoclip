"""FastAPI router — mounts into the existing backend:

    from amp_autoclip.service.api import router as autoclip_router
    app.include_router(autoclip_router, prefix="/autoclip")

Trigger flow: the existing upload-complete handler (or a Supabase Storage
webhook) POSTs /autoclip/jobs with the storage key. Everything else is
reads plus the review endpoints the manual-trim UI calls.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .jobs import SQLiteJobStore

router = APIRouter(tags=["autoclip"])
_store: Optional[SQLiteJobStore] = None


def get_store() -> SQLiteJobStore:
    global _store
    if _store is None:
        _store = SQLiteJobStore()
    return _store


class CreateJob(BaseModel):
    video_path: str          # storage key of the finished session upload
    mode: str = "auto"       # bowling | batting | both | auto


class Retrim(BaseModel):
    start_timestamp: float
    end_timestamp: float


@router.post("/jobs", status_code=202)
def create_job(body: CreateJob):
    if body.mode not in ("bowling", "batting", "both", "auto"):
        raise HTTPException(422, "invalid mode")
    job_id = get_store().enqueue(body.video_path, body.mode)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    out = {"job_id": job.id, "status": job.status, "stage": job.stage,
           "progress": job.progress, "attempts": job.attempts}
    if job.status == "failed":
        out["error"] = job.error
    if job.result is not None:
        out["manifest"] = job.result
    return out


@router.get("/review")
def review_queue():
    """Sessions holding low-confidence detections. The app's manual-trim UI
    is seeded with the engine's best-guess boundaries — the user adjusts,
    never trims from scratch."""
    jobs = get_store().list_review_queue()
    return {"jobs": [{
        "job_id": j.id,
        "flags": (j.result or {}).get("session_flags", []),
        "deliveries": [d for d in (j.result or {}).get("deliveries", [])
                       if d["status"] == "review"],
    } for j in jobs]}


@router.post("/review/{job_id}/deliveries/{ordinal}/approve")
def approve(job_id: str, ordinal: int):
    """Reviewer accepted the candidate boundaries as-is. In production this
    updates the deliveries row to 'approved' and releases the clip to frame
    extraction; the approval is also a positive training label."""
    return _review_action(job_id, ordinal, "approved", None)


@router.post("/review/{job_id}/deliveries/{ordinal}/retrim")
def retrim(job_id: str, ordinal: int, body: Retrim):
    """Reviewer adjusted the boundaries. The corrected pair is gold training
    data — it flows into the labels table for the next retraining pass."""
    if body.end_timestamp <= body.start_timestamp:
        raise HTTPException(422, "end must be after start")
    return _review_action(job_id, ordinal, "retrimmed",
                          (body.start_timestamp, body.end_timestamp))


@router.post("/review/{job_id}/deliveries/{ordinal}/reject")
def reject(job_id: str, ordinal: int):
    """Not a delivery (false positive) — also a training label."""
    return _review_action(job_id, ordinal, "rejected", None)


def _review_action(job_id: str, ordinal: int, action: str, bounds):
    store = get_store()
    job = store.get(job_id)
    if job is None or job.result is None:
        raise HTTPException(404, "no such job or job not finished")
    deliveries = job.result.get("deliveries", [])
    if not (0 <= ordinal < len(deliveries)):
        raise HTTPException(404, "no such delivery")
    d = deliveries[ordinal]
    d["status"] = action
    if bounds:
        d["corrected_start_timestamp"], d["corrected_end_timestamp"] = bounds
    still_pending = any(x["status"] == "review" for x in deliveries)
    store.finish(job_id, job.result, needs_review=still_pending)
    return {"ok": True, "delivery": d}
