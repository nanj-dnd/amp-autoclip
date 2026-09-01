"""Job store: Postgres (Supabase) in production, SQLite for local dev.

The queue IS the database — a `clip_jobs` row per session upload, claimed
with FOR UPDATE SKIP LOCKED (Postgres) so any number of workers can pull
safely without a broker. Rationale: the backend already runs on Supabase
Postgres; adding Redis/Celery buys nothing at current volume and adds an
ops surface. If GPU inference later forces a separate service, only the
worker moves — the queue contract stays.

Job lifecycle:
    queued -> processing -> done | review_needed | failed
Failures retry with exponential backoff up to max_attempts, then land in
`failed` with the error preserved — surfaced, never swallowed.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 30.0


@dataclass
class Job:
    id: str
    video_path: str          # storage key or local path
    mode: str
    status: str
    progress: float
    stage: str
    attempts: int
    error: Optional[str]
    result: Optional[Dict]
    created_at: float
    not_before: float


class SQLiteJobStore:
    """Local/dev store. Same interface a PostgresJobStore implements against
    service/schema.sql for production."""

    def __init__(self, path: str = "autoclip_jobs.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS clip_jobs (
                id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'auto',
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL NOT NULL DEFAULT 0,
                stage TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                result TEXT,
                created_at REAL NOT NULL,
                not_before REAL NOT NULL DEFAULT 0
            )""")
        self.db.commit()

    def enqueue(self, video_path: str, mode: str = "auto") -> str:
        job_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO clip_jobs (id, video_path, mode, created_at) "
            "VALUES (?, ?, ?, ?)", (job_id, video_path, mode, time.time()))
        self.db.commit()
        return job_id

    def claim(self) -> Optional[Job]:
        """Atomically claim the oldest runnable job. (Postgres version uses
        SELECT ... FOR UPDATE SKIP LOCKED; SQLite serializes writes anyway.)"""
        cur = self.db.execute(
            "SELECT id FROM clip_jobs WHERE status='queued' AND not_before<=? "
            "ORDER BY created_at LIMIT 1", (time.time(),))
        row = cur.fetchone()
        if not row:
            return None
        n = self.db.execute(
            "UPDATE clip_jobs SET status='processing', attempts=attempts+1 "
            "WHERE id=? AND status='queued'", (row[0],)).rowcount
        self.db.commit()
        return self.get(row[0]) if n else None

    def progress(self, job_id: str, stage: str, frac: float) -> None:
        self.db.execute("UPDATE clip_jobs SET stage=?, progress=? WHERE id=?",
                        (stage, frac, job_id))
        self.db.commit()

    def finish(self, job_id: str, result: Dict, needs_review: bool) -> None:
        status = "review_needed" if needs_review else "done"
        self.db.execute(
            "UPDATE clip_jobs SET status=?, progress=1.0, result=? WHERE id=?",
            (status, json.dumps(result), job_id))
        self.db.commit()

    def fail(self, job_id: str, error: str) -> None:
        job = self.get(job_id)
        if job and job.attempts < MAX_ATTEMPTS:
            delay = BACKOFF_BASE_S * (2 ** (job.attempts - 1))
            self.db.execute(
                "UPDATE clip_jobs SET status='queued', error=?, not_before=? "
                "WHERE id=?", (error, time.time() + delay, job_id))
        else:
            self.db.execute(
                "UPDATE clip_jobs SET status='failed', error=? WHERE id=?",
                (error, job_id))
        self.db.commit()

    def get(self, job_id: str) -> Optional[Job]:
        cur = self.db.execute(
            "SELECT id, video_path, mode, status, progress, stage, attempts, "
            "error, result, created_at, not_before FROM clip_jobs WHERE id=?",
            (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Job(id=row[0], video_path=row[1], mode=row[2], status=row[3],
                   progress=row[4], stage=row[5], attempts=row[6], error=row[7],
                   result=json.loads(row[8]) if row[8] else None,
                   created_at=row[9], not_before=row[10])

    def list_review_queue(self) -> List[Job]:
        cur = self.db.execute(
            "SELECT id FROM clip_jobs WHERE status='review_needed' "
            "ORDER BY created_at")
        return [self.get(r[0]) for r in cur.fetchall()]
