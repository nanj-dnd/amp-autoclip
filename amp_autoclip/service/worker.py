"""Worker process: claims clip jobs, runs the engine, writes outputs.

Run as its own process (Railway/Render worker dyno, same repo):

    python -m amp_autoclip.service.worker --db autoclip_jobs.db --out-root clips/

Separate from the API workers on purpose — a 5-minute session must never
occupy a request worker. Progress is written to the job row, so the app
can poll GET /autoclip/jobs/{id} (or subscribe via Supabase Realtime on
clip_jobs) and render "your session is being analysed" with a live stage
and percentage instead of blocking.

In production the video_path is a storage key: download to a temp file
first, run, upload clips back, then finish the job. The hook points are
marked below.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import traceback

from ..clipper import write_outputs
from ..config import DEFAULT
from ..pipeline import run_session
from .jobs import Job, SQLiteJobStore

log = logging.getLogger("amp_autoclip.worker")


def process_job(store: SQLiteJobStore, job: Job, out_root: str) -> None:
    # PRODUCTION HOOK: download job.video_path from cloud storage to a local
    # temp file here; the engine only ever sees a local file.
    local_path = job.video_path

    def progress(stage: str, frac: float) -> None:
        store.progress(job.id, stage, frac)

    result = run_session(local_path, mode=job.mode, cfg=DEFAULT,
                         progress=progress)
    out_dir = os.path.join(out_root, job.id)
    manifest = write_outputs(result, out_dir, DEFAULT)
    # PRODUCTION HOOK: upload out_dir clips + manifest to cloud storage and
    # insert `deliveries` rows here; downstream frame extraction is triggered
    # per delivery with status 'auto' only.
    needs_review = any(d["status"] == "review" for d in manifest["deliveries"]) \
        or bool(manifest["session_flags"])
    store.finish(job.id, manifest, needs_review=needs_review)


def run_worker(store: SQLiteJobStore, out_root: str,
               poll_s: float = 2.0, once: bool = False) -> None:
    log.info("worker up, polling every %.1fs", poll_s)
    while True:
        job = store.claim()
        if job is None:
            if once:
                return
            time.sleep(poll_s)
            continue
        log.info("job %s claimed (attempt %d): %s", job.id, job.attempts,
                 job.video_path)
        try:
            process_job(store, job, out_root)
            log.info("job %s finished", job.id)
        except Exception as exc:   # noqa: BLE001 — every failure must land in the row
            log.error("job %s failed: %s", job.id, exc)
            store.fail(job.id, "%s\n%s" % (exc, traceback.format_exc()))
        if once:
            return


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="autoclip_jobs.db")
    p.add_argument("--out-root", default="clips")
    p.add_argument("--once", action="store_true", help="process one job and exit")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    run_worker(SQLiteJobStore(args.db), args.out_root, once=args.once)


if __name__ == "__main__":
    main()
