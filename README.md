# amp-autoclip

Auto-clipping engine for Amp Intelligence: takes a continuous net-session
video (1–20+ deliveries) and outputs one trimmed clip per delivery, ready for
the existing frame-extraction → KPI pipeline unchanged. Sits between "session
upload finished" and "ffmpeg frame extraction". See [DESIGN.md](DESIGN.md) for
the full architecture, boundary spec encoding, data/training plan, and
evaluation gates.

> Every numeric threshold is an unvalidated starting point — they all live in
> [`amp_autoclip/config.py`](amp_autoclip/config.py), flagged. Nothing here is
> production-ready until it has been evaluated against labeled ground truth.

## Layout

```
amp_autoclip/
  config.py            ALL thresholds (untuned, flagged) — nothing hardcoded elsewhere
  pipeline.py          run_session(video) -> SessionResult   (engine entrypoint)
  video_io.py          probe, decode, motion gating
  pose.py              YOLO11-pose, track persistence, role inference
  features.py          kinematic series (torso-normalized) + model feature matrix
  audio.py             transient channel (corroboration only)
  events_bowling.py    run-up initiation -> follow-through resolution + BFC/FFC/BR
  events_batting.py    stance set -> follow-through resolution + impact/leaves/aborts
  segmenter.py         padding, plausibility, confidence, dedupe, role suppression
  refine.py            fine pass: native-fps anchor re-timing per detected delivery
  clipper.py           ffmpeg cuts + manifest.json
  service/             FastAPI router + worker + Postgres/SQLite job queue + schema.sql
  model/               MS-TCN temporal model: label space, dataset, train, infer
  eval/                IoU/boundary/anchor/calibration evaluation vs ground truth
scripts/               bootstrap_dataset.py — engine detections -> training data
labeler/               internal web tool: verify & correct engine pre-labels
tests/                 unit + synthetic-kinematics tests
```

## Two-pass design (why an hour-long session is tractable)

Detection runs coarse (`pose.detect_fps`, default 20fps) over motion-gated
windows only, seeking directly to each window instead of decoding the whole
file, batching frames per inference call. Anchors are then re-timed by a
**fine pass at native fps** in a short window around each kept delivery
(`refine.py`) — so a 60-minute session pays full-rate pose on a few minutes of
actual action, not on sixty minutes of standing around. Tracks are stitched
across brief occlusions so a run-up broken by a net pole still yields one
run-up onset.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m amp_autoclip session.mp4 --mode bowling --out clips/
```

**Always pass a single role** (`--mode bowling` or `--mode batting`) — the app
knows which kind of session was recorded, and a single-role run is markedly
more accurate: it removes the entire class of cross-role false positives (a
bowler's arm swing read as a bat swing, a batter's pull read as a release).
`both`/`auto` exist for exploration and are best-effort. `--device mps` on
Apple Silicon. `--no-cut` for detection-only.

Output: `clips/delivery_NN_<type>.mp4` + `manifest.json` with
`{start_timestamp, end_timestamp, type, confidence_score}` per delivery,
anchors, and `status: auto|review|discard`.

## Backend integration

```python
from amp_autoclip.service.api import router as autoclip_router
app.include_router(autoclip_router, prefix="/autoclip")
```

Worker (own process/dyno): `python -m amp_autoclip.service.worker`.
Production tables: `amp_autoclip/service/schema.sql` (Supabase). Job flow:
upload-complete → `POST /autoclip/jobs` → worker claims → progress on the row
→ `done | review_needed | failed(error kept)`. Review endpoints:
`approve / retrim / reject` — every decision doubles as a training label.

## Labeling

```bash
.venv/bin/uvicorn labeler.app:app --port 8017
# open http://localhost:8017/?dir=/path/to/videos
```

"Prelabel" runs the engine so humans verify/correct rather than label from
scratch. Keyboard: `,`/`.` frame-step, `[`/`]` boundaries, `B/F/R/S/T/I`
anchors, `X` reject, `L` swing/leave. Saving validates against the rubric and
writes `<video>.jsonl` ground truth.

## Train / evaluate

Bootstrap a dataset with no human labeling — the engine's structurally
complete, high-confidence detections become pseudo-labels, one sequence per
(video, track). Train **one model per role**, never a shared one:

```bash
python -m scripts.bootstrap_dataset --videos test_videos --role bowling --out dataset --device mps
python -m amp_autoclip.model.train --data dataset/ --role bowling --out models/bowling_v1.pt
python -m amp_autoclip.eval.evaluate --pred clips/manifest.json --truth labels/session01.jsonl
```

A pseudo-label model imitates the heuristic where the heuristic is confident;
its value is as the second opinion in an ensemble (agreement raises
confidence, disagreement routes to review). Review-corrected boundaries from
production replace pseudo-labels as they accumulate. No accuracy claim holds
until `eval.evaluate` reports it on held-out labeled footage.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 10 tests, no video files needed
```
