# Auto-Clipping Engine — Design

The processing stage between "session video finished uploading" and "ffmpeg
frame extraction": one continuous net-session video in, one trimmed clip per
delivery out, each ready for the existing KPI pipeline unchanged. Post-upload,
server-side, asynchronous — no on-device inference, no streaming constraints.

**Status disclaimer, first because it matters most:** nothing in this document
is a validated accuracy claim. The v1 engine has been run end-to-end on exactly
one real net video (plus synthetic tests). Every numeric threshold is an
unvalidated starting point, collected in [`amp_autoclip/config.py`](amp_autoclip/config.py)
and flagged there. Accuracy numbers below that reference published benchmarks
are analogies about what architectures achieve on *their* datasets, not
predictions about ours. Nothing gates real coaching reports until the
evaluation harness ([`amp_autoclip/eval/evaluate.py`](amp_autoclip/eval/evaluate.py))
produces numbers on our own held-out labeled footage.

---

## 1. Architecture

### Two layers: spec-encoded detectors now, trained temporal model as refinement

```
session.mp4 (cloud storage)
  └─ job worker (service/worker.py)
       ├─ probe + motion gating          cheap frame-diff energy; finds active
       │                                 windows so pose skips dead net time
       ├─ audio transient channel        band-limited onsets (2-8 kHz), used
       │                                 only to corroborate + snap timing
       ├─ pose + tracking                YOLO11-pose per frame, centroid
       │                                 tracker for ID persistence
       ├─ subject role inference         which human is bowler / batter
       ├─ SEGMENTATION                   v1: anchor-first heuristic detectors
       │                                 (events_bowling.py, events_batting.py)
       │                                 v2: MS-TCN over pose sequences
       │                                 (model/) behind the same interface
       ├─ padding / plausibility /       segmenter.py
       │  confidence / pairing
       └─ ffmpeg cuts + manifest         clipper.py → existing frame extraction
```

The brief asks for a trained model, not a rule FSM — agreed, and that is where
this lands. But the cold-start reality is zero labeled clips today, so v1
encodes the boundary spec directly as pose-derived event detection. That v1 is
not throwaway: it is the **pre-labeling machine** for the labeling tool
(humans verify/correct instead of labeling from scratch), the **fallback path**
whenever the model's confidence is low, and the source of the anchor-first
decoding logic the model path reuses. Labeled data is a refinement pass, not a
prerequisite to shipping — exactly as the brief requires.

### Why MS-TCN for the temporal model

Chosen: **multi-stage temporal convolutional network (MS-TCN) over per-frame
pose feature sequences** (85 dims/frame: pelvis-centered torso-normalized
keypoints, their velocities, keypoint confidences), with two heads:

- **Phase head**: per-frame classification — bowling `idle / runup / delivery /
  followthrough`, batting `idle / stance / shot / followthrough`.
- **Event head**: per-frame Gaussian heatmap regression for the anchors —
  bowling `BFC / FFC / ball_release`, batting `stance_set / impact`. Anchors
  are near-instant, so they're ill-posed as phases; heatmap regression is how
  spatial keypoint detectors solve the same problem, applied to time.

Against the alternatives:

| Option | Verdict |
|---|---|
| Sliding-window classifier | Simple but poor boundary localization; redundant compute; no per-frame phase output for the KPI model to reuse. |
| **MS-TCN (chosen)** | ~1M params; receptive field of many seconds via dilated convs; the published temporal-action-segmentation results that made it standard were achieved on datasets of **28–50 videos** (GTEA, 50Salads) — the same order as our bootstrap dataset. CPU-realtime on keypoint features, so no forced GPU service. |
| ActionFormer / transformer detectors | Stronger ceiling, data-hungrier; right choice at thousands of labeled deliveries. The dataset/label format doesn't change, so migrating later is a model swap. |

Post-processing (model path, `model/infer.py`): argmax → median filter →
min-phase-duration merge with grammar constraints (a `followthrough` can never
precede its `delivery`) → segments; anchor times from event-heatmap peaks
constrained to the right phase; then the same padding/plausibility/confidence
layer v1 uses.

### Label space, precisely

Defined in code at [`amp_autoclip/model/labels.py`](amp_autoclip/model/labels.py),
which the labeling tool and dataset builder both import — labels physically
cannot be produced against a different space. Per-frame phases + per-frame
event heatmaps (σ = 2 frames), per role. `-100` = unlabeled frames (ignored in
the loss), so partially-labeled sessions are usable.

## 2. Boundary spec encoding (v1 detectors)

Every boundary is a detectable condition on named signals, per the spec.

**Bowling — anchor-first.** Release is the strongest signal, so it's found
first: wrist-speed peak gated on wrist-above-shoulder (≈ max arm extension at
the top of the arc). BFC/FFC: ankle-y ground plants in the window before
release (FFC = last plant; BFC = last plant of the *other* foot, ≥1 frame
earlier). Then extend outward:

- **Start (run-up initiation):** the hard lesson from the first real-footage
  run — pelvis *x*-velocity is the wrong signal in general. A front-on run-up
  barely moves in x; it shows up as torso-scale growth. So: run-up = sustained
  **locomotion** (whole-body motion energy = stride cadence proxy, with gap
  tolerance across the gather) whose **net displacement projects positively
  onto the bowling direction**, where the bowling direction is estimated in
  image space (x, y, scale) from the late run-up before BFC — *not* from the
  delivery stride, because trunk flexion at delivery corrupts the shoulder-hip
  scale proxy. Camera-angle-agnostic by construction. Negatives per spec:
  walking back projects negative (rejected), shuffling at the mark fails the
  net-displacement test, a fixed lookback from release is never used as the
  primary definition — when onset genuinely can't be found, the fallback is
  taken *and flagged* (`runup_onset_not_found`), dropping confidence toward
  review.
- **End (follow-through resolution):** earliest time ≥ release +
  `min_followthrough_s` where ALL hold for the stability window: bowling-arm
  wrist speed decayed, trunk back toward upright, and momentum resolved —
  near-stationary **or** clearly broken out (sustained movement *away* from
  the crease = walking back). Requiring all simultaneously implements
  "whichever resolves last governs". If nothing resolves by the cap, the clip
  is capped long (over-inclusive) and flagged.

**Batting — anchor-first.** Impact = wrist-speed peak, snapped to an audio
transient when one aligns within ±150 ms (sub-frame timing) and
confidence-boosted. Then:

- **Start (stance set):** latest sustained ankle-stillness window before the
  swing; the clip keeps up to `stance_max_include_s` of it (a batter standing
  set for ten seconds is idle), and pre-roll padding guarantees stable stance
  frames are present — the KPI model scores stance directly, so starting at
  the trigger is a failure case, tested for in `tests/test_events.py`.
- **End:** wrist speed and trunk-rotation velocity decayed, sustained — but
  never earlier than `min_post_impact_s` after impact, which protects
  fast-decaying defensive shots from truncation.
- **Leaves:** emitted, tagged `shot_type: "leave"` (recommendation accepted:
  stance/trigger mechanics are scoreable; the tag tells downstream not to
  expect impact mechanics). v1 honesty: leave detection requires the audio
  channel to catch the ball's pitch-bounce, so silent/noisy footage will miss
  leaves. Fixing this properly needs ball tracking or the trained model.
- **Aborts:** stance broken into sustained walking with no swing → discarded,
  not emitted.

**Neighbor separation:** delivery N's window must never contain N+1's action.
When raw windows collide, the cut lands at the motion-energy minimum between
the two releases (the walk-back lull), both clips are flagged, and padded
windows are hard-clamped to exclude a neighboring release/impact anchor.

## 3. Padding, tolerances, plausibility (all UNTUNED — proposals with reasoning)

Bias everything outward: a few over-included frames cost processing pennies;
an under-included boundary silently corrupts a KPI score and produces a wrong
coaching report.

| Quantity | Bowling | Batting | Why |
|---|---|---|---|
| Pre-roll | **750 ms** | **1000 ms** | Run-up onset is the highest-variance boundary; batting pre-roll is bigger because stance frames are directly scored. |
| Post-roll | **1000 ms** | **750 ms** | Follow-through dissipation is gradual; bowling gets more because deceleration strides vary most. |
| Anchor tolerance (BFC/FFC/release/impact) | ±66 ms (±2 frames @30fps) | same | These feed KPI scoring directly — tight. Halves at 60fps. |
| Outer-boundary tolerance | ±250 ms | same | Loose by design; the gating metric is **phase truncation**, not symmetric offset — a boundary 400 ms too *early* on a start is fine, 100 ms too *late* is a truncation. |
| Plausible clip duration | 3–30 s | 2–15 s | 3-step spinner floor → 20 m+ fast-bowler run-up ceiling. Outside range = detection failure → discard + flag, never shipped. |

## 4. Frame rate

Release and bat-ball impact are sub-100 ms events. At 30fps release occupies
1–2 frames and impact can fall between frames entirely.

- **Request 60fps** from the React Native camera (`react-native-vision-camera`
  format selection); fall back to 30fps when the device can't, and record the
  actual fps in session metadata. 120fps is not worth the default storage/
  upload cost for auto-clipping; consider it later as a premium capture mode
  for the KPI model's benefit, which is a separate decision.
- Precision floor by fps: boundary *clip* edges are padding-dominated and
  barely care. **Anchor events care a lot**: the video-derived error floor is
  ±½ frame ≈ ±17 ms @30fps, ±8 ms @60fps *best case*, realistically 1–2 frames
  (33–66 ms @30fps). The audio channel pins impact to sub-millisecond when a
  clean transient aligns, independent of fps — the cheapest precision upgrade
  available, already implemented.
- **Confidence down-weighting at low fps: yes** — implemented as
  `low_fps_penalty` below 50fps, so 30fps footage skews toward review.

## 5. Signals extracted (and the bat problem)

Bowling: pelvis/hip-centroid trajectory *including the torso-scale depth
proxy* (run-up, direction), ankle y/velocity (stride cadence, BFC/FFC),
shoulder-elbow-wrist chain speed with arm-raised gating (release), trunk lean
(dissipation; lateral/forward flexion split for the injury-flag KPIs comes
from the same keypoints downstream). Batting: ankle stillness (stance),
wrist height/speed (backlift → impact → follow-through), shoulder-line
rotation velocity (trunk rotation), head stability (available in the feature
matrix for the model).

**Bat tracking — the honest options:**

| Option | Cost | What you get |
|---|---|---|
| Infer from wrist kinematics (v1, chosen) | Free | Backlift/downswing/impact *timing* — wrists and bat are rigidly coupled in time. Loses bat angle and contact point, which auto-clipping doesn't need. |
| Audio transient (v1, chosen) | Free | Impact timing far better than any frame-based method; no spatial info; fails in multi-lane noise — corroboration only, never sole trigger. |
| Dedicated bat detector (YOLO fine-tune on bat boxes) | ~1–2k labeled boxes + an extra inference pass (~30–40% pipeline time) | Bat trajectory and angle. Needed for KPI-level bat metrics someday; **not needed for clipping v1**. Revisit only if eval shows impact-timing failures on silent footage. |

## 6. Person disambiguation & the both-in-one-video question

Track persistence: centroid tracker with gap timeout (deliberately simple; when
it fails, role scores drop and the clip lands in review rather than shipping
around the wrong human). Role inference per track: bowlers show directed
sustained translation + release signature; batters show low positional
variance + periodic wrist activity; a coach shows neither. Evidence feeds the
clip confidence.

- **The app should pass the session type** (`bowling` / `batting` / `both`)
  into the job — the product already knows it, since KPI reports are
  role-specific. `auto` exists and works but is best-effort: on the one real
  test video, a batter's pull shot produced a plausible bowling reading, which
  same-subject conflict resolution then had to referee (it kept the correct
  reading on 2 of 3 actions). Explicit mode removes the entire failure class.
  A tap-to-select subject at recording time is the right *final* fallback for
  crowded frames — worth adding to the RN capture screen, not required for v1.
- **One video containing both bowler and batter → two clips, not one.**
  Decision: emit a `bowling` clip and a `batting` clip from overlapping
  timestamps, linked by `pair_id`. Reasons: the downstream KPI pipeline's
  input contract is one subject per clip; the two roles have different
  boundary definitions (the bowler's clip starts seconds before the batter's);
  and a shared clip would need per-consumer re-trimming anyway. Same-subject
  double-readings are resolved (never paired) — a pair requires two distinct
  tracks.

## 7. Cold start → trained model: data & training plan

**Step 0 (now):** run the v1 engine over the existing pilot footage (100+
reports, 6 countries). Every session becomes pre-labeled candidates.

**Step 1 — labeling (weeks 1–2):** the labeling tool ([`labeler/`](labeler/))
serves footage + engine pre-labels; humans scrub (keyboard frame-stepping),
drag boundaries, confirm/correct anchors, reject false positives. Validation
enforces the rubric (a swing without an impact anchor won't save; leaves are
tagged; the rubric text lives in the UI). **Hardik and Rakshit label first**
— the first ~50 clips are really rubric calibration: both label the same
sessions, measure inter-labeler boundary disagreement, and that disagreement
becomes the empirical floor for the tolerance windows (if humans disagree by
±120 ms, ±66 ms model tolerance is fantasy). Outsource only after the rubric
survives contact, using their doubly-labeled set as the qualification gold
standard.

**Step 2 — v1 dataset target: ~300 verified deliveries** (150/role), which at
~10 deliveries/session is ~30 sessions — realistic from pilot footage in a
couple of focused days *because verifying pre-labels is 5–10× faster than
labeling cold*. Split **by session** (never by clip — clips from one session
share camera/lighting/players and leak), deterministic hash split 70/15/15,
stratified sanity-check across country/fps/lane type.

**Expected accuracy vs data volume — carefully worded:** at ~300 clips,
MS-TCN-class models on comparable-scale public benchmarks reach frame
accuracies in the 75–85% range with most boundary errors within a few hundred
ms. That is an *analogy from other datasets*, not a claim about ours; our
footage is harder (phone cameras, multi-person nets) in some ways and easier
(two rigid phase grammars, strong anchors) in others. What ~300 clips is
genuinely enough for: beating the heuristic on the held-out split and finding
out *where* it fails. **Production-grade auto-clipping** (auto-ship rate >80%
with ~zero truncations) realistically needs **1,500–2,500 labeled deliveries**
spanning angles, lighting, fps, body types, and multi-lane scenes — reachable
within weeks of launch from the production feedback loop, below.

**Step 3 — the flywheel:** every review-queue decision is a free label —
approve = positive, re-trim = corrected boundaries (gold), reject = hard
negative. The schema stores these (`deliveries.corrected_*`, reviewed_by).
Retrain monthly at first; each retrain must beat the incumbent on the frozen
test split *and* on truncation rate before the checkpoint swaps in.

## 8. Output spec & integration

Per session: ordered `manifest.json` —

```json
{"video": {...fps, duration...},
 "session_flags": ["high_discard_rate_review_session", ...],
 "deliveries": [{
   "start_timestamp": 9.10, "end_timestamp": 17.10,
   "type": "bowling", "confidence_score": 0.89,
   "status": "auto | review | discard",
   "raw_start_timestamp": 9.85, "raw_end_timestamp": 16.10,
   "anchors": [{"name": "ball_release", "t": 13.90, "confidence": 1.0,
                "source": "pose|audio|pose+audio"}, ...],
   "shot_type": "swing|leave", "pair_id": "pair_01",
   "clip_file": "delivery_02_bowling.mp4", "notes": [...]}]}
```

plus the cut clips (mp4/h264, re-encoded — stream copy snaps to keyframes and
can eat up to a GOP off the start, silently violating the padding guarantee).
The frame-extraction step's input contract is untouched: it receives one
trimmed mp4 per delivery exactly as it does today; the manifest's anchors are
*extra* information the KPI stage can adopt whenever useful. Raw timestamps
are stored so padding changes never require re-detection.

**Gating:** `confidence ≥ 0.75` → auto-ship to frame extraction; `0.40–0.75` →
cut the clip but hold it in the review queue (the app's manual-trim UI opens
**seeded with the engine's boundaries** — the user adjusts, never trims from
scratch); `< 0.40` → discard, and if too many discards, the whole session is
flagged rather than silently thinned. A coaching report is never generated
from an unreviewed low-confidence clip. Both thresholds are untuned and should
be set from the confidence-calibration table the eval harness produces.

## 9. Job orchestration

Kept inside the existing FastAPI repo — no new service until GPU inference
forces one (pose on keypoints is CPU-viable; if a GPU box happens later, only
the worker moves).

- **Trigger:** upload-complete handler (or Supabase Storage webhook) POSTs
  `/autoclip/jobs`; a `clip_jobs` row is the job.
- **Queue = Postgres** (`FOR UPDATE SKIP LOCKED` claim; schema in
  [`service/schema.sql`](amp_autoclip/service/schema.sql)). No Redis/Celery:
  the DB is already there, the volume is sessions-per-minute not
  thousands-per-second, and one fewer moving part on Railway. SQLite store
  implements the same interface for local dev.
- **Progress:** worker writes stage + fraction to the row; the app polls
  `GET /autoclip/jobs/{id}` or subscribes via Supabase Realtime — "your
  session is being analysed · pose 60%" instead of a spinner.
- **Failures:** retried with exponential backoff to `max_attempts`, then
  status `failed` with the traceback preserved in the row — surfaced, never
  swallowed. Review endpoints (`approve` / `retrim` / `reject`) close the
  loop and feed the label flywheel.

## 10. Evaluation before anything ships

`python -m amp_autoclip.eval.evaluate --pred manifest.json --truth labels.jsonl`

1. **Phase-truncation rate** — % of matched clips whose boundary cuts into a
   labeled phase. *The* gating metric; target ≈0 among auto-shipped clips.
2. **Detection precision/recall** at IoU ≥ 0.5 — missed or hallucinated
   deliveries are worse than sloppy boundaries.
3. **Boundary error distributions** (signed, ms) against the loose tolerance.
4. **Anchor timing errors** against the tight tolerance.
5. **Confidence calibration** — success rate per confidence bin; decides
   whether 0.75/0.40 gates mean anything, and sets them empirically.

Ship gate proposal: auto-clipping goes live (per role) only when, on a ≥50-
session held-out set: truncation rate <2% among would-be-auto clips, recall
>90%, precision >90%. Until then every session routes through review. These
gate values are themselves proposals to be argued with once real numbers
exist.

## 11. Edge cases

| Case | Handling |
|---|---|
| Multiple people in frame | Track persistence + role scoring; weak evidence → review, never silent wrong-subject clips. Tap-to-select at capture is the endgame fallback. |
| Aborted run-up / batter backs out | Bowling: no release anchor → nothing fires. Batting: stance→walking with no swing → detected and discarded. |
| Very short gaps between deliveries | Anchor-first detection with min-gap merge; neighbor separation cuts at the motion lull; padded windows clamped off neighboring anchors. Verified on real footage 4.6 s apart. |
| Poor lighting / low fps | Keypoint confidences feed the features; low-fps penalty skews to review; motion gating percentile adapts per video. |
| Portrait vs landscape | All thresholds torso-normalized; direction estimated from data, not axis assumptions. The one real test video was portrait. |
| No / unusable audio | Audio is corroboration-only, so the pipeline runs identically without it; leaves become undetectable (flagged limitation). |

## 12. Known v1 gaps (deliberate, documented)

- **Role classification in `auto` mode is weak** — a batter's pull can read as
  a release. Mitigated by same-subject conflict resolution; properly solved by
  explicit session type from the app (recommended) and the trained model.
- **Leaves need audio** in v1 (no ball tracking).
- **BFC/FFC at 30fps portrait-distance footage is marginal** — ankle keypoint
  noise; the tight anchor tolerance will likely need the 60fps capture path
  and/or the model's event head.
- **The centroid tracker will swap IDs when players cross** — accepted;
  confidence collapses to review. ByteTrack is a drop-in later if eval demands.
- **One real video so far — and it's an edited social-media reel** (jump cuts
  between shots), not a continuous session. Useful as a smoke test (the
  engine found all three swing events and routed the ambiguous ones to
  review); useless for judging boundary quality — and the fact that one
  clip still auto-shipped at 0.93 confidence on out-of-domain footage says
  the confidence model is optimistic and the auto threshold must be set
  from calibration data, not defaults. The next engineering task is not more
  code — it's running the engine across the pilot footage and labeling 30
  sessions.

## Most-sensitive thresholds — tune these first

1. `runup_loco_energy` / `runup_min_approach` — bowling clip start.
2. `resolution_*` stillness set — both clip ends (dominant source of
   `followthrough_capped` on the real sample).
3. `confidence_auto` / `confidence_review` — what ships vs queues; set from
   the calibration table, not by feel.
4. `min_event_gap_s` — delivery N vs N+1 merging.
5. `stance_still_std` / `stance_min_s` — stance misses caused 2 of 3 review
   flags on the real batting sample.
