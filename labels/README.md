# Ground-truth labels

Human-produced annotations. These are the only copy of work done by hand —
the footage they refer to is deliberately not in this repo (size + rights),
but these must be versioned.

## Schemas

**`*-clip-segments-v1.csv`** (`amp-clip-segments-v1`) — purpose-built for
auto-clipping. One row per segment, `segment_type` of `delivery` or
`background`, with `start_ms` / `end_ms` / `event_ms`. `coverage_complete`
means the whole video was swept, so the `background` rows are true negatives
rather than merely unlabeled time. **This is the schema to use for clipping
work.**

**`*-draft-training-labels-v2.csv`** (`amp-training-labels-long-v2`) — a KPI
scoring export, not a clipping export: one row per KPI per delivery, so a
14-delivery video is ~250 rows. Deliveries are recoverable by deduping on
`delivery_id` and keeping `training_row_status == ready_scored_label`, which
is what `scripts/ingest_csv_labels.py` does.

## What's here

| file | video | deliveries | notes |
|---|---|---|---|
| `part-2--clip-segments-v1.csv` | `part 2 .mp4` | 5 (+6 background) | net session, side-on, 60fps |
| `part-3-clip-segments-v1.csv` | `part 3.mp4` | 14 (+15 background) | net session, side-on, 30fps |
| `cover-drive-draft-training-labels-v2.csv` | `cover drive.mov` | 1 | front-on |
| `loft-draft-training-labels-v2.csv` | `loft.mov` | 0 ready | all rows excluded |
| `vidssave-*-cover-drives-*.csv` | highlight reel | 14 | see caveat |
| `vidssave-*-lofted-drives-*.csv` | highlight reel | 16 | see caveat |

**Highlight-reel caveat:** every delivery in the two `vidssave` files is a
different player, ground and camera, spliced with hard cuts. Track continuity
does not hold across them, so each delivery becomes its own ~40-frame track:
60% of the sequences but 9% of the frames, carrying no idle context. Training
with them included measured worse than without (macro recall 0.876 vs 0.908 on
a held-out net session). Prefer the net sessions.

## Known gaps in the schema

- **No trigger / run-in onset.** Only start, contact and end are marked, so
  any internal phase split has to be invented. This is why the ingest defaults
  to the boundary-only (`idle`/`delivery`) label space.
- **No `stance_set` anchor.** The event loss is unmasked BCE, so an absent
  channel trains as "never fires" rather than "unknown"; the ingest derives it
  from the delivery start.
- **Annotator start precedes motion onset by 0.26–2.45s** across these 19
  deliveries. That range is directly usable for calibrating
  `motion_gate_percentile` / `motion_gate_pad_s`, which currently start
  windows too late for the labeled start to be learnable.

## Use

    python -m scripts.ingest_csv_labels --csv labels/'*clip-segments-v1.csv' \
        --videos /path/to/videos --role batting --out dataset --windows labels

`--dry-run` parses and reports without needing the videos or any dependency.
