"""CLI for local runs and debugging.

    python -m amp_autoclip session.mp4 --mode bowling --out clips/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .clipper import write_outputs
from .config import DEFAULT
from .pipeline import run_session


def main() -> None:
    p = argparse.ArgumentParser(prog="amp_autoclip",
                                description="Auto-clip a cricket session video "
                                            "into per-delivery clips.")
    p.add_argument("video")
    p.add_argument("--mode", choices=["bowling", "batting", "both", "auto"],
                   default="auto")
    p.add_argument("--out", default="clips")
    p.add_argument("--device", default=None,
                   help="pose inference device: cpu | mps | cuda")
    p.add_argument("--no-cut", action="store_true",
                   help="detection + manifest only, skip ffmpeg cuts")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")
    cfg = DEFAULT
    if args.device:
        cfg.pose.device = args.device

    result = run_session(args.video, mode=args.mode, cfg=cfg)
    manifest = write_outputs(result, args.out, cfg, cut_files=not args.no_cut)
    print(json.dumps(manifest, indent=2))
    n_auto = sum(1 for d in result.deliveries if d.status == "auto")
    n_review = sum(1 for d in result.deliveries if d.status == "review")
    n_discard = sum(1 for d in result.deliveries if d.status == "discard")
    print("\n%d deliveries: %d auto, %d flagged for review, %d discarded"
          % (len(result.deliveries), n_auto, n_review, n_discard), file=sys.stderr)
    for flag in result.session_flags:
        print("SESSION FLAG: %s" % flag, file=sys.stderr)


if __name__ == "__main__":
    main()
