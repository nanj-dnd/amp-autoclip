"""Training loop for the MS-TCN segmentation model.

    python -m amp_autoclip.model.train --data dataset/ --role bowling \
        --out models/bowling_v1.pt

Small-data regime by design: full-sequence batches of 1, heavy dropout in
the TCN, horizontal-flip augmentation (mirrors left/right arm bowlers and
handedness), and early stopping on val frame accuracy. Reports honest val
metrics — no claims beyond what the held-out split shows.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from . import labels as L
from .dataset import Sample, load_samples, session_split
from .tcn import MSTCN, segmentation_loss


def _flip(s: Sample) -> Sample:
    """Horizontal flip: negate x-relative coords and x-velocities, swap
    left/right keypoint channels. Feature layout: rel(17*2), vel(17*2), conf(17)."""
    swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    X = s.X.copy()
    for block in (0, 34):                      # rel block, vel block
        blk = X[:, block:block + 34].reshape(-1, 17, 2)
        blk = blk[:, swap, :]
        blk[:, :, 0] *= -1.0
        X[:, block:block + 34] = blk.reshape(-1, 34)
    X[:, 68:] = X[:, 68:][:, swap]
    return Sample(s.video_id + "_flip", s.role, X, s.phase_targets,
                  s.event_targets)


def _to_tensors(s: Sample, device):
    X = torch.from_numpy(s.X.T[None]).float().to(device)          # (1, F, N)
    y = torch.from_numpy(s.phase_targets[None]).long().to(device)  # (1, N)
    e = torch.from_numpy(s.event_targets[None]).float().to(device)  # (1, E, N)
    return X, y, e


@torch.no_grad()
def macro_recall(model, samples, device, n_phases: int) -> float:
    """Mean per-class recall. Frame accuracy rewards collapsing onto the
    majority phase (idle+stance are ~90% of frames); this does not, so it is
    the honest early-stopping signal once the CE is class-weighted."""
    if not samples:
        return float("nan")
    hit = np.zeros(n_phases)
    tot = np.zeros(n_phases)
    model.eval()
    for s in samples:
        X, y, _ = _to_tensors(s, device)
        outs, _ = model(X)
        pred = outs[-1].argmax(dim=1)
        mask = y != -100
        yv = y[mask].cpu().numpy()
        pv = pred[mask].cpu().numpy()
        for k in range(n_phases):
            sel = yv == k
            tot[k] += sel.sum()
            hit[k] += (pv[sel] == k).sum()
    seen = tot > 0
    return float((hit[seen] / tot[seen]).mean()) if seen.any() else float("nan")


@torch.no_grad()
def frame_accuracy(model, samples, device) -> float:
    if not samples:
        return float("nan")
    correct = total = 0
    model.eval()
    for s in samples:
        X, y, _ = _to_tensors(s, device)
        outs, _ = model(X)
        pred = outs[-1].argmax(dim=1)
        mask = y != -100
        correct += int((pred[mask] == y[mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--role", choices=["bowling", "batting"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--class-weights", choices=["none", "inverse", "sqrt"],
                   default="none",
                   help="reweight CE by inverse phase frequency (computed on "
                        "the train split only). 'sqrt' is the milder variant.")
    p.add_argument("--phase-space", choices=["full", "boundary"],
                   default="full",
                   help="'boundary' collapses the internal phases to "
                        "idle/delivery — clip start+end only")
    p.add_argument("--select", choices=["frame_acc", "macro_recall"],
                   default="frame_acc",
                   help="metric the checkpoint is selected on")
    p.add_argument("--holdout", default=None,
                   help="video_id prefix forced into the val split (all "
                        "tracks of one session hold out together — the split "
                        "must never leak a session across sides)")
    args = p.parse_args()

    samples = load_samples(args.data, args.role, args.phase_space)
    if args.holdout:
        val = [s for s in samples if s.video_id.startswith(args.holdout)]
        train = [s for s in samples if not s.video_id.startswith(args.holdout)]
        test = []
    else:
        train, val, test = session_split(samples)
    print("sessions: %d train / %d val / %d test" % (len(train), len(val), len(test)))
    if not train:
        raise SystemExit("no training samples — label some sessions first "
                         "(see labeler/)")
    train = train + [_flip(s) for s in train]

    n_phases = len(L.phases_for(args.role, args.phase_space))
    class_w = None
    if args.class_weights != "none":
        counts = np.zeros(n_phases, dtype=np.float64)
        for s in train:
            y = s.phase_targets[s.phase_targets != -100]
            counts += np.bincount(y, minlength=n_phases)
        freq = counts / max(counts.sum(), 1)
        with np.errstate(divide="ignore"):
            w = np.where(counts > 0, 1.0 / np.maximum(freq, 1e-12), 0.0)
        if args.class_weights == "sqrt":
            w = np.sqrt(w)
        w = w / w[w > 0].mean()          # keep the loss on the same scale
        class_w = torch.tensor(w, dtype=torch.float32, device=args.device)
        print("class weights: %s" % {n: round(float(v), 2) for n, v
                                     in zip(L.phases_for(args.role, args.phase_space), w)})

    model = MSTCN(n_phases=n_phases,
                  n_events=len(L.events_for(args.role))).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train)
        ep_loss = 0.0
        for s in train:
            X, y, e = _to_tensors(s, args.device)
            outs, events = model(X)
            loss = segmentation_loss(outs, y, events, e,
                                     class_weights=class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
        acc = frame_accuracy(model, val, args.device)
        mrec = macro_recall(model, val, args.device, n_phases)
        print("epoch %3d  loss %.3f  val_frame_acc %.3f  val_macro_recall %.3f" %
              (epoch, ep_loss / len(train), acc, mrec))
        score = mrec if args.select == "macro_recall" else acc
        if score > best_score:
            best_score, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "role": args.role,
                "phase_space": args.phase_space,
                "selected_on": args.select, "val_score": best_score,
                "val_frame_acc": frame_accuracy(model, val, args.device),
                "val_macro_recall": macro_recall(model, val, args.device,
                                                 n_phases)}, args.out)
    print("saved %s (val frame acc %.3f, val macro recall %.3f, test acc %.3f "
          "— evaluate boundaries with amp_autoclip.eval before trusting)" %
          (args.out, frame_accuracy(model, val, args.device),
           macro_recall(model, val, args.device, n_phases),
           frame_accuracy(model, test, args.device)))


if __name__ == "__main__":
    main()
