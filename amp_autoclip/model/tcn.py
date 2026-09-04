"""MS-TCN-lite: multi-stage temporal convolutional network over per-frame
pose features.

Why this architecture (vs sliding-window classifiers, ActionFormer, or a
transformer): MS-TCN's dilated temporal convolutions give a receptive field
of many seconds with ~1M parameters, it holds published state-of-the-art-
class results on temporal action segmentation benchmarks whose *entire*
training sets are a few dozen videos (50Salads: 50, GTEA: 28) — which is
exactly Amp's cold-start data regime — and it runs realtime-or-faster on
CPU over keypoint features, so no GPU service is forced. ActionFormer-class
detectors become attractive once labeled data reaches the thousands.

Input:  (B, F, N) feature sequences (F=85 pose features, N frames)
Output: phase logits (B, P, N) per stage + event heatmaps (B, E, N)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.dropout(self.conv_1x1(out))
        return x + out


class Stage(nn.Module):
    def __init__(self, in_dim: int, channels: int, n_layers: int, out_dim: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2 ** i, channels) for i in range(n_layers)])
        self.conv_out = nn.Conv1d(channels, out_dim, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.conv_out(x)


class MSTCN(nn.Module):
    def __init__(self, in_features: int = 85, n_phases: int = 4,
                 n_events: int = 3, channels: int = 64, n_layers: int = 10,
                 n_stages: int = 2):
        super().__init__()
        self.stage1 = Stage(in_features, channels, n_layers, n_phases)
        self.refine = nn.ModuleList(
            [Stage(n_phases, channels, n_layers, n_phases)
             for _ in range(n_stages - 1)])
        self.event_head = Stage(in_features, channels, n_layers, n_events)

    def forward(self, x):
        """x: (B, F, N). Returns (phase_logits_per_stage, event_logits)."""
        out = self.stage1(x)
        outs = [out]
        for stage in self.refine:
            out = stage(F.softmax(out, dim=1))
            outs.append(out)
        events = self.event_head(x)
        return outs, events


def segmentation_loss(stage_outputs, phase_targets, event_logits,
                      event_targets, smooth_weight: float = 0.15,
                      class_weights=None):
    """MS-TCN loss: CE per stage + truncated MSE smoothing on log-probs
    (suppresses over-segmentation flicker) + BCE on event heatmaps.

    class_weights: optional (n_phases,) tensor. The phases that define a
    clip's boundaries (shot/followthrough, or delivery/followthrough) are a
    few percent of frames, so unweighted CE is minimized by never predicting
    them at all — see train.py --class-weights."""
    loss = torch.zeros((), device=phase_targets.device)
    for out in stage_outputs:
        loss = loss + F.cross_entropy(out, phase_targets, ignore_index=-100,
                                      weight=class_weights)
        logp = F.log_softmax(out, dim=1)
        diff = torch.clamp((logp[:, :, 1:] - logp[:, :, :-1].detach()) ** 2,
                           max=16.0)
        loss = loss + smooth_weight * diff.mean()
    loss = loss + F.binary_cross_entropy_with_logits(event_logits, event_targets)
    return loss
