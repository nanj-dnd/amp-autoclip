"""Synthetic FeatureSeries builders for unit tests: idealized kinematics of
a bowling delivery and a batting shot, with controllable event times."""
from __future__ import annotations

import numpy as np

from amp_autoclip.features import FeatureSeries


def _base(n: int, fps: float) -> dict:
    t = np.arange(n) / fps
    z = np.zeros(n)
    return dict(
        t=t, fps=fps, kpts=np.zeros((n, 17, 2)), conf=np.ones((n, 17)),
        torso_len=100.0, pelvis_x=z.copy(), pelvis_y=z.copy(),
        pelvis_speed_x=z.copy(), pelvis_speed_y=z.copy(),
        torso_scale=z.copy() + 100.0, scale_rate=z.copy(),
        centroid_speed=z.copy(), wrist_speed_l=z.copy(), wrist_speed_r=z.copy(),
        wrist_above_shoulder_l=z.copy(), wrist_above_shoulder_r=z.copy(),
        ankle_y_l=z.copy() + 500, ankle_y_r=z.copy() + 500,
        ankle_vy_l=z.copy(), ankle_vy_r=z.copy(),
        ankle_still_std=z.copy() + 0.001,
        trunk_lean=z.copy() + 0.1, trunk_rot_vel=z.copy(),
        motion_energy=z.copy(),
    )


def bowling_series(fps: float = 30.0, runup_start: float = 2.0,
                   release: float = 6.0, settle: float = 8.0,
                   total: float = 12.0) -> FeatureSeries:
    """idle -> run-up (pelvis translating) -> release spike -> settle."""
    n = int(total * fps)
    d = _base(n, fps)
    t = d["t"]
    run = (t >= runup_start) & (t < release - 0.2)
    d["pelvis_speed_x"][run] = 2.5            # torso-lengths/s toward crease
    d["centroid_speed"][run] = 2.5
    d["motion_energy"][run] = 1.5             # stride cadence
    d["pelvis_x"] = np.cumsum(d["pelvis_speed_x"]) / fps * 100

    # ankle plants: BFC then FFC just before release
    for foot, at in (("ankle_y_r", release - 0.30), ("ankle_y_l", release - 0.15)):
        i = int(at * fps)
        d[foot][i - 1:i + 2] += 12.0          # local max = plant

    spike = (t >= release - 0.15) & (t <= release + 0.15)
    d["wrist_speed_r"][spike] = 3.0
    peak = int(release * fps)
    d["wrist_speed_r"][peak] = 4.0
    d["wrist_above_shoulder_r"][(t >= release - 0.3) & (t <= release + 0.2)] = 1.0

    decel = (t > release) & (t < settle)
    d["centroid_speed"][decel] = np.linspace(2.0, 0.0, int(np.sum(decel)))
    d["wrist_speed_r"][decel] = np.linspace(1.5, 0.0, int(np.sum(decel)))
    d["trunk_lean"][(t >= release - 0.5) & (t < settle)] = 0.6
    d["trunk_lean"][t >= settle] = 0.1
    return FeatureSeries(**d)


def batting_series(fps: float = 30.0, stance_start: float = 1.0,
                   trigger: float = 3.0, impact: float = 3.6,
                   settle: float = 5.2, total: float = 8.0) -> FeatureSeries:
    """idle (moving) -> stance stillness -> swing spike -> settle."""
    n = int(total * fps)
    d = _base(n, fps)
    t = d["t"]
    d["ankle_still_std"][:] = 0.05                       # fidgeting baseline
    still = (t >= stance_start) & (t < trigger)
    d["ankle_still_std"][still] = 0.003                  # planted
    swing = (t >= trigger) & (t <= impact + 0.2)
    d["wrist_speed_l"][swing] = 1.5
    peak = int(impact * fps)
    d["wrist_speed_l"][peak] = 3.5
    d["ankle_still_std"][swing] = 0.03
    d["trunk_rot_vel"][swing] = 4.0
    decay = (t > impact + 0.2) & (t < settle)
    nd = int(np.sum(decay))
    d["wrist_speed_l"][decay] = np.linspace(1.0, 0.05, nd)
    d["trunk_rot_vel"][decay] = np.linspace(3.0, 0.1, nd)
    return FeatureSeries(**d)
