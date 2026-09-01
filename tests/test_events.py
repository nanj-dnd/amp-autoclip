from __future__ import annotations

from amp_autoclip.config import EngineConfig
from amp_autoclip.events_batting import detect_batting_events
from amp_autoclip.events_bowling import detect_bowling_events

from .synthetic import batting_series, bowling_series


def test_bowling_single_delivery_boundaries():
    cfg = EngineConfig()
    fs = bowling_series(runup_start=2.0, release=6.0, settle=8.0)
    events = detect_bowling_events(fs, cfg)
    assert len(events) == 1
    ev = events[0]
    assert abs(ev.release_t - 6.0) < 0.25
    # start at run-up onset, not at a fixed lookback from release
    assert ev.start_source == "runup"
    assert abs(ev.raw_start_t - 2.0) < 0.5
    # end only after full dissipation, never at release
    assert ev.raw_end_t > 6.5
    assert ev.end_source == "resolution"
    # anchors present: FFC after BFC, both before release
    assert ev.bfc_t is not None and ev.ffc_t is not None
    assert ev.bfc_t < ev.ffc_t < ev.release_t


def test_bowling_no_fire_on_idle():
    cfg = EngineConfig()
    fs = bowling_series()
    fs.wrist_speed_r[:] = 0.1          # nobody bowls
    fs.wrist_speed_l[:] = 0.1
    assert detect_bowling_events(fs, cfg) == []


def test_batting_includes_stance_and_min_dwell():
    cfg = EngineConfig()
    fs = batting_series(stance_start=1.0, trigger=3.0, impact=3.6)
    events = detect_batting_events(fs, cfg, transients=[])
    swings = [e for e in events if e.shot_type == "swing"]
    assert len(swings) == 1
    ev = swings[0]
    assert abs(ev.impact_t - 3.6) < 0.25
    # clip start captures the set stance, not the trigger movement
    assert ev.stance_t is not None
    assert ev.raw_start_t <= 3.0 - cfg.batting.stance_min_s + 0.2
    # defensive-shot protection: end never sooner than min dwell post impact
    assert ev.raw_end_t >= ev.impact_t + cfg.batting.min_post_impact_s - 1e-6


def test_batting_audio_snaps_and_boosts():
    from amp_autoclip.audio import AudioTransient
    cfg = EngineConfig()
    fs = batting_series(impact=3.6)
    fs.wrist_speed_l *= 0.3          # weak swing so confidence isn't saturated
    base = detect_batting_events(fs, cfg, transients=[])[0]
    snapped = detect_batting_events(
        fs, cfg, transients=[AudioTransient(t=3.55, strength=2.0)])[0]
    assert snapped.impact_t == 3.55          # audio pins timing
    assert snapped.impact_source == "pose+audio"
    assert snapped.impact_conf > base.impact_conf
