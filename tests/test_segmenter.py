from __future__ import annotations

from amp_autoclip.segmenter import (_clamp_padding_to_neighbors,
                                    _pair_overlaps, _resolve_role_conflicts)
from amp_autoclip.types import Anchor, Delivery


def _d(start, end, typ, conf=0.8, track=0, anchors=None):
    return Delivery(start_t=start, end_t=end, type=typ, confidence=conf,
                    anchors=anchors or [], raw_start_t=start + 0.75,
                    raw_end_t=end - 0.75, subject_track_id=track)


def test_same_subject_role_conflict_keeps_stronger_reading():
    bowling = _d(2.0, 10.0, "bowling", conf=0.6, track=0,
                 anchors=[Anchor("ball_release", 6.0, 1.0)])
    batting = _d(2.5, 9.0, "batting", conf=0.9, track=0,
                 anchors=[Anchor("impact", 6.1, 1.0)])
    ds = [bowling, batting]
    _resolve_role_conflicts(ds)
    assert bowling.status == "discard"
    assert "role_conflict_same_subject" in bowling.notes
    assert batting.status != "discard"
    _pair_overlaps(ds)                     # discarded reading never pairs
    assert batting.pair_id is None


def test_different_subjects_pair_up():
    bowling = _d(2.0, 10.0, "bowling", track=0)
    batting = _d(4.0, 9.0, "batting", track=1)
    ds = [bowling, batting]
    _resolve_role_conflicts(ds)
    _pair_overlaps(ds)
    assert bowling.status != "discard" and batting.status != "discard"
    assert bowling.pair_id == batting.pair_id is not None


def test_padding_never_reaches_neighbor_release():
    a = _d(2.0, 12.0, "bowling",
           anchors=[Anchor("ball_release", 6.0, 1.0)])
    b = _d(9.0, 18.0, "bowling",
           anchors=[Anchor("ball_release", 11.5, 1.0)])
    _clamp_padding_to_neighbors([a, b])
    assert a.end_t <= 11.5 - 0.3 + 1e-9    # a's padded tail excludes b's release
    assert b.start_t >= 6.0 + 0.3 - 1e-9   # b's padded head excludes a's release


def test_fragment_and_weak_role_cleanup():
    from amp_autoclip.config import EngineConfig
    from amp_autoclip.segmenter import _recheck_plausibility, _suppress_weak_role
    cfg = EngineConfig()
    # complete bowling detections (have FFC) + stance-less batting noise
    b1 = _d(2.0, 8.0, "bowling", conf=0.7,
            anchors=[Anchor("ball_release", 5.0, 1.0),
                     Anchor("front_foot_contact", 4.8, 0.5)])
    b2 = _d(20.0, 26.0, "bowling", conf=0.7,
            anchors=[Anchor("ball_release", 23.0, 1.0),
                     Anchor("front_foot_contact", 22.8, 0.5)])
    noise = _d(10.0, 12.5, "batting", conf=0.52,
               anchors=[Anchor("impact", 11.0, 0.6)])
    frag = _d(30.0, 30.9, "bowling", conf=0.6,
              anchors=[Anchor("ball_release", 30.5, 1.0)])
    ds = [b1, b2, noise, frag]
    _recheck_plausibility(ds, cfg)
    assert frag.status == "discard"          # 0.9s << bowling minimum
    assert b1.status != "discard"
    _suppress_weak_role(ds)
    assert noise.status == "discard"         # no stance anywhere -> noise
    assert "role_suppressed_no_defining_anchor" in noise.notes
    assert b1.status != "discard" and b2.status != "discard"
