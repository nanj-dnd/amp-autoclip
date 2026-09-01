"""Amp auto-clipping engine: session video -> per-delivery clips.

Sits between "session upload finished" and "ffmpeg frame extraction".
See DESIGN.md for the full architecture and the boundary spec encoding.
"""
from .config import DEFAULT, EngineConfig
from .pipeline import run_session
from .types import Anchor, Delivery, SessionResult, VideoInfo

__all__ = ["DEFAULT", "EngineConfig", "run_session", "Anchor", "Delivery",
           "SessionResult", "VideoInfo"]
__version__ = "0.1.0"
