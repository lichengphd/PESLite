"""Compatibility imports for controller building blocks.

The implementations live in peslite.control.blocks.
"""

from ..control.blocks import (
    HighPass1,
    HoldTimer,
    LowPass1,
    MovingWindow,
    clamp,
    peak_abs,
    smoothstep,
)

__all__ = [
    "clamp", "smoothstep", "peak_abs", "LowPass1", "HighPass1",
    "HoldTimer", "MovingWindow",
]
