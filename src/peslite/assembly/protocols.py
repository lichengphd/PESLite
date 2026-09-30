"""Hardware protocols retained while the components layer is migrated.

Controller-facing data and protocols have a single definition in
peslite.control and are re-exported here only for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray
import numpy as np

from ..control import ControlMeasurement, ControlOutput, Controller, Measurement
from ..control.modulation import PWMMethod, SignalLimiter

__all__ = [
    "Controller", "ControlOutput", "Measurement", "ControlMeasurement",
    "PWMMethod", "SignalLimiter", "Delay", "Modulator", "SwitchingSequence",
]


@runtime_checkable
class Delay(Protocol):
    """An N-sample delay line for duty ratios."""

    n_samples: int

    def __call__(self, d_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def reset(self, d_abc: NDArray[np.float64]) -> None: ...


@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern over one PWM publication interval."""

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]


@runtime_checkable
class Modulator(Protocol):
    """Convert duty ratios into a switching sequence spanning T_c seconds."""

    def __call__(self, t: float, T_c: float, d_abc: NDArray[np.float64],
                 theta: float | None = None,
                 omega: float | None = None) -> SwitchingSequence: ...
