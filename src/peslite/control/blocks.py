"""Building blocks of the controller: space-vector transforms, discrete-time filters and timers, helpers.

Space vectors are peak-value, ``u = (2/3)(u_a + a u_b + a^2 u_c)`` with ``a = exp(j 2 pi/3)``, the
zero sequence dropped; ``u_dq = u_ab * exp(-j theta)``. Filters accept real or complex signals;
filters and timers expose named states.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

__all__ = ["A120", "abc2complex", "complex2abc", "phases", "clamp", "smoothstep", "peak_abs",
           "PI", "LowPass1", "HighPass1", "HoldTimer", "MovingWindow"]


# ------------------------------------------------------------------ space vectors

A120 = cmath.exp(2j * math.pi / 3.0)  # 120-degree rotation
_A240 = A120 * A120
_A120_CONJ = A120.conjugate()


def abc2complex(u_abc) -> complex:
    """Convert phase quantities to a peak-scaled space vector (zero sequence dropped)."""
    u_a = float(u_abc[0])
    u_b = float(u_abc[1])
    u_c = float(u_abc[2])
    return (2.0 / 3.0) * (u_a + A120 * u_b + _A240 * u_c)


def phases(u: complex) -> tuple[float, float, float]:
    """Convert a space vector to three phase quantities (floats)."""
    return u.real, (u * _A120_CONJ).real, (u * A120).real


def complex2abc(u: complex) -> NDArray[np.float64]:
    """Convert a space vector to a phase-quantity array."""
    return np.array(phases(u))


# ------------------------------------------------------------------ scalar helpers

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def smoothstep(x: float) -> float:
    """Smooth ramp from 0 to 1 on [0, 1], constant outside."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def peak_abs(values) -> float:
    """Return the largest absolute value in ``values`` (NaN if any value is NaN)."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    it = iter(values)
    peak = abs(next(it))
    for v in it:
        a = abs(v)
        if a > peak or a != a:
            peak = a
    return float(peak)


# ------------------------------------------------------------------ control law

class PI:
    """PI law. Output constraints are owned by the controller; tracking stays here."""

    def __init__(self, kp: float, ki: float, period: float | None, *,
                 antiwindup: bool = True, initial=0.0) -> None:
        self.kp, self.ki, self.T = float(kp), float(ki), period
        self.antiwindup_enabled = bool(antiwindup)
        self.integral = initial

    def sample(self, r, fb, ff=0.0):
        """Advance one sampled step and return the unconstrained intended output."""
        error = r - fb
        self.integral += self.T * error
        return ff + self.kp * error + self.ki * self.integral

    def flow(self, r, fb, ff=0.0):
        """Return the unconstrained output and integral derivative."""
        error = r - fb
        return ff + self.kp * error + self.ki * self.integral, error

    def antiwindup(self, intended, output) -> None:
        """Apply sampled tracking anti-windup after a controller constraint acts."""
        if self.antiwindup_enabled:
            self.integral += self.T * (output - intended) / self.kp

    def antiwindup_flow(self, intended, output):
        """Return the continuous-time tracking term for the integral derivative."""
        return ((output - intended) / self.kp) if self.antiwindup_enabled else 0.0

    def reset(self) -> None:
        self.integral = self.integral * 0


# ------------------------------------------------------------------ filters and timers

class LowPass1:
    """First-order low-pass filter: ``y += (1 - exp(-2*pi*bw_hz*T))*(x - y)``.

    ``bw_hz`` in Hz (``<= 0`` passes the input through), ``T`` in s. ``init_on_first``
    copies the first sample to the output; ``y0`` is the initial output.
    Named state: ``y``, NaN while not yet seeded.
    """

    def __init__(self, bw_hz: float, T: float | None, y0=0.0, init_on_first: bool = True) -> None:
        self.alpha = (0.0 if T is None else
                      1.0 - math.exp(-2.0 * math.pi * bw_hz * T) if bw_hz > 0.0 else 1.0)
        self.y = y0
        self._y0 = y0
        self._init_on_first = init_on_first
        self._first = init_on_first

    def update(self, x):
        if self._first:
            self.y = x
            self._first = False
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y

    def get_state(self) -> dict[str, Any]:
        if self._first:
            return {"": complex(math.nan, math.nan) if isinstance(self._y0, complex) else math.nan}
        return {"": self.y}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" not in values:
            return
        y = values[""]
        if (y.real != y.real) or (isinstance(y, complex) and y.imag != y.imag):  # nan: not seeded
            self.y, self._first = self._y0, self._init_on_first
        else:
            self.y, self._first = y, False


class HighPass1(LowPass1):
    """First-order high-pass filter ``x - LowPass1(x)`` with corner ``bw_hz`` (Hz).

    Named state: the low-passed signal.
    """

    def update(self, x):
        return x - super().update(x)


class HoldTimer:
    """Return how long (s) a condition has held continuously; resets when it clears."""

    def __init__(self, T: float) -> None:
        self.T = T
        self.held = 0.0

    def update(self, condition: bool) -> float:
        self.held = self.held + self.T if condition else 0.0
        return self.held

    def get_state(self) -> dict[str, Any]:
        return {"": self.held}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" in values:
            self.held = float(values[""])


class MovingWindow:
    """Ring buffer whose ``push`` returns the sample ``n`` steps ago (``None`` until full)."""

    def __init__(self, n: int) -> None:
        self.n = max(1, n)
        self.buf = [0.0] * self.n
        self.idx = 0
        self.full = False

    def push(self, x: float):
        old = self.buf[self.idx] if self.full else None
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.n
        if self.idx == 0:
            self.full = True
        return old
