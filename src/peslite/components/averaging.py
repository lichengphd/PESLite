"""Ideal averaged bridge actuation with the PWM-equivalent output delay.

The bridge itself remains the lossless controlled voltage source in :mod:`.converter`.  This
component only delays its continuous duty command.  It has no carrier, switching sequence or PWM
registers.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex

__all__ = ["AveragingActuator"]


_EPS = 1e-10


class AveragingActuator:
    """Delay continuous duty commands before applying them to an ideal averaged bridge.

    A command sampled at ``t`` is represented at the centre of the first equivalent PWM update
    interval whose load is not earlier than ``t + computation``.  Thus the default single-update
    timing with nonzero computation has a delay of 1.5 carrier periods; zero computation has half
    a carrier period.  No PWM peripheral is constructed or advanced.

    The current output and a fixed amount of recent input history are states.  The history is
    automatically filled with the initial output, so a fresh run naturally holds that output until
    the first delayed command arrives.
    """

    def __init__(self, period: float, load_period: float, offset: float, computation: float) -> None:
        self.period = float(period)
        self.load_period = float(load_period)
        self.offset = float(offset)
        self.computation = float(computation)
        if not math.isfinite(self.period) or self.period <= 0.0:
            raise ValueError(f"the control period must be finite and positive, got {period}")
        if not math.isfinite(self.load_period) or self.load_period <= 0.0:
            raise ValueError(f"the equivalent update period must be finite and positive, got {load_period}")
        if not math.isfinite(self.offset):
            raise ValueError(f"the timer offset must be finite, got {offset}")
        if not math.isfinite(self.computation) or self.computation < 0.0:
            raise ValueError(f"the computation time must be finite and nonnegative, got {computation}")
        delays = [self.delay(self.interrupt(k)) for k in range(4)]
        self.history_length = max(1, int(math.ceil(max(delays) / self.period - 1e-12)))
        zero = np.zeros(3)
        self.active = zero.copy()
        self.history = [zero.copy() for _ in range(self.history_length)]  # newest first
        self.k = 0
        self.t_interrupt = self.interrupt(0)
        self._queue: list[tuple[float, int, NDArray[np.float64]]] = []
        self.t_apply = math.inf

    def interrupt(self, k: int) -> float:
        return self.offset + k * self.period

    @property
    def t_next(self) -> float:
        return self.t_interrupt

    def after(self, t: float) -> int:
        """Index of the first control interrupt strictly after ``t``."""
        return int(math.floor((t - self.offset + _EPS) / self.period)) + 1

    def _load_at_or_after(self, t: float) -> float:
        j = int(math.ceil((t - self.offset - _EPS) / self.load_period))
        return self.offset + j * self.load_period

    def apply_time(self, t: float) -> float:
        """Time at which the command sampled at ``t`` reaches the controlled source."""
        load = self._load_at_or_after(t + self.computation)
        return load + 0.5 * self.load_period

    def delay(self, t: float) -> float:
        return self.apply_time(t) - t

    def describe(self) -> str:
        delays = [self.delay(self.interrupt(k)) for k in range(8)]
        lo, hi = min(delays), max(delays)
        if abs(hi - lo) <= _EPS:
            return f"ideal averaging, delay {lo * 1e6:g} us"
        return f"ideal averaging, delay {lo * 1e6:g}..{hi * 1e6:g} us"

    def reset(self, d_abc: NDArray[np.float64]) -> None:
        """Fill the output and delay history with one initial duty command."""
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        self.active = d.copy()
        self.history = [d.copy() for _ in range(self.history_length)]
        self._queue.clear()
        self.t_apply = math.inf

    def start(self, t: float, continued: bool) -> complex:
        """Resume the interrupt grid and rebuild pending delayed commands from saved history."""
        self.k = self.after(t)
        self.t_interrupt = self.interrupt(self.k)
        self._queue = []
        if continued:
            for age, duty in enumerate(self.history):
                index = self.k - 1 - age
                due = self.apply_time(self.interrupt(index))
                if due > t + _EPS:
                    self._queue.append((due, index, duty.copy()))
            self._queue.sort(key=lambda item: item[:2])
        self.t_apply = self._queue[0][0] if self._queue else math.inf
        return abc2complex(self.active)

    def write(self, t: float, d_abc: NDArray[np.float64]) -> None:
        """Accept one controller output and schedule its delayed application."""
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        self.history = [d.copy(), *self.history[:-1]]
        self._queue.append((self.apply_time(t), self.k, d.copy()))
        self.k += 1
        self.t_interrupt = self.interrupt(self.k)
        self.t_apply = self._queue[0][0]

    def apply(self, t: float) -> complex:
        """Apply all commands due at ``t`` and return the controlled-source duty vector."""
        while self._queue and self._queue[0][0] <= t + _EPS:
            self.active = self._queue.pop(0)[2]
        self.t_apply = self._queue[0][0] if self._queue else math.inf
        return abc2complex(self.active)

    def get_state(self) -> dict[str, Any]:
        out = {f"d_{phase}": float(self.active[k]) for k, phase in enumerate("abc")}
        for age, duty in enumerate(self.history):
            out.update({f"history.{age}.d_{phase}": float(duty[k])
                        for k, phase in enumerate("abc")})
        return out

    def set_state(self, values: Mapping[str, Any]) -> None:
        known = set(self.get_state())
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"averaging actuator: no state(s) {sorted(unknown)}; known: {sorted(known)}")
        for key, value in values.items():
            parts = key.split(".")
            target = self.active if len(parts) == 1 else self.history[int(parts[1])]
            target["abc".index(parts[-1][-1])] = float(value)
