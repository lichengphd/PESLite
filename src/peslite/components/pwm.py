"""The PWM peripheral of a converter: its publications, computation delay, carrier and modulators.

A modulator is called as ``mod(t, T_s, d_abc, theta=None, omega=None) -> SwitchingSequence``
with times in s, duty ratios in [0, 1], ``theta`` in rad and ``omega`` in rad/s; it turns the duty
ratios of one PWM publication interval into piecewise-constant switching states of the bridge.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex

__all__ = ["PWM", "SwitchingSequence", "Modulator", "Delay", "ComputationDelay", "CarrierComparison",
           "SynchronousCarrier", "ZOH", "TimeStepAveragedCarrier", "carrier", "carrier_position", "duty_fraction",
           "make_modulator"]


@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern over one PWM publication interval.

    ``dt``: interval lengths (s) summing to the publication interval.
    ``q_abc[i]``: phase states in interval ``i`` (0/1 when switching, fractional when averaged).
    """

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]


@runtime_checkable
class Modulator(Protocol):
    """Convert duty ratios into a SwitchingSequence spanning ``T_c`` (s), the PWM publication interval.

    Optional ``theta`` (rad) and ``omega`` (rad/s) set synchronous carrier timing; asynchronous modulators ignore them.
    """

    def __call__(self, t: float, T_c: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence: ...


@runtime_checkable
class Delay(Protocol):
    """An N-sample delay line for the duty ratios (computation delay)."""

    n_samples: int

    def __call__(self, d_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def reset(self, d_abc: NDArray[np.float64]) -> None: ...


class ComputationDelay:
    """Delay duty ratios by ``n_samples`` periods: ``d_applied[k] = d_ref[k - n_samples]``.

    :meth:`reset` fills the pipeline with initial duty ratios; ``n_samples = 0`` passes through.
    Named states: ``"<j>.d_a"``, ``"<j>.d_b"``, ``"<j>.d_c"``; ``j = 0`` is applied next.
    """

    def __init__(self, n_samples: int = 0) -> None:
        if n_samples < 0:
            raise ValueError("n_samples must be >= 0")
        self.n_samples = int(n_samples)
        self._buf: deque[NDArray[np.float64]] = deque()

    def reset(self, d_abc: NDArray[np.float64]) -> None:
        self._buf = deque(np.array(d_abc, dtype=float, copy=True) for _ in range(self.n_samples))

    def __call__(self, d_abc: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.asarray(d_abc, dtype=float)
        if self.n_samples == 0:
            return d
        if len(self._buf) != self.n_samples:
            self.reset(d)
        self._buf.append(d)
        return self._buf.popleft()

    def get_state(self) -> dict[str, Any]:
        return {f"{j}.d_{ph}": float(d[k]) for j, d in enumerate(self._buf) for k, ph in enumerate("abc")}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            j_str, _, name = key.partition(".")
            if not j_str.isdigit() or name not in ("d_a", "d_b", "d_c") or int(j_str) >= len(self._buf):
                raise KeyError(f"computation delay: no state {key!r} (pipeline length {len(self._buf)})")
            self._buf[int(j_str)]["abc".index(name[-1])] = float(value)


class PWM:
    """The PWM peripheral of a unit: at each publication, every ``period`` (s), the controller's duty
    ratios pass the computation delay ``delay``, and ``modulator`` turns the duty ratios in force into
    the switching instants of the period.

    Named states: ``d_a``, ``d_b``, ``d_c``, the duty ratios in force.
    """

    def __init__(self, period: float, modulator: Modulator, delay: Delay) -> None:
        self.period, self.modulator, self.delay = period, modulator, delay
        self.k = 0  # the next publication, at k * period
        self.d = np.zeros(3)
        self.sync: tuple[float | None, float | None] = (None, None)  # controller angle, frequency
        self.start, self.end = 0.0, math.inf  # the current period
        self.schedule: list[tuple[float, complex]] = []  # its switching instants still to come, and states
        self.next_switch = math.inf

    @property
    def t_next(self) -> float:
        """The time of the next publication (s)."""
        return self.k * self.period

    def publish(self, d_abc: NDArray[np.float64], theta: float | None = None, omega: float | None = None) -> None:
        """Take the controller's duty ratios (and angle and frequency) through the computation delay."""
        self.d = self.delay(d_abc)
        self.sync = (theta, omega)

    def modulate(self, t: float) -> complex:
        """Start the period at ``t`` with the duty ratios in force: schedule its switching instants and
        return the switching state (space vector) at ``t``."""
        seq = self.modulator(t, self.period, self.d, *self.sync)
        q = [abc2complex(row) for row in np.asarray(seq.q_abc, dtype=float).tolist()]
        times, t_i = [], t
        for dt in np.asarray(seq.dt, dtype=float).tolist()[:-1]:
            t_i = t_i + dt
            times.append(t_i)
        self.start, self.end = t, t + self.period
        self.schedule = list(zip(times, q[1:]))
        self.next_switch = self.schedule[0][0] if self.schedule else math.inf
        self.k += 1
        return q[0]

    def switches(self, t: float, eps: float) -> list[complex]:
        """Return the switching states due at ``t`` (within ``eps`` s), in order."""
        due = []
        while self.schedule and abs(self.schedule[0][0] - t) < eps:
            due.append(self.schedule.pop(0)[1])
        self.next_switch = self.schedule[0][0] if self.schedule else math.inf
        return due

    def get_state(self) -> dict[str, Any]:
        return {f"d_{ph}": float(self.d[k]) for k, ph in enumerate("abc")}

    def set_state(self, values: Mapping[str, Any]) -> None:
        d = np.array(self.d, dtype=float)
        for k, ph in enumerate("abc"):
            if f"d_{ph}" in values:
                d[k] = values[f"d_{ph}"]
        self.d = d


# ------------------------------------------------------------------ the triangular carrier

_SNAP = 1e-9  # vertex snapping tolerance (carrier periods)


def carrier_position(t: float, f_sw: float, phase: float) -> float:
    """Return the carrier position in [0, 1) periods; ``f_sw`` in Hz, ``phase`` in periods."""
    u = math.fmod(t * f_sw + phase, 1.0)
    if u < 0.0:
        u += 1.0
    if u > 1.0 - _SNAP or u < _SNAP:
        return 0.0
    if abs(u - 0.5) < _SNAP:
        return 0.5
    return u


def _value(u: float) -> float:
    """The carrier value in [-1, 1] at position ``u`` (periods): rising from -1 at 0, falling from 1 at 1/2."""
    return -1.0 + 4.0 * u if u < 0.5 else 3.0 - 4.0 * u


def carrier(t: float, f_sw: float, phase: float = 0.0) -> float:
    """Return the triangular carrier value in [-1, 1]; ``phase`` in carrier periods.

    With ``phase = 0`` the carrier is at -1 and rising at ``t = 0``.
    """
    return _value(carrier_position(t, f_sw, phase))


def _pieces(t0: float, span: float, f_sw: float, phase: float, tol: float,
            limit: int) -> Iterator[tuple[float, float, bool, float, float]]:
    """The monotonic pieces of the carrier over ``[t0, t0 + span]``: ``(offset, length, rising,
    value at the start, slope)``, at most ``limit`` of them."""
    covered = 0.0
    for _ in range(limit):
        if covered >= span - tol:
            return
        u = carrier_position(t0 + covered, f_sw, phase)
        rising = u < 0.5
        seg = min(((0.5 if rising else 1.0) - u) / f_sw, span - covered)
        yield covered, seg, rising, _value(u), 4.0 * f_sw if rising else -4.0 * f_sw
        covered += seg


def duty_fraction(m: float, t0: float, dt: float, f_sw: float, phase: float = 0.0) -> float:
    """Return the fraction of ``[t0, t0 + dt]`` during which ``m`` is at or above the carrier."""
    duty_time = 0.0
    for _start, seg, rising, c0, slope in _pieces(t0, dt, f_sw, phase, 1e-18, 8):
        crossing = min(seg, max(0.0, (m - c0) / slope))
        duty_time += crossing if rising else seg - crossing
    return duty_time / dt


# ------------------------------------------------------------------ modulators

_EPS = 1e-13


class CarrierComparison:
    """Triangular-carrier comparison with exact switching instants.

    ``f_sw`` in Hz, ``phase`` in carrier periods, ``min_interval`` in s.
    """

    def __init__(self, f_sw: float, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.f_sw = float(f_sw)
        self.phase = float(phase)
        self.min_interval = min_interval

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0).tolist()
        # initial switching state at t
        u_t = carrier_position(t, self.f_sw, self.phase)
        c_t = _value(u_t)
        if u_t < 0.5:
            q = [1.0 if m[k] > c_t else 0.0 for k in range(3)]
        else:
            q = [1.0 if m[k] >= c_t else 0.0 for k in range(3)]
        events: list[tuple[float, int, float]] = []
        for start, seg, rising, c0, slope in _pieces(t, T_s, self.f_sw, self.phase, _EPS, 64):
            for k in range(3):
                x = (m[k] - c0) / slope
                if 0.0 < x < seg:
                    # rising carrier crosses m from below -> switch off; falling -> on
                    events.append((start + x, k, 0.0 if rising else 1.0))
        events.sort(key=lambda e: e[0])
        dts: list[float] = []
        states: list[list[float]] = []
        t_prev = 0.0
        for time, k, new_state in events:
            if time - t_prev > self.min_interval:
                dts.append(time - t_prev)
                states.append(list(q))
                t_prev = time
            q[k] = new_state
        if T_s - t_prev > self.min_interval or not dts:
            dts.append(T_s - t_prev)
            states.append(list(q))
        else:  # absorb a vanishing last interval into the previous one
            dts[-1] += T_s - t_prev
        return SwitchingSequence(np.asarray(dts), np.asarray(states))


class SynchronousCarrier:
    """Carrier comparison with the carrier locked to the controller angle.

    Carrier position is ``pulse_ratio * theta / (2 pi) + phase`` (periods);
    requires ``theta`` (rad) and ``omega`` (rad/s).
    """

    def __init__(self, pulse_ratio: int, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.pulse_ratio = int(pulse_ratio)
        self.phase = float(phase)
        self.min_interval = min_interval

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        if theta is None or omega is None:
            raise ValueError("synchronous modulation needs the controller's angle: this controller "
                             "reports no theta/omega in its ControlOutput")
        f = self.pulse_ratio * omega / (2.0 * np.pi)  # carrier frequency (Hz)
        if f <= 0.0:
            raise ValueError(f"synchronous modulation needs omega > 0, got {omega}")
        position = self.pulse_ratio * theta / (2.0 * np.pi) + self.phase  # carrier position at t (periods)
        return CarrierComparison(f, position - t * f, self.min_interval)(t, T_s, d_abc)


class ZOH:
    """PWM-period averaging: hold the duty ratios as the switching state for the period."""

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        return SwitchingSequence(np.array([T_s]), d.reshape(1, 3))


class TimeStepAveragedCarrier:
    """Time-step averaging: per solver step, the on-fraction of the carrier comparison."""

    def __init__(self, f_sw: float, dt: float, phase: float = 0.0) -> None:
        self.f_sw, self.dt, self.phase = float(f_sw), float(dt), float(phase)

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        n = max(1, int(round(T_s / self.dt)))
        h = T_s / n
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0)
        states = np.empty((n, 3))
        for i in range(n):
            t0 = t + i * h
            for k in range(3):
                states[i, k] = duty_fraction(m[k], t0, h, self.f_sw, self.phase)
        return SwitchingSequence(np.full(n, h), states)


def make_modulator(pwm: Any, f0: float, averaging: Any, sim: Any) -> Modulator:
    """Return the modulator selected by one unit's ``averaging`` and ``pwm.sync``.

    A disabled averaging section uses carrier comparison. PWM-period averaging holds the duty
    ratios continuously; time-step averaging uses the carrier's on-fraction per solver step.
    """
    if not averaging.enable:
        if pwm.sync == "synchronous":
            return SynchronousCarrier(round(pwm.f_sw / f0), pwm.carrier_phase)
        return CarrierComparison(pwm.f_sw, pwm.carrier_phase)
    if averaging.over == "time_step":
        return TimeStepAveragedCarrier(pwm.f_sw, sim.solver.dt, pwm.carrier_phase)
    return ZOH()
