"""The PWM peripheral of a converter: timer, duty registers, carrier and modulators.

A modulator is called at every load of the compare registers for the interval until the next load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex

__all__ = ["PWM", "TIME_EPS", "SwitchingSequence", "Modulator", "CarrierComparison",
           "SynchronousCarrier", "ZOH", "TimeStepAveragedCarrier", "carrier", "carrier_position", "duty_fraction",
           "make_modulator"]


@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern from one compare-register load to the next.

    ``dt``: interval lengths (s) summing to the interval between compare-register loads.
    ``q_abc[i]``: phase states in interval ``i`` (0/1 when switching, fractional when averaged).
    """

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]


@runtime_checkable
class Modulator(Protocol):
    """Convert loaded duty ratios into a SwitchingSequence spanning ``T`` seconds.

    Optional ``theta`` (rad) and ``omega`` (rad/s) set synchronous carrier timing; asynchronous modulators ignore them.
    """

    def __call__(self, t: float, T: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence: ...


TIME_EPS = 1e-10


class PWM:
    """Carrier timer, PWM duty registers and switching schedule.

    Interrupts occur every ``period`` from ``offset``. Compare registers load every
    ``load_period``. An interrupt writes duty ratios and the PWM enable to the shadow registers;
    a load copies them to the active registers only after ``computation`` seconds. The enable is
    carried through exactly the same register path as the duty ratios.
    """

    _NAMES = ("d_a", "d_b", "d_c", "on")

    def __init__(self, period: float, load_period: float, offset: float, computation: float,
                 carrier_period: float, modulator: Modulator) -> None:
        if not math.isfinite(computation) or computation < 0.0:
            raise ValueError(f"the computation time must be finite and >= 0, got {computation}")
        self.period, self.load_period, self.offset = float(period), float(load_period), float(offset)
        self.computation, self.carrier_period = float(computation), float(carrier_period)
        self.modulator = modulator
        self.k = self.j = 0
        self.t_interrupt, self.t_load = self.interrupt(0), self.load_time(0)
        self.active, self.shadow = np.zeros(4), np.zeros(4)
        self.sync: tuple[float | None, float | None] = (None, None)
        self.t_sync = 0.0
        self.schedule: list[tuple[float, complex]] = []
        self.next_switch = math.inf
        self.load_end = math.inf

    def interrupt(self, k: int) -> float:
        return self.offset + k * self.period

    def load_time(self, j: int) -> float:
        return self.offset + j * self.load_period

    @property
    def t_next(self) -> float:
        """Next control interrupt (temporary name retained while the run loop is migrated)."""
        return self.t_interrupt

    def after(self, t: float, period: Optional[float] = None) -> int:
        """Index of the first timer point after ``t``; a point at ``t`` is not after it."""
        T = self.period if period is None else period
        return int(math.floor((t - self.offset + TIME_EPS) / T)) + 1

    def count(self, t_a: float, t_b: float) -> int:
        """Number of control interrupts in ``[t_a, t_b]``, including both ends."""
        first = math.ceil((t_a - self.offset - TIME_EPS) / self.period)
        return max(0, self.after(t_b) - first)

    def describe(self) -> str:
        double = self.load_period < 0.75 * self.carrier_period
        return (f"control {1e-3 / self.period:g} kHz, computation {self.computation * 1e6:g} us, "
                f"{'double' if double else 'single'} update")

    @staticmethod
    def _word(d_abc: NDArray[np.float64], on: bool) -> NDArray[np.float64]:
        word = np.empty(4, dtype=float)
        word[:3] = np.asarray(d_abc, dtype=float).ravel()
        word[3] = 1.0 if on else 0.0
        return word

    def reset(self, d_abc: NDArray[np.float64], on: bool = False) -> None:
        """Put duty ratios and enable in both register banks; the default is blocked."""
        word = self._word(d_abc, on)
        self.active, self.shadow = word.copy(), word.copy()

    def tick(self, t: float, theta: float | None = None, omega: float | None = None) -> None:
        """Advance the control-interrupt timer and its synchronous-carrier reference."""
        self.sync, self.t_sync = (theta, omega), t
        self.k += 1
        self.t_interrupt = self.interrupt(self.k)

    def write_shadow(self, d_abc: NDArray[np.float64], on: bool = True) -> None:
        """Write duty ratios and enable directly to an immediately eligible shadow register."""
        self.shadow = self._word(d_abc, on)

    def write(self, t: float, d_abc: NDArray[np.float64], on: bool = True,
              theta: float | None = None,
              omega: float | None = None) -> None:
        """Advance an interrupt and write its computed duty ratios and enable to shadow."""
        self.tick(t, theta, omega)
        self.shadow = self._word(d_abc, on)

    @property
    def duty(self) -> NDArray[np.float64]:
        """Duty ratios in the active compare registers."""
        return self.active[:3].copy()

    @property
    def on(self) -> bool:
        """PWM enable in the active compare registers."""
        return bool(self.active[3])

    def start(self, t: float, sync: tuple[float | None, float | None], continued: bool) -> complex:
        """Resume the timer and switching sequence at ``t`` after named states were loaded."""
        self.k, self.j = self.after(t), self.after(t, self.load_period)
        self.t_interrupt, self.t_load = self.interrupt(self.k), self.load_time(self.j)
        last = self.interrupt(self.k - 1)
        self.sync, self.t_sync = sync, min(last, t)
        q = self._switching(self.load_time(self.j - 1), self.active[:3], t)
        if not continued:
            self.t_sync = t
        return q

    def load(self, t: float) -> complex:
        """Load shadow into active registers and schedule switching until the next load."""
        completed = self.interrupt(self.k - 1) + self.computation
        if completed <= t + TIME_EPS:
            self.active = self.shadow.copy()
        q = self._switching(t, self.active[:3], t)
        self.j += 1
        self.t_load = self.load_time(self.j)
        return q

    def _switching(self, t_l: float, d: NDArray[np.float64], t_now: float) -> complex:
        theta, omega = self.sync
        if theta is not None and omega is not None and t_l != self.t_sync:
            theta = theta + omega * (t_l - self.t_sync)
        seq = self.modulator(t_l, self.load_period, d, theta, omega)
        dt = np.asarray(seq.dt, dtype=float).tolist()
        states = np.asarray(seq.q_abc, dtype=float).tolist()
        self.load_end = t_l + self.load_period
        schedule: list[tuple[float, complex]] = []
        q_now, t_i = abc2complex(states[0]), t_l
        for i in range(len(dt) - 1):
            t_i += dt[i]
            q = abc2complex(states[i + 1])
            if t_i <= t_now + TIME_EPS:
                q_now = q
            else:
                schedule.append((t_i, q))
        self.schedule = schedule
        self.next_switch = schedule[0][0] if schedule else math.inf
        return q_now

    def edge(self, t: float) -> bool:
        return abs(self.next_switch - t) < TIME_EPS or abs(self.load_end - t) < TIME_EPS

    def switches(self, t: float) -> list[complex]:
        """Return switching states due at ``t`` in order."""
        due = []
        while self.schedule and abs(self.schedule[0][0] - t) < TIME_EPS:
            due.append(self.schedule.pop(0)[1])
        self.next_switch = self.schedule[0][0] if self.schedule else math.inf
        return due

    def get_state(self) -> dict[str, Any]:
        return {f"{head}{name}": (bool(word[k]) if name == "on" else float(word[k]))
                for head, word in (("", self.active), ("shadow.", self.shadow))
                for k, name in enumerate(self._NAMES)}

    def set_state(self, values: Mapping[str, Any]) -> None:
        names = set(self.get_state())
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"PWM registers: no state(s) {sorted(unknown)}; known: {sorted(names)}")
        for key, value in values.items():
            head, _, name = key.rpartition(".")
            k = self._NAMES.index(name)
            target = {"": self.active, "shadow": self.shadow}[head]
            target[k] = (1.0 if value else 0.0) if name == "on" else float(value)


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
