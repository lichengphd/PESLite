"""The ADC of a converter: instantaneous samples of its measurement ports, or their means over an
averaging window ``(X(t) - X(t - length)) / length`` of trapezoidal integrals (SI values), taken at
each control interrupt and, when oversampling, between interrupts."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..control.blocks import complex2abc
from ..control.controller import Measurement
from .pwm import TIME_EPS

__all__ = ["MeasurementPorts", "ADC"]


@dataclass(frozen=True)
class MeasurementPorts:
    """Zero-argument callables that read one converter's quantities from the model (SI).

    ``u_g`` terminal voltage, ``i_c`` converter current, ``u_dc`` dc voltage;
    ``i_c_state`` reads the current from the state vector; ``i_dc`` is recorded only.
    """

    u_g: Callable[[], complex]
    i_c: Callable[[], complex]
    i_c_state: Callable[[], complex]
    u_dc: Callable[[], float]
    i_dc: Optional[Callable[[], float]] = None

    def read(self) -> dict[str, complex | float]:
        """The instantaneous values of the ADC channels ``v``, ``i`` and ``dc``."""
        return {"v": self.u_g(), "i": self.i_c(), "dc": self.u_dc()}


class ADC:
    """Sampler of the measurement ports, ``samples`` times per control ``period`` (s).

    The channels in ``channels`` (``"v"``, ``"i"``, ``"dc"``, each with its zero value) are averaged
    over a window of ``length`` s ending at the interrupt; the others are instantaneous.
    Named state per averaged channel ``c``: ``x_c`` (the integral accumulated in the open window).
    """

    def __init__(self, ports: MeasurementPorts, period: float, samples: int = 1, length: Optional[float] = None,
                 channels: Mapping[str, complex | float] | None = None) -> None:
        self.ports = ports
        channels = dict(channels or {})
        if channels and (length is None or length <= 0.0):
            raise ValueError(f"an averaging window must be longer than zero, got {length}")
        self.length = float(length) if channels else None  # None: instantaneous sampling
        self.channels = tuple(channels)
        self._zero = channels
        self.accumulated = dict(channels)  # integrals since the current window opened
        self._last = dict(channels)  # previous sample
        self.period, self.samples = period, samples
        self.sample_period = period / samples
        self.peeks: list[Measurement] = []  # the samples taken since the last publication
        self.n_samp = 1  # sample 0 of a period is its control interrupt
        self.latest: Optional[Measurement] = None
        # the next interrupt's window is open (a full-period window opens at the interrupt itself)
        self.window_open = self.length is None or self._full

    @property
    def _full(self) -> bool:
        return self.length >= self.period * (1.0 - 1e-12)

    @property
    def averaging(self) -> bool:
        return bool(self.channels)

    def start(self, t: float, last: float, next_: float, held: bool) -> None:
        """Resume sampling at ``t`` between the interrupts ``last`` and ``next_``.

        ADC integrals are named states, while the position of the sampler and whether the next
        averaging window is open follow from ``t`` and its timer grid.  Samples already taken
        between interrupts are not states; validation therefore permits a continued oversampled
        run only at an interrupt.  For a new run, ``held`` makes a window which began before
        ``t`` count the value at ``t`` over that preceding part.
        """
        self.n_samp = max(1, int(math.floor((t - last + TIME_EPS) / self.sample_period)) + 1)
        length = self.length
        self.window_open = length is None or self._full or t >= next_ - length - TIME_EPS
        if length is not None and held and self.window_open:
            self.held_before(max(0.0, t - (next_ - length)))
        self.peeks = []
        if self.n_samp > 1:  # a new run between interrupts: earlier samples use its start values
            first = self.measure(t)
            self.peeks = [dataclasses.replace(first, t=last + i * self.sample_period)
                          for i in range(1, self.n_samp)]

    # ------------------------------------------------------------ the window
    def seed(self) -> None:
        """Take the current values as the start of the integrals (at the start of a run)."""
        values = self.ports.read()
        self._last = {c: values[c] for c in self.channels}

    def held_before(self, duration: float) -> None:
        """Count the current values as held for ``duration`` seconds before the run starts."""
        for channel in self.channels:
            self.accumulated[channel] += self._last[channel] * duration

    def accumulate(self, dt: float) -> None:
        """Add the trapezoidal integral over the interval of length ``dt`` (s) ending now."""
        if dt <= 0.0 or not self.channels:
            return
        values = self.ports.read()
        half = 0.5 * dt
        for c in self.channels:
            now = values[c]
            if self.window_open:
                self.accumulated[c] += half * (self._last[c] + now)
            self._last[c] = now

    def open(self) -> None:
        """Start the window ending at the next control interrupt."""
        self.accumulated = dict(self._zero)
        self.window_open = True

    def t_window(self, t_next: float) -> float:
        """The opening of the window ending at interrupt ``t_next`` (s); ``inf`` when open."""
        return math.inf if self.window_open else t_next - self.length

    # ------------------------------------------------------------ the samples
    def t_sample(self, start: float) -> float:
        """The next sample between two interrupts in the period starting at ``start``; ``inf`` if none."""
        return (start + self.n_samp * self.sample_period) if self.n_samp < self.samples else math.inf

    def peek(self, t: float) -> None:
        """Take an oversample between two control interrupts."""
        self.latest = sample = self.measure(t)
        self.peeks.append(sample)
        self.n_samp += 1

    def sample(self, t: float) -> Measurement:
        """Take the sample at the control interrupt ``t``, with the preceding oversamples,
        and rearm the averaging window."""
        self.latest = meas = self.measure(t)
        if self.length is not None:
            self.window_open = self._full
            if self.window_open:
                self.open()
            else:
                self.accumulated = dict(self._zero)
        if self.samples > 1:  # oversampling: this publication's samples, oldest first
            meas = dataclasses.replace(meas, samples=tuple(self.peeks) + (meas,))
        self.peeks, self.n_samp = [], 1
        return meas

    def measure(self, t: float) -> Measurement:
        """Return the :class:`Measurement` at ``t``; model outputs must be up to date at ``t``.

        ``u_g``, ``i_c``, ``u_dc`` are window means for averaged channels;
        ``u_g_raw``, ``i_c_raw``, ``u_dc_raw`` and ``i_abc`` are instantaneous.
        """
        ports = self.ports
        u_raw, i_raw, dc_raw = ports.u_g(), ports.i_c(), ports.u_dc()
        mean = {c: self.accumulated[c] / self.length for c in self.channels}
        return Measurement(t=t,
                           u_g=mean.get("v", u_raw), i_c=mean.get("i", i_raw),
                           u_dc=mean.get("dc", dc_raw), i_abc=complex2abc(i_raw),
                           u_g_raw=u_raw, i_c_raw=i_raw, u_dc_raw=dc_raw)

    def phase_currents(self) -> np.ndarray:
        """Return the instantaneous phase currents read from the state vector."""
        return complex2abc(self.ports.i_c_state())

    # ------------------------------------------------------------ its states
    def get_state(self) -> dict[str, Any]:
        return {f"x_{c}": value for c, value in self.accumulated.items()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for channel, zero in self._zero.items():
            cast = complex if isinstance(zero, complex) else float
            key = f"x_{channel}"
            if key in values:
                self.accumulated[channel] = cast(values[key])
