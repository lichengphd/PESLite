"""The ADC of a converter: instantaneous samples of its measurement ports, or their means over an
averaging window ``(X(t) - X(t - length)) / length`` of trapezoidal integrals (SI values), taken at
each publication of the PWM and, oversampling, between them."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..control.blocks import complex2abc
from ..control.controller import Measurement

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
    """Sampler of the measurement ports, ``samples`` times per publication ``period`` (s) of the PWM.

    The channels in ``channels`` (``"v"``, ``"i"``, ``"dc"``, each with its zero value) are averaged
    over a window of ``length`` s ending at the publication, the others are instantaneous.
    Named states per averaged channel ``c``: ``x_c`` (integral) and ``x_c_open`` (integral at window start).
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
        self.integral = dict(channels)  # the running integrals
        self.opened = dict(channels)  # integrals at window start
        self._last = dict(channels)  # previous sample
        self.period, self.samples = period, samples
        self.sample_period = period / samples
        self.peeks: list[Measurement] = []  # the samples taken since the last publication
        self.n_samp = 1  # sample 0 of a period is its publication
        self.latest: Optional[Measurement] = None
        # the window of the next publication is open (a full-period window opens at the publication itself)
        self.window_open = self.length is None or self._full

    @property
    def _full(self) -> bool:
        return self.length >= self.period * (1.0 - 1e-12)

    @property
    def averaging(self) -> bool:
        return bool(self.channels)

    # ------------------------------------------------------------ the window
    def seed(self) -> None:
        """Take the current values as the start of the integrals (at the start of a run)."""
        values = self.ports.read()
        self._last = {c: values[c] for c in self.channels}

    def accumulate(self, dt: float) -> None:
        """Add the trapezoidal integral over the interval of length ``dt`` (s) ending now."""
        if dt <= 0.0 or not self.channels:
            return
        values = self.ports.read()
        half = 0.5 * dt
        for c in self.channels:
            now = values[c]
            self.integral[c] += half * (self._last[c] + now)
            self._last[c] = now

    def open(self) -> None:
        """Start the window of the next publication at the current instant."""
        self.opened = dict(self.integral)
        self.window_open = True

    def t_window(self, t_next: float) -> float:
        """The opening of the window of the publication at ``t_next`` (s); ``inf`` when it is open."""
        return math.inf if self.window_open else t_next - self.length

    # ------------------------------------------------------------ the samples
    def t_sample(self, start: float) -> float:
        """The next sample between two publications of the period that started at ``start``; ``inf`` when none."""
        return (start + self.n_samp * self.sample_period) if self.n_samp < self.samples else math.inf

    def peek(self, t: float) -> None:
        """Take a sample between two publications (oversampling)."""
        self.latest = sample = self.measure(t)
        self.peeks.append(sample)
        self.n_samp += 1

    def sample(self, t: float) -> Measurement:
        """Take the sample of the publication at ``t``, with those of the period before it (oversampling),
        and rearm the averaging window."""
        self.latest = meas = self.measure(t)
        if self.length is not None:
            self.window_open = self._full
            if self.window_open:
                self.open()
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
        length, opened = self.length, self.opened
        mean = {c: (self.integral[c] - opened[c]) / length for c in self.channels}
        return Measurement(t=t,
                           u_g=mean.get("v", u_raw), i_c=mean.get("i", i_raw),
                           u_dc=mean.get("dc", dc_raw), i_abc=complex2abc(i_raw),
                           u_g_raw=u_raw, i_c_raw=i_raw, u_dc_raw=dc_raw)

    def phase_currents(self) -> np.ndarray:
        """Return the instantaneous phase currents read from the state vector."""
        return complex2abc(self.ports.i_c_state())

    # ------------------------------------------------------------ its states
    def get_state(self) -> dict[str, Any]:
        s: dict[str, Any] = {f"x_{c}": v for c, v in self.integral.items()}
        s.update({f"x_{c}_open": v for c, v in self.opened.items()})
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        for channel, zero in self._zero.items():
            cast = complex if isinstance(zero, complex) else float
            for key, into in ((f"x_{channel}", self.integral), (f"x_{channel}_open", self.opened)):
                if key in values:
                    into[channel] = cast(values[key])
