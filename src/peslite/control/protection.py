"""Protection of a converter unit: over-current, ac-voltage, frequency and dc-voltage trips and a ROCOF alarm.

Trips are enabled while ``armed(t)``; timed criteria must hold for ``hold`` (s).
Named states: ``tripped`` and the hold timers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..solver.model import gather, scatter
from .blocks import HoldTimer, MovingWindow

__all__ = ["Protection", "TripEvent", "ProtectionStats"]


@dataclass
class TripEvent:
    t: float
    cause: str
    detail: str = ""


@dataclass
class ProtectionStats:
    i_alarm_first_t: float = -1.0
    i_alarm_steps: int = 0
    vac_uv_first_t: float = -1.0
    vac_ov_first_t: float = -1.0
    freq_band_first_t: float = -1.0
    vdc_band_first_t: float = -1.0
    rocof_first_t: float = -1.0
    rocof_max_hz_s: float = 0.0
    max_current_pu: float = 0.0
    vac_min_pu: float = -1.0
    vac_max_pu: float = -1.0
    alarms: list[str] = field(default_factory=list)


class Protection:
    """Trip and alarm decisions of one unit from its ``protection`` parameters; ``T``: its sampling period (s)."""

    def __init__(self, cfg: Any, T: float, armed: Callable[[float], bool]) -> None:
        self.cfg, self.T, self.armed = cfg, T, armed
        self.trip: TripEvent | None = None
        self.stats = ProtectionStats()
        self._timers = {"hold_uv": HoldTimer(T), "hold_ov": HoldTimer(T), "hold_freq": HoldTimer(T),
                        "hold_vdc": HoldTimer(T)}
        self._window(cfg.rocof.window)

    def _window(self, window: float) -> None:
        """Start an empty ROCOF window with the requested duration."""
        n = max(1, int(window / self.T + 0.5))
        self._rocof = MovingWindow(n)
        self._rocof_window = n * self.T

    def retune(self, cfg: Any) -> None:
        """Apply new protection settings; a changed ROCOF window starts empty."""
        if cfg.rocof.window != self.cfg.rocof.window:
            self._window(cfg.rocof.window)
        self.cfg = cfg

    @property
    def tripped(self) -> bool:
        return self.trip is not None

    def get_state(self) -> dict[str, Any]:
        return {"tripped": self.tripped, **gather(self._timers)}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "tripped" in values:
            if values["tripped"] and self.trip is None:
                self.trip = TripEvent(math.nan, "restored", "latched trip loaded with the initial values")
            elif not values["tripped"]:
                self.trip = None
        scatter(self._timers, values)

    def _raise(self, name: str) -> None:
        if name not in self.stats.alarms:
            self.stats.alarms.append(name)

    def _do_trip(self, t: float, cause: str, detail: str) -> None:
        if self.trip is None:
            self.trip = TripEvent(t, cause, detail)
            self._raise("TRIP_" + cause.upper())

    # ---------------------------------------------------------------- fast
    def check_current(self, t: float, peak_pu: float) -> bool:
        """Check the phase-current peak ``peak_pu`` (pu); return ``True`` if this call trips."""
        self.stats.max_current_pu = max(self.stats.max_current_pu, peak_pu)
        overcurrent = self.cfg.overcurrent
        if overcurrent.enable and peak_pu > overcurrent.limit_pu:
            if self.stats.i_alarm_steps == 0:
                self.stats.i_alarm_first_t = t
                self._raise("OVERCURRENT")
            self.stats.i_alarm_steps += 1
            if not self.tripped and self.armed(t):
                self._do_trip(t, "overcurrent", f"|i|={peak_pu:.4f} pu > {overcurrent.limit_pu} pu")
                return True
        return False

    # ---------------------------------------------------------- sampled
    def _timed(self, t: float, met: bool, first: str, alarm: str, timer: str, cause: str, detail) -> bool:
        """A timed criterion: alarm when first met, trip once met for ``hold``; return True on a trip."""
        if met and getattr(self.stats, first) < 0.0:
            setattr(self.stats, first, t)
            self._raise(alarm)
        if self._timers[timer].update(met) >= self.cfg.hold:
            self._do_trip(t, cause, f"{detail()} held {self.cfg.hold} s")
            return True
        return False

    def check_sampled(self, t: float, vac_pu: float, freq_dev_hz: float, vdc_err_pu: float) -> None:
        """Check the timed criteria and the ROCOF alarm; call once per control period.

        ``vac_pu`` ac voltage magnitude (pu), ``freq_dev_hz`` frequency deviation (Hz),
        ``vdc_err_pu`` dc-voltage error (pu).
        """
        cfg, stats, armed = self.cfg, self.stats, self.armed(t)
        if not self.tripped:
            old = self._rocof.push(freq_dev_hz)
            if old is not None:
                rocof = (freq_dev_hz - old) / self._rocof_window
                stats.rocof_max_hz_s = max(stats.rocof_max_hz_s, abs(rocof))
                if cfg.rocof.enable and abs(rocof) >= cfg.rocof.limit and armed:
                    if stats.rocof_first_t < 0.0:
                        stats.rocof_first_t = t
                    self._raise("ROCOF")
        if self.tripped or not armed:
            return
        if cfg.dc_voltage.enable and self._timed(
                t, abs(vdc_err_pu) >= cfg.dc_voltage.limit_pu,
                "vdc_band_first_t", "VDC_BAND", "hold_vdc", "vdc",
                lambda: f"|vdc error|={abs(vdc_err_pu):.4f} pu"):
            return
        if cfg.frequency.enable and self._timed(
                t, abs(freq_dev_hz) >= cfg.frequency.limit,
                "freq_band_first_t", "FREQ_BAND", "hold_freq", "freq",
                lambda: f"|f_pll - f0|={abs(freq_dev_hz):.4f} Hz"):
            return
        if stats.vac_min_pu < 0.0 or vac_pu < stats.vac_min_pu:
            stats.vac_min_pu = vac_pu
        stats.vac_max_pu = max(stats.vac_max_pu, vac_pu)
        if cfg.undervoltage.enable and self._timed(
                t, vac_pu <= cfg.undervoltage.limit_pu,
                "vac_uv_first_t", "VAC_UNDER", "hold_uv", "vac_uv",
                lambda: f"|v|={vac_pu:.4f} pu <= {cfg.undervoltage.limit_pu} pu"):
            return
        if cfg.overvoltage.enable:
            self._timed(t, vac_pu >= cfg.overvoltage.limit_pu,
                        "vac_ov_first_t", "VAC_OVER", "hold_ov", "vac_ov",
                        lambda: f"|v|={vac_pu:.4f} pu >= {cfg.overvoltage.limit_pu} pu")
