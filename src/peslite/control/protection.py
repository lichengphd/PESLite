"""Protection of a converter unit: over-current, ac-voltage, frequency and dc-voltage trips and a ROCOF alarm.

Trips are enabled while ``armed(t)``; timed criteria must hold for ``hold`` (s).
Named states: ``tripped`` and the hold timers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

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
    """Protection observations; absent crossings and samples are ``None``."""

    overcurrent_first_t: Optional[float] = None
    overcurrent_steps: int = 0
    undervoltage_first_t: Optional[float] = None
    overvoltage_first_t: Optional[float] = None
    frequency_first_t: Optional[float] = None
    dc_voltage_first_t: Optional[float] = None
    rocof_first_t: Optional[float] = None
    rocof_max: float = 0.0
    max_current_pu: float = 0.0
    vac_min_pu: Optional[float] = None
    vac_max_pu: Optional[float] = None
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
        stats, overcurrent = self.stats, self.cfg.overcurrent
        stats.max_current_pu = max(stats.max_current_pu, peak_pu)
        if overcurrent.enable and peak_pu > overcurrent.limit_pu:
            if stats.overcurrent_steps == 0:
                stats.overcurrent_first_t = t
                self._raise("OVERCURRENT")
            stats.overcurrent_steps += 1
            if not self.tripped and self.armed(t):
                self._do_trip(t, "overcurrent", f"|i|={peak_pu:.4f} pu > {overcurrent.limit_pu} pu")
                return True
        return False

    # ---------------------------------------------------------- sampled
    def _timed(self, t: float, met: bool, criterion: str, alarm: str, timer: str) -> bool:
        """Record a timed criterion and return whether it is currently held long enough."""
        if met and getattr(self.stats, f"{criterion}_first_t") is None:
            setattr(self.stats, f"{criterion}_first_t", t)
            self._raise(alarm)
        held = self._timers[timer].update(met)
        return met and held >= self.cfg.hold

    def check_sampled(self, t: float, vac_pu: float, freq_dev: float, vdc_err_pu: float) -> None:
        """Check the timed criteria and the ROCOF alarm; call once per control period.

        ``vac_pu`` ac voltage magnitude (pu), ``freq_dev`` frequency deviation (Hz),
        ``vdc_err_pu`` dc-voltage error (pu).
        """
        cfg, stats, armed = self.cfg, self.stats, self.armed(t)
        if not self.tripped:
            old = self._rocof.push(freq_dev)
            if old is not None:
                rocof = (freq_dev - old) / self._rocof_window
                stats.rocof_max = max(stats.rocof_max, abs(rocof))
                if cfg.rocof.enable and abs(rocof) >= cfg.rocof.limit and armed:
                    if stats.rocof_first_t is None:
                        stats.rocof_first_t = t
                    self._raise("ROCOF")
        if self.tripped or not armed:
            return
        if cfg.dc_voltage.enable and self._timed(
                t, abs(vdc_err_pu) >= cfg.dc_voltage.limit_pu, "dc_voltage", "VDC_BAND", "hold_vdc"):
            self._do_trip(t, "vdc", f"|vdc error|={abs(vdc_err_pu):.4f} pu held {cfg.hold} s")
            return
        if cfg.frequency.enable and self._timed(
                t, abs(freq_dev) >= cfg.frequency.limit, "frequency", "FREQ_BAND", "hold_freq"):
            self._do_trip(t, "freq", f"|f_pll - f0|={abs(freq_dev):.4f} Hz held {cfg.hold} s")
            return
        stats.vac_min_pu = vac_pu if stats.vac_min_pu is None else min(stats.vac_min_pu, vac_pu)
        stats.vac_max_pu = vac_pu if stats.vac_max_pu is None else max(stats.vac_max_pu, vac_pu)
        if cfg.undervoltage.enable and self._timed(
                t, vac_pu <= cfg.undervoltage.limit_pu, "undervoltage", "VAC_UNDER", "hold_uv"):
            self._do_trip(t, "vac_uv", f"|v|={vac_pu:.4f} pu <= {cfg.undervoltage.limit_pu} pu "
                          f"held {cfg.hold} s")
            return
        if cfg.overvoltage.enable and self._timed(
                t, vac_pu >= cfg.overvoltage.limit_pu, "overvoltage", "VAC_OVER", "hold_ov"):
            self._do_trip(t, "vac_ov", f"|v|={vac_pu:.4f} pu >= {cfg.overvoltage.limit_pu} pu "
                          f"held {cfg.hold} s")
