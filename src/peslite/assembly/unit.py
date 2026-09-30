"""Converter unit: power stage, sensing, controller and PWM peripherals.

Plant quantities are in SI; the controller works in the unit's pu bases. The
controller itself lives in :mod:`peslite.control`; this module only assembles
it with the hardware that has not yet moved to the refactored components layer.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..control import Controller, Measurement, make_controller
from ..firmware import ComputationDelay
from ..modulation import make_modulator
from ..params import SimulationParams, UnitParams
from ..power import Bridge, RLBranch, make_dclink
from ..sensing import MeasurementPorts, Sampler, SamplingWindow
from .events import UnitScenario
from .protocols import Delay, Modulator

__all__ = ["Unit"]


class Unit:
    """One converter unit built from its parameter section, connected to ``bus``."""

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None,
                 delay: Optional[Delay] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = sc = UnitScenario(cfg)
        base = cfg.base

        # ------------------------------------------------------------ power
        self.dclink = make_dclink(cfg.dclink, ramp=sc.startup)
        self.bridge = Bridge()
        self.branch_f = RLBranch(cfg.ac_filter.l_f, cfg.ac_filter.r_f)

        # ------------------------------------------------------------ sensing
        # ADC: instantaneous samples or window means; the window holds only the averaged channels
        meas = cfg.measurement
        channels: dict[str, complex | float] = {}
        if meas.average == "window":
            channels["v"], channels["i"] = 0j, 0j
        if meas.u_dc == "window":
            channels["dc"] = 0.0
        self.window: Optional[SamplingWindow] = SamplingWindow(
            meas.window_s if meas.window_s is not None else cfg.pwm.update_period, channels
        ) if channels else None
        self.T_avg: Optional[float] = self.window.length if self.window is not None else None
        self.ports = MeasurementPorts(
            u_g=lambda: bus.out.u, i_c=lambda: self.branch_f.out.i,
            i_c_state=lambda: self.branch_f.state.i, u_dc=lambda: self.dclink.out.u_dc,
            i_dc=lambda: self.dclink.inp.i_dc)
        self.sampler = Sampler(self.ports, self.window)
        self.breakers = [self.branch_f, self.dclink]
        self.breaker_open = False

        # ------------------------------------------------------------ control
        self.ctrl = ctrl if ctrl is not None else make_controller(cfg, sc)
        self.modulator = modulator if modulator is not None else make_modulator(cfg.pwm, base.f0, sim)
        self.delay = delay if delay is not None else ComputationDelay(cfg.delay.steps)
        c = cfg.control
        self.T_s = cfg.pwm.update_period                    # PWM publication interval
        self.samples_per_update = int(c.samples_per_update)
        self.T_samp = self.T_s / self.samples_per_update   # sampling period
        self.zoh = f"{name}.q"                             # held bridge switching state

    # ---------------------------------------------------------------- assembly
    def subsystems(self) -> dict[str, Any]:
        """Return this unit's blocks keyed by namespaced name."""
        return {f"{self.name}.branch_f": self.branch_f, f"{self.name}.bridge": self.bridge,
                f"{self.name}.dclink": self.dclink}

    def connections(self) -> dict:
        """Return this unit's internal wiring and its filter branch's connection to the bus."""
        return {
            (self.branch_f, "u_to"): (self.bus, "u"),
            **self.bridge.connections(self.dclink, self.branch_f),
        }

    def zoh_connections(self) -> dict:
        return self.bridge.zoh_connections(self.zoh)

    @property
    def bus_name(self) -> str:
        return self.cfg.bus

    @property
    def injection(self) -> tuple:
        """Current this unit feeds into its bus: the filter branch current, positive into the node."""
        return (self.branch_f, "i")

    def aliases(self) -> dict[str, str]:
        """Return aliases for converter current and dc-link voltage."""
        out = {f"{self.name}.i_c": f"{self.name}.branch_f.i"}
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    def presets(self, t: float) -> dict[str, Any]:
        """Return this unit's ``initial.states`` keyword values."""
        return {"rated": self.cfg.dclink.vdc_ref}

    # ---------------------------------------------------------------- states
    def get_state(self) -> dict[str, Any]:
        return {"breaker_open": self.breaker_open}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values.get("breaker_open", False) and not self.breaker_open:
            self.trip()

    # ---------------------------------------------------------------- the loop
    def measure(self, t: float) -> Measurement:
        """Return the sampled measurement (SI) at ``t``; model outputs must be synced."""
        return self.sampler.measure(t)

    def open_window(self) -> None:
        """Open the next averaging window at the current instant."""
        self.sampler.open_window()

    def _measured(self) -> dict[str, complex | float]:
        ports = self.ports
        return {"v": ports.u_g(), "i": ports.i_c(), "dc": ports.u_dc()}

    def seed_window(self) -> None:
        """Seed the averaging window at the start instant."""
        if self.window is not None:
            self.window.seed(self._measured())

    def accumulate(self, dt: float) -> None:
        """Advance the averaging window over the interval ending now."""
        if self.window is not None:
            self.window.accumulate(dt, self._measured())

    def phase_currents(self) -> np.ndarray:
        """Return the instantaneous phase currents (A) from the state."""
        return self.sampler.phase_currents()

    def trip(self) -> None:
        """Open this unit's breakers; other units keep running."""
        self.breaker_open = True
        for breaker in self.breakers:
            breaker.open_breaker()

    def signals(self) -> dict[str, float | complex]:
        """Return this unit's recorded signals (SI); outputs must be synced first."""
        ports = self.ports
        return {f"{self.name}.i_c": ports.i_c(), f"{self.name}.u_g": ports.u_g(),
                f"{self.name}.u_dc": ports.u_dc(), f"{self.name}.i_dc": ports.i_dc()}
