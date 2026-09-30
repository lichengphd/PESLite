"""A converter unit: dc link, bridge and filter branch, its ADC, controller and PWM.

Plant quantities are in SI; the controller works in the unit's pu bases.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..components.adc import ADC, MeasurementPorts
from ..components.converter import Bridge, make_dclink
from ..components.network import RLBranch
from ..components.pwm import PWM, ComputationDelay, Delay, Modulator, make_modulator
from ..control.controller import Controller, make_controller
from .events import UnitScenario
from .params import SimulationParams, UnitParams

__all__ = ["Unit"]


class Unit:
    """One converter unit built from its parameter section, connected to ``bus``.

    ``ctrl``, ``modulator``, ``delay``: replacements of the parts built from the section. The unit
    samples with ``adc``, controls with ``ctrl`` and switches its bridge through ``pwm``.
    Events connect or disconnect its filter and DC source and may retune runtime parameters.
    """

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None,
                 delay: Optional[Delay] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = sc = UnitScenario(cfg.events)

        # ------------------------------------------------------------ power
        self.dclink = make_dclink(cfg.dclink)
        t0 = sim.initial.t
        since, ramp = sc.since(t0)
        self.dclink.connect(sc.connected(t0), since, ramp if t0 < since + ramp else 0.0)
        self.bridge = Bridge()
        self.branch_f = RLBranch(cfg.ac_filter.l_f, cfg.ac_filter.r_f)
        self.breakers = [self.branch_f, self.dclink]
        self.tripped = False

        # ------------------------------------------------------------ ADC
        # instantaneous samples or window means; the window holds only the averaged channels
        meas = cfg.measurement
        channels: dict[str, complex | float] = {}
        if meas.average == "window":
            channels["v"], channels["i"] = 0j, 0j
        if meas.u_dc == "window":
            channels["dc"] = 0.0
        self.ports = MeasurementPorts(
            u_g=lambda: bus.out.u, i_c=lambda: self.branch_f.out.i,
            i_c_state=lambda: self.branch_f.state.i, u_dc=lambda: self.dclink.out.u_dc,
            i_dc=lambda: self.dclink.inp.i_dc)
        T_s = cfg.pwm.update_period  # PWM publication interval
        self.adc = ADC(self.ports, T_s, int(cfg.control.samples_per_update),
                       meas.window if meas.window is not None else T_s, channels)

        # ------------------------------------------------------------ control
        self.ctrl = ctrl if ctrl is not None else make_controller(cfg, sc)
        self.pwm = PWM(T_s, modulator if modulator is not None else
                       make_modulator(cfg.pwm, cfg.base.f0, cfg.averaging, sim),
                       delay if delay is not None else ComputationDelay(cfg.delay.steps))
        self.zoh = f"{name}.q"                   # model label of the bridge's held switching state

    # ---------------------------------------------------------------- assembly
    def subsystems(self) -> dict[str, Any]:
        """Return this unit's blocks keyed by namespaced name."""
        return {f"{self.name}.branch_f": self.branch_f, f"{self.name}.bridge": self.bridge,
                f"{self.name}.dclink": self.dclink}

    def connections(self) -> dict:
        """Return this unit's internal wiring and its filter branch's connection to the bus."""
        return {(self.branch_f, "u_to"): (self.bus, "u"), **self.bridge.connections(self.dclink, self.branch_f)}

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
        """Return the aliases ``<unit>.i_c`` (filter current) and ``<unit>.u_dc`` (dc-link voltage)."""
        out = {f"{self.name}.i_c": f"{self.name}.branch_f.i"}
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    # ---------------------------------------------------------------- states
    def get_state(self) -> dict[str, Any]:
        return {"tripped": self.tripped}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values.get("tripped", False) and not self.tripped:
            self.trip()

    # ---------------------------------------------------------------- the run
    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the filter branch and DC source at ``t``."""
        if on and not self.tripped:
            self.branch_f.close_breaker()
        elif not on:
            self.branch_f.open_breaker()
        self.dclink.connect(on, t, ramp)

    def retune(self, cfg: UnitParams, paths: list[str], t: float) -> None:
        """Apply runtime-changeable DC-source, control and protection parameters."""
        if any(path.startswith("dclink.source.") for path in paths):
            source = cfg.dclink.source
            self.dclink.retune_source(source.i, source.k, source.v, source.r)
        rest = [path for path in paths if not path.startswith("dclink.")]
        if rest:
            self.ctrl.retune(cfg, rest, t)

    def trip(self) -> None:
        """Open this unit's breakers (filter branch and dc source); other units keep running.

        The caller must repack the solver state vector afterwards.
        """
        self.tripped = True
        for breaker in self.breakers:
            breaker.open_breaker()

    def signals(self) -> dict[str, float | complex]:
        """Return this unit's recorded signals (SI); outputs must be synced first."""
        ports = self.ports
        return {f"{self.name}.i_c": ports.i_c(), f"{self.name}.u_g": ports.u_g(),
                f"{self.name}.u_dc": ports.u_dc(), f"{self.name}.i_dc": ports.i_dc()}
