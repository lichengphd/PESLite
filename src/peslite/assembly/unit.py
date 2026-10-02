"""A converter unit: power stage, sensing, controller and protection.

Switching and PWM-period averaging use an ADC and a closed sampled controller. Ideal averaging
wires instantaneous measurements and a continuous controller into the plant ODE. All protection
decisions, their single trip latch, breakers and gates belong to the unit.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ContextManager, Mapping, Optional, Protocol

import numpy as np

from ..components.adc import ADC, MeasurementPorts
from ..components.converter import make_bridge, make_dclink
from ..components.network import RLBranch, Terminal
from ..components.pwm import TIME_EPS, Modulator
from ..control.blocks import phases
from ..control.controller import Controller, Measurement, make_controller
from .events import Scenario
from .params import SimulationParams, UnitParams
from .protection import Protection

__all__ = ["Plant", "Unit"]


class Plant(Protocol):
    """The plant operations a unit may use during a run."""

    def outputs(self, t: float) -> None: ...

    def hold(self, label: str, value: Any) -> None: ...

    def change(self) -> ContextManager[None]: ...


class Unit:
    """One converter unit built from its parameter section, connected to ``bus``.

    ``ctrl`` and ``modulator`` replace the parts built from the section. The unit samples with
    ``adc`` and controls a bridge which owns its model-specific actuation timing.
    Events connect or disconnect its filter and DC source and may retune runtime parameters.
    """

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = sc = Scenario(cfg.events)

        # ------------------------------------------------------------ power
        T_c = cfg.ctrl.period
        self.dclink = make_dclink(cfg.dclink)
        t0 = sim.initial.t
        connected, since, ramp = sc.at(t0)
        self.dclink.connect(connected, since, ramp if t0 < since + ramp else 0.0)
        try:
            self.bridge = make_bridge(cfg, sim, modulator)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
        self.branch_f = RLBranch(cfg.ac_filter.l_f, cfg.ac_filter.r_f)
        self.continuous = bool(getattr(self.bridge, "continuous", False))
        self.breaker = True
        self.gates = False
        protection_period = (sim.output.period if self.continuous else cfg.ctrl.period)
        self.protection = Protection(cfg.protection, protection_period, cfg.base.i_phase_peak)
        self._trip_due = False
        self._diodes_warned = False

        # ------------------------------------------------------------ ADC
        # instantaneous samples or window means; the window holds only the averaged channels
        meas = cfg.meas
        zero = {"u_g": 0j, "i_c": 0j, "u_dc": 0.0}
        channels = {name: zero[name] for name in meas.average}
        self.ports = MeasurementPorts(
            u_g=lambda: bus.out.u, i_c=lambda: self.branch_f.out.i,
            i_c_state=lambda: self.branch_f.state.i, u_dc=lambda: self.dclink.out.u_dc,
            i_dc=lambda: self.dclink.inp.i_dc)
        if self.continuous:
            self.adc = None
            self.windowed = self._oversamples = False
        else:
            sample_period = meas.period if meas.period is not None else T_c
            self.adc = ADC(self.ports, T_c, sample_period,
                           meas.window if meas.window is not None else T_c, channels)
            self.windowed = self.adc.averaging
            self.accumulate, self.restart_windows = self.adc.accumulate, self.adc.seed
            self._oversamples = self.adc.samples > 1

        # ------------------------------------------------------------ bridge actuation and control
        self.bridge.reset(np.full(3, 0.5), on=False)
        self.zoh = f"{name}.q"                   # model label of the bridge's held switching state
        self.ctrl = ctrl if ctrl is not None else make_controller(cfg)
        if bool(getattr(self.ctrl, "continuous", False)) != self.continuous:
            raise TypeError(
                f"{name}: the controller and bridge must both be sampled or both be continuous"
            )
        parts = [(self.ctrl, Controller)]
        modulator = getattr(self.bridge, "modulator", None)
        if modulator is not None:
            parts.append((modulator, Modulator))
        for obj, proto in parts:
            if not isinstance(obj, proto):
                raise TypeError(f"{name}: {type(obj).__name__} does not satisfy the {proto.__name__} protocol")
        self._initially_connected = False
        self._track_due = False
        self._write_due: Optional[tuple[float, Any]] = None

    # ---------------------------------------------------------------- assembly
    def subsystems(self) -> dict[str, Any]:
        """Return this unit's blocks keyed by namespaced name."""
        parts = {f"{self.name}.branch_f": self.branch_f,
                 f"{self.name}.bridge": self.bridge,
                 f"{self.name}.dclink": self.dclink}
        if self.continuous:
            parts[f"{self.name}.ctrl"] = self.ctrl
        return parts

    def connections(self) -> dict:
        """Return this unit's internal and measurement wiring."""
        connections = {
            **self.bridge.connections(self.dclink, self.branch_f),
        }
        if self.continuous:
            connections.update({
                (self.ctrl, "u_g"): (self.bus, "u"),
                (self.ctrl, "i_c"): (self.branch_f, "i"),
                (self.ctrl, "u_dc"): (self.dclink, "u_dc"),
                (self.bridge, "q"): (self.ctrl, "q"),
            })
        return connections

    def zoh_connections(self) -> dict:
        return {} if self.continuous else self.bridge.zoh_connections(self.zoh)

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        """Bind the filter's second terminal to the unit bus."""
        return ((self.cfg.bus, self.branch_f.terminal2),)

    def aliases(self) -> dict[str, str]:
        """Return the aliases ``<unit>.i_c`` (filter current) and ``<unit>.u_dc`` (dc-link voltage)."""
        out = {f"{self.name}.i_c": f"{self.name}.branch_f.i"}
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    def state_parts(self) -> list[tuple[str, Any]]:
        """Return this unit's fixed state owners and their public prefixes."""
        parts = [(self.name, self), (f"{self.name}.ctrl", self.ctrl),
                 (f"{self.name}.{self.bridge.state_prefix}", self.bridge.state_owner)]
        if self.adc is not None and self.adc.averaging:
            parts.append((f"{self.name}.meas", self.adc))
        return parts

    @property
    def periods(self) -> list[float]:
        if self.continuous:
            return []
        return [*self.bridge.event_periods,
                *(period for period in getattr(self.ctrl, "periods", {}).values() if period)]

    # ---------------------------------------------------------------- states
    @property
    def tripped(self) -> bool:
        return self.protection.tripped

    def get_state(self) -> dict[str, Any]:
        return {"tripped": self.tripped, "fault": self.protection.fault,
                **{f"prot.{name}": value for name, value in self.protection.get_state().items()}}

    def set_state(self, values: Mapping[str, Any]) -> None:
        self.protection.set_state({name[5:]: value for name, value in values.items()
                                   if name.startswith("prot.")})
        tripped = values.get("tripped")
        if tripped is None and values.get("fault"):
            tripped = True
        self.protection.restore_latches(tripped, values.get("fault"))
        if self.tripped:
            self.trip(math.nan, "restored")

    # ---------------------------------------------------------------- events and trips
    def _terminal(self) -> None:
        if self.breaker and self.gates and not self.tripped:
            self.branch_f.close_breaker()
        else:
            self.branch_f.open_breaker()

    @property
    def blocked(self) -> bool:
        return self.breaker and not self.gates and not self.tripped

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect/disconnect the unit and give the controller its run/stop command."""
        was_active = self.breaker and self.gates and not self.tripped
        effective = bool(on) and not self.tripped
        self.breaker = effective
        self._terminal()
        self.dclink.connect(effective, t, ramp)
        command_at = getattr(self.ctrl, "command_at", None)
        command = getattr(self.ctrl, "command", None)
        if self.continuous and command_at is not None:
            self._track_due = effective and not was_active
            command_at(effective, ramp, t)
            self.bridge.enabled = effective
            self._set_gates(effective)
        elif command is not None:
            command(effective, ramp)

    def _set_gates(self, on: bool) -> None:
        self.gates = bool(on) and not self.tripped
        self._terminal()

    def retune(self, cfg: UnitParams, paths: list[str], t: float) -> None:
        """Apply runtime-changeable DC-source, control and protection parameters."""
        self.cfg = cfg
        if any(path.startswith("dclink.source.") for path in paths):
            source = cfg.dclink.source
            self.dclink.retune_source(source.i, source.k, source.v, source.r)
        if any(path.startswith("protection.") for path in paths):
            self.protection.retune(cfg.protection)
        rest = [path for path in paths
                if not path.startswith(("dclink.", "protection."))]
        if rest:
            self.ctrl.retune(cfg, rest, t)

    def trip(self, t: float, cause: str, plant: Optional[Plant] = None, detail: str = "") -> None:
        """Latch one unit trip, stop the controller and open its gates, AC and DC sides."""
        if plant is not None:
            with plant.change():
                self.trip(t, cause, detail=detail)
            return
        self.protection.latch(t, cause, detail)
        self._track_due = False
        command_at = getattr(self.ctrl, "command_at", None)
        command = getattr(self.ctrl, "command", None)
        if self.continuous and command_at is not None:
            command_at(False, 0.0, t)
            self.bridge.enabled = False
        elif command is not None:
            command(False)
        self.gates, self.breaker = False, False
        self._terminal()
        self.dclink.open_breaker()

    # ---------------------------------------------------------------- run instants
    def start(self, t: float) -> None:
        """Prepare the controller command before initial named states are loaded."""
        self._initially_connected, since, ramp = self.scenario.at(t)
        command_at = getattr(self.ctrl, "command_at", None)
        command = getattr(self.ctrl, "command", None)
        if command is None and command_at is None:
            return
        if self.continuous and command_at is not None:
            start = since if self._initially_connected and math.isfinite(since) else t
            command_at(self._initially_connected,
                       ramp if self._initially_connected else 0.0,
                       start, reset=False)
        else:
            command(self._initially_connected, ramp if self._initially_connected else 0.0)
        if not self._initially_connected:
            return
        if self.continuous:
            return
        states = getattr(self.ctrl, "get_state", None)
        if math.isfinite(since) and states is not None and "startup.steps" in states():
            self.ctrl.set_state({"startup.steps": self.bridge.control_count(since, t)})

    def states_loaded(self) -> None:
        """Apply the loaded PWM enable to the physical gates."""
        given_current = self.branch_f.state.i
        if self.tripped:
            command_at = getattr(self.ctrl, "command_at", None)
            command = getattr(self.ctrl, "command", None)
            if self.continuous and command_at is not None:
                command_at(False, 0.0, 0.0, reset=False)
            elif command is not None:
                command(False)
        if self.continuous:
            # There is deliberately no PWM/register state to restore.  The persisted start-up
            # command is the source of truth for whether the ideal bridge is in service.  Apply
            # it before checking the restored branch current so a continuation never clears a
            # valid current merely because the bridge object was constructed disabled.
            self.bridge.enabled = self.ctrl.startup.active and not self.tripped
        self._set_gates(self.bridge.on)
        if self.blocked and given_current != 0j:
            warnings.warn(
                f"simulation.initial.states: unit {self.name!r} starts with its PWM blocked, "
                f"so {self.name}.branch_f.i starts at zero, not at the value given",
                stacklevel=5,
            )

    def begin(self, t: float, plant: Plant, continued: bool, window_given: bool) -> None:
        """Pre-synchronise, resume PWM timing and reconstruct the ADC at run start."""
        track = getattr(self.ctrl, "track", None)
        if self._initially_connected and not continued and track is not None:
            plant.outputs(t)
            ports = self.ports
            track(Measurement(t, ports.u_g(), ports.i_c(), ports.u_dc(),
                              np.asarray(phases(ports.i_c()))))
        self._track_due = False
        if self.continuous:
            self.ctrl.startup.at(t)
            self.bridge.enabled = self.ctrl.startup.active
            self._set_gates(self.bridge.on)
            return
        sync = self.ctrl.initial_sync() if hasattr(self.ctrl, "initial_sync") else (None, None)
        plant.hold(self.zoh, self.bridge.start(t, sync, continued))
        if self.windowed:
            plant.outputs(t)
            self.adc.seed()
        bridge = self.bridge
        self.adc.start(t, bridge.previous_control_time(), bridge.next_control_time,
                       held=not window_given)

    def next_time(self) -> float:
        if self.continuous:
            return math.inf
        bridge = self.bridge
        next_ = bridge.next_time()
        if self.windowed:
            next_ = min(next_, self.adc.t_window(bridge.next_control_time))
        if self._oversamples:
            next_ = min(next_, self.adc.t_sample(bridge.previous_control_time()))
        return next_

    def protect(self, t: float, plant: Plant) -> bool:
        """Run fast protection before any controller samples at this instant."""
        if self.tripped or (not self.continuous and not self.bridge.fast_edge(t)):
            return False
        if not self.protection.check_fast(t, phases(self.branch_f.state.i)):
            return False
        event = self.protection.trip
        assert event is not None
        self.trip(event.t, event.cause, plant, event.detail)
        return True

    def sense(self, t: float, plant: Plant) -> Optional[tuple[int, Any]]:
        """Take oversamples or run the controller interrupt, before any unit actuates."""
        if self.continuous:
            plant.outputs(t)
            if self._track_due:
                ports = self.ports
                with plant.change():
                    self.ctrl.track(Measurement(
                        t, ports.u_g(), ports.i_c(), ports.u_dc(),
                        np.asarray(phases(ports.i_c())),
                    ))
                self._track_due = False
                plant.outputs(t)
            startup = self.ctrl.startup
            startup.at(t)
            if not self.tripped:
                vac_pu = abs(self.ports.u_g()) / self.cfg.base.v_phase_peak
                vdc_pu = self.ports.u_dc() / self.cfg.dc_base.v
                omega = self.ctrl.omega
                freq_dev = (omega - self.cfg.base.w0) / (2.0 * math.pi)
                vdc_ref = self.cfg.ctrl.references.vdc_ref_pu
                if self.protection.check_sampled(
                        t, vac_pu, freq_dev, vdc_pu - vdc_ref, startup.complete):
                    self._trip_due = True
            return None
        bridge = self.bridge
        if not bridge.control_due(t):
            if (self._oversamples
                    and abs(self.adc.t_sample(bridge.previous_control_time()) - t) < TIME_EPS):
                plant.outputs(t)
                self.adc.peek(t)
            return None
        plant.outputs(t)
        index = bridge.control_index
        measurement = self.adc.sample(t)
        if self.blocked and not self._diodes_warned:
            self._warn_diodes(t, measurement)
        output = self.ctrl(t, measurement)
        if not self.tripped:
            vac_pu = abs(measurement.u_g) / self.cfg.base.v_phase_peak
            vdc_pu = measurement.u_dc / self.cfg.dc_base.v
            if output.omega is None:
                if self.protection.needs_frequency:
                    raise ValueError(
                        f"unit {self.name!r}: frequency or ROCOF protection needs "
                        "the controller's omega output")
                freq_dev = 0.0
            else:
                freq_dev = (output.omega - self.cfg.base.w0) / (2.0 * math.pi)
            vdc_ref = self.cfg.ctrl.references.vdc_ref_pu
            if self.protection.check_sampled(
                    t, vac_pu, freq_dev, vdc_pu - vdc_ref,
                    getattr(output, "startup_complete", True)):
                self._trip_due = True
        # Keep the result only until every unit has sampled this same instant.  It is then written
        # to the one shadow register in actuate(); this is not another delayed register or state.
        self._write_due = (t, output)
        return index, output

    def _warn_diodes(self, t: float, measurement: Measurement) -> None:
        u_g = measurement.u_g_raw if measurement.u_g_raw is not None else measurement.u_g
        u_dc = measurement.u_dc_raw if measurement.u_dc_raw is not None else measurement.u_dc
        v_ll_peak = math.sqrt(3.0) * abs(u_g)
        if v_ll_peak > u_dc:
            self._diodes_warned = True
            warnings.warn(
                f"t = {t:.6g} s: unit {self.name!r} has blocked gates with peak line voltage "
                f"({v_ll_peak:.4g} V) above dc voltage ({u_dc:.4g} V); its diodes would conduct, "
                "which the open-circuit model leaves out",
                stacklevel=4,
            )

    def actuate(self, t: float, plant: Plant) -> bool:
        """Apply a sampled-protection trip, PWM load, ADC-window opening and bridge switches."""
        tripped_now = self._trip_due
        if tripped_now:
            self._trip_due = False
            event = self.protection.trip
            assert event is not None
            self.trip(event.t, event.cause, plant, event.detail)
        if self.continuous:
            return tripped_now
        bridge = self.bridge
        write = self._write_due
        changes = bridge.actuate(t, write, not self.tripped)
        enabled = bridge.on and not self.tripped
        if enabled != self.gates:
            with plant.change():
                self._set_gates(enabled)
        for value in changes:
            plant.hold(self.zoh, value)
        self._write_due = None
        if self.windowed and abs(self.adc.t_window(bridge.next_control_time) - t) < TIME_EPS:
            self.adc.open()
        return tripped_now

    # ---------------------------------------------------------------- results
    def summary(self) -> dict[str, Any]:
        seen = dict(self.ctrl.summary()) if hasattr(self.ctrl, "summary") else {}
        seen.update(self.protection.summary())
        return seen

    def signals(self) -> dict[str, float | complex]:
        """Return this unit's recorded signals (SI); outputs must be synced first."""
        ports = self.ports
        return {f"{self.name}.i_c": ports.i_c(), f"{self.name}.u_g": ports.u_g(),
                f"{self.name}.u_dc": ports.u_dc(), f"{self.name}.i_dc": ports.i_dc()}
