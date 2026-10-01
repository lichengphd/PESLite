"""A converter unit: power stage, ADC/PWM peripherals, controller and protection.

The controller is a closed sampled block. All protection decisions, their single trip latch,
breakers and gates belong to the unit and act on the continuous plant.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ContextManager, Mapping, Optional, Protocol

import numpy as np

from ..components.adc import ADC, MeasurementPorts
from ..components.converter import Bridge, make_dclink
from ..components.network import RLBranch
from ..components.pwm import PWM, TIME_EPS, Modulator, make_modulator
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
    ``adc``, controls with ``ctrl`` and switches its bridge through ``pwm``.
    Events connect or disconnect its filter and DC source and may retune runtime parameters.
    """

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = sc = Scenario(cfg.events)

        # ------------------------------------------------------------ power
        self.dclink = make_dclink(cfg.dclink)
        t0 = sim.initial.t
        since, ramp = sc.since(t0)
        self.dclink.connect(sc.connected(t0), since, ramp if t0 < since + ramp else 0.0)
        self.bridge = Bridge()
        self.branch_f = RLBranch(cfg.ac_filter.l_f, cfg.ac_filter.r_f)
        self.breaker = True
        self.gates = False
        self.protection = Protection(cfg.protection, cfg.ctrl.period, cfg.base.i_phase_peak)
        self._trip_due = False
        self._diodes_warned = False

        # ------------------------------------------------------------ ADC
        # instantaneous samples or window means; the window holds only the averaged channels
        meas = cfg.meas
        channels: dict[str, complex | float] = {}
        if meas.average == "window":
            channels["v"], channels["i"] = 0j, 0j
        if meas.u_dc == "window":
            channels["dc"] = 0.0
        self.ports = MeasurementPorts(
            u_g=lambda: bus.out.u, i_c=lambda: self.branch_f.out.i,
            i_c_state=lambda: self.branch_f.state.i, u_dc=lambda: self.dclink.out.u_dc,
            i_dc=lambda: self.dclink.inp.i_dc)
        T_c = cfg.ctrl.period
        sample_period = meas.period if meas.period is not None else T_c
        self.adc = ADC(self.ports, T_c, sample_period,
                       meas.window if meas.window is not None else T_c, channels)
        self.windowed = self.adc.averaging
        self.accumulate, self.restart_windows = self.adc.accumulate, self.adc.seed
        self._oversamples = self.adc.samples > 1

        # ------------------------------------------------------------ PWM and control
        pwm = cfg.pwm
        self.pwm = PWM(
            T_c, pwm.load_period, pwm.grid_offset, cfg.ctrl.computation, pwm.switching_period,
            modulator if modulator is not None else make_modulator(pwm, cfg.base.f0, cfg.averaging, sim),
        )
        self.pwm.reset(np.full(3, 0.5), on=False)
        self.zoh = f"{name}.q"                   # model label of the bridge's held switching state
        self.ctrl = ctrl if ctrl is not None else make_controller(cfg)
        for obj, proto in ((self.ctrl, Controller), (self.pwm.modulator, Modulator)):
            if not isinstance(obj, proto):
                raise TypeError(f"{name}: {type(obj).__name__} does not satisfy the {proto.__name__} protocol")
        self._running = False
        self._write_due: Optional[tuple[float, Any]] = None

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

    def state_parts(self) -> list[tuple[str, Any]]:
        """Return this unit's fixed state owners and their public prefixes."""
        parts = [(self.name, self), (f"{self.name}.ctrl", self.ctrl),
                 (f"{self.name}.pwm", self.pwm)]
        if self.adc.averaging:
            parts.append((f"{self.name}.meas", self.adc))
        return parts

    @property
    def periods(self) -> list[float]:
        pwm = self.pwm
        return [pwm.period, pwm.load_period, pwm.carrier_period,
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
        effective = bool(on) and not self.tripped
        self.breaker = effective
        self._terminal()
        self.dclink.connect(effective, t, ramp)
        command = getattr(self.ctrl, "command", None)
        if command is not None:
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
        command = getattr(self.ctrl, "command", None)
        if command is not None:
            command(False)
        self.gates, self.breaker = False, False
        self._terminal()
        self.dclink.open_breaker()

    # ---------------------------------------------------------------- run instants
    def start(self, t: float) -> None:
        """Prepare the controller command before initial named states are loaded."""
        self._running = self.scenario.connected(t)
        command = getattr(self.ctrl, "command", None)
        if not self._running or command is None:
            return
        since, ramp = self.scenario.since(t)
        command(True, ramp)
        states = getattr(self.ctrl, "get_state", None)
        if math.isfinite(since) and states is not None and "sequence.steps" in states():
            self.ctrl.set_state({"sequence.steps": self.pwm.count(since, t)})

    def states_loaded(self) -> None:
        """Apply the loaded PWM enable to the physical gates."""
        given_current = self.branch_f.state.i
        if self.tripped:
            command = getattr(self.ctrl, "command", None)
            if command is not None:
                command(False)
        self._set_gates(self.pwm.on)
        if self.blocked and given_current != 0j:
            warnings.warn(
                f"simulation.initial.states: unit {self.name!r} starts with its PWM blocked, "
                f"so {self.name}.branch_f.i starts at zero, not at the value given",
                stacklevel=5,
            )

    def begin(self, t: float, plant: Plant, continued: bool, window_given: bool) -> None:
        """Pre-synchronise, resume PWM timing and reconstruct the ADC at run start."""
        track = getattr(self.ctrl, "track", None)
        if self._running and not continued and track is not None:
            plant.outputs(t)
            ports = self.ports
            track(Measurement(t, ports.u_g(), ports.i_c(), ports.u_dc(), self.adc.phase_currents()))
        sync = self.ctrl.initial_sync() if hasattr(self.ctrl, "initial_sync") else (None, None)
        plant.hold(self.zoh, self.pwm.start(t, sync, continued))
        if self.windowed:
            plant.outputs(t)
            self.adc.seed()
        pwm = self.pwm
        self.adc.start(t, pwm.interrupt(pwm.k - 1), pwm.t_interrupt, held=not window_given)

    def next_time(self) -> float:
        pwm = self.pwm
        next_ = min(pwm.t_interrupt, pwm.t_load, pwm.next_switch)
        if self.windowed:
            next_ = min(next_, self.adc.t_window(pwm.t_interrupt))
        if self._oversamples:
            next_ = min(next_, self.adc.t_sample(pwm.interrupt(pwm.k - 1)))
        return next_

    def protect(self, t: float, plant: Plant) -> bool:
        """Run fast protection before any controller samples at this instant."""
        if self.tripped or not self.pwm.edge(t):
            return False
        if not self.protection.check_fast(t, phases(self.branch_f.state.i)):
            return False
        event = self.protection.trip
        assert event is not None
        self.trip(event.t, event.cause, plant, event.detail)
        return True

    def sense(self, t: float, plant: Plant) -> Optional[tuple[int, Any]]:
        """Take oversamples or run the controller interrupt, before any unit actuates."""
        pwm = self.pwm
        if abs(pwm.t_interrupt - t) >= TIME_EPS:
            if self._oversamples and abs(self.adc.t_sample(pwm.interrupt(pwm.k - 1)) - t) < TIME_EPS:
                plant.outputs(t)
                self.adc.peek(t)
            return None
        plant.outputs(t)
        index = pwm.k
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
        pwm = self.pwm
        write = self._write_due
        if write is not None and pwm.computation == 0.0:
            at, output = write
            pwm.write(at, output.d_abc, getattr(output, "gates", True) and not self.tripped,
                      output.theta, output.omega)
        if abs(pwm.t_load - t) < TIME_EPS:
            switching = pwm.load(t)
            enabled = pwm.on and not self.tripped
            if enabled != self.gates:
                with plant.change():
                    self._set_gates(enabled)
            plant.hold(self.zoh, switching)
        if write is not None and pwm.computation > 0.0:
            at, output = write
            pwm.write(at, output.d_abc, getattr(output, "gates", True) and not self.tripped,
                      output.theta, output.omega)
        self._write_due = None
        if self.windowed and abs(self.adc.t_window(pwm.t_interrupt) - t) < TIME_EPS:
            self.adc.open()
        for switching in pwm.switches(t):
            plant.hold(self.zoh, switching)
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
