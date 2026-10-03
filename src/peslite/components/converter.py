"""The converter's power stage: bridge models and the dc link.

Quantities are SI. Every bridge keeps the same AC/DC and held-modulation connections; a concrete
bridge also owns the timing which turns sampled controller commands into its modulation value.
"""
from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import smoothstep
from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, Empty, OutputStage
from .pwm import TIME_EPS, Modulator, PWM, ZOH, make_modulator

__all__ = ["Bridge", "PWMBridge", "AveragingBridge", "DCLink", "DCCapacitor",
           "DCCurrentSource", "DCVoltageSource", "make_bridge", "make_dclink"]


class _BridgeInp(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOut(Bag):
    __slots__ = ("u_c", "i_dc")


class Bridge:
    """Lossless bridge: ``u_c = q u_dc``, ``i_dc = 1.5 Re(q conj(i_c))`` (SI).

    ``connections`` and ``zoh_connections`` return the power and switching-state wiring.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    has_switching_events: ClassVar[bool] = False
    outputs_need_inputs: ClassVar[bool] = True
    dirac: ClassVar[bool] = True
    output_stages: ClassVar[tuple[OutputStage, ...]] = (
        OutputStage("set_dc_current", inputs=("q", "i_c"), outputs=("i_dc",)),
        OutputStage("set_ac_voltage", inputs=("q", "u_dc"), outputs=("u_c",)),
    )

    def __init__(self) -> None:
        self.state = Empty()
        self.inp = _BridgeInp(q=0j, u_dc=0.0, i_c=0j)
        self.out = _BridgeOut(u_c=0j, i_dc=0.0)

    def set_dc_current(self, t: float) -> None:
        self.out.i_dc = 1.5 * (self.inp.q * self.inp.i_c.conjugate()).real

    def set_ac_voltage(self, t: float) -> None:
        self.out.u_c = self.inp.q * self.inp.u_dc

    def set_outputs(self, t: float) -> None:
        """Evaluate dc current and ac voltage from the current inputs."""
        self.set_dc_current(t)
        self.set_ac_voltage(t)

    def rhs(self, t: float):
        return ()

    def connections(self, dc_link, ac_branch) -> dict:
        """Return the connections of the bridge to ``dc_link`` and ``ac_branch``."""
        return {
            (ac_branch, "u1"): (self, "u_c"),
            (self, "i_c"): (ac_branch, "i"),
            (dc_link, "i_dc"): (self, "i_dc"),
            (self, "u_dc"): (dc_link, "u_dc"),
        }

    def zoh_connections(self, label: str) -> dict:
        """Return the connection of the held switching state ``label`` to ``q``."""
        return {(self, "q"): label}


class PWMBridge(Bridge, PWM):
    """Physical bridge with its PWM peripheral behind one scheduling interface.

    Exact switching and PWM-period averaging differ only in their modulator. Keeping the PWM
    inside the bridge gives every bridge model the same external power, held-input and scheduling
    interface.
    """

    state_prefix = "pwm"

    def __init__(self, period: float, load_period: float, offset: float, computation: float,
                 carrier_period: float, modulator: Modulator) -> None:
        Bridge.__init__(self)
        PWM.__init__(self, period, load_period, offset, computation, carrier_period, modulator)
        self.state_owner = self
        self.has_switching_events = not isinstance(modulator, ZOH)

    @property
    def event_periods(self) -> list[float]:
        """Periods which can change this bridge's sampled-data map."""
        periods = [self.period, self.load_period]
        if self.has_switching_events:
            periods.append(self.carrier_period)
        return periods

    def next_time(self) -> float:
        """Next controller, register-load or internal switching event."""
        return min(self.t_interrupt, self.t_load, self.next_switch)

    @property
    def next_control_time(self) -> float:
        return self.t_interrupt

    @property
    def control_index(self) -> int:
        return self.k

    def previous_control_time(self) -> float:
        return self.interrupt(self.k - 1)

    def control_count(self, t_a: float, t_b: float) -> int:
        return self.count(t_a, t_b)

    def control_due(self, t: float) -> bool:
        return abs(self.t_interrupt - t) < TIME_EPS

    def fast_edge(self, t: float) -> bool:
        """Whether the old bridge output interval ends at ``t``."""
        return self.edge(t)

    def actuate(self, t: float, command: tuple[float, Any] | None, enabled: bool) -> list[complex]:
        """Accept one sampled command and return bridge inputs which become active at ``t``.

        The bridge owns the computation/load ordering.  Unit deliberately does not know whether
        the implementation uses compare registers, a delayed averaged command or no carrier.
        """
        changes: list[complex] = []
        if command is not None and self.computation == 0.0:
            at, output = command
            self.write(at, output.d_abc, bool(output.gates) and enabled,
                       output.theta, output.omega)
        if abs(self.t_load - t) < TIME_EPS:
            changes.append(self.load(t))
        if command is not None and self.computation > 0.0:
            at, output = command
            self.write(at, output.d_abc, bool(output.gates) and enabled,
                       output.theta, output.omega)
        changes.extend(self.switches(t))
        return changes


class AveragingBridge(Bridge):
    """Ideal averaged bridge driven directly by the continuous controller output.

    There is no timer, command history, held modulation input or delay state: the controller and
    power stage are evaluated in the same Model RHS plan.
    """

    state_prefix = "bridge"

    def __init__(self) -> None:
        Bridge.__init__(self)
        self.modulator = None
        self.state_owner = self
        self.continuous = True
        self.enabled = False

    @property
    def on(self) -> bool:
        return self.enabled

    @property
    def event_periods(self) -> list[float]:
        return []

    @property
    def next_control_time(self) -> float:
        return math.inf

    @property
    def control_index(self) -> int:
        return 0

    def previous_control_time(self) -> float:
        return -math.inf

    def control_count(self, t_a: float, t_b: float) -> int:
        return 0

    def next_time(self) -> float:
        return math.inf

    def control_due(self, t: float) -> bool:
        return False

    def fast_edge(self, t: float) -> bool:
        return False

    def reset(self, d_abc: NDArray[np.float64], on: bool = False) -> None:
        self.enabled = bool(on)

    def start(self, t: float, sync: tuple[float | None, float | None], continued: bool) -> complex:
        return 0j

    def actuate(self, t: float, command: tuple[float, Any] | None,
                enabled: bool) -> list[complex]:
        self.enabled = bool(enabled)
        return []

    def zoh_connections(self, label: str) -> dict:
        return {}

    def describe(self) -> str:
        return "ideal averaging, continuous control"

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values:
            raise KeyError(f"continuous averaging bridge has no states, got {sorted(values)}")


def make_bridge(cfg: Any, sim: Any = None,
                modulator: Modulator | None = None) -> Bridge:
    """Build the selected bridge implementation behind the common bridge boundary."""
    pwm = cfg.pwm
    if cfg.bridge.model == "averaging":
        if modulator is not None:
            raise ValueError("an averaging bridge does not use a PWM modulator")
        return AveragingBridge()
    return PWMBridge(
        cfg.ctrl.period, pwm.load_period, pwm.grid_offset, cfg.ctrl.computation,
        pwm.switching_period,
        modulator if modulator is not None else make_modulator(pwm, cfg.base.f0, cfg.bridge),
    )

class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCInp(Bag):
    __slots__ = ("i_dc",)


class _DCOut(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C, self.R_esr = float(C), float(R_esr)
        self.state = _DCState(u_C=float(u0))

    def terminal_voltage(self, i: float) -> float:
        return self.state.u_C + self.R_esr * i

    def derivative(self, i: float) -> float:
        return i / self.C

    def dissipated_power(self, i: float) -> float:
        return self.R_esr * i ** 2


def _zero(t: float) -> float:
    return 0.0


class DCCurrentSource:
    """Supply current ``ramp(t) i_nom + k_dc (u_ref - u_C)`` (A); may be negative.

    ``k_dc`` is in A/V and ``u_ref`` in V. :meth:`connect` controls its nominal-current ramp.
    """

    def __init__(self, i_nom: float, k_dc: float = 0.0, u_ref: float = 0.0) -> None:
        self.i_nom, self.k_dc, self.u_ref = float(i_nom), float(k_dc), float(u_ref)
        self.ramp: Optional[Callable[[float], float]] = None

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect at ``t``, optionally with a smooth ramp, or disconnect the nominal current."""
        if not on:
            self.ramp = _zero
        elif ramp <= 0.0:
            self.ramp = None
        else:
            end = t + ramp
            self.ramp = lambda at: 1.0 if at >= end else smoothstep((at - t) / ramp)

    def __call__(self, t: float, u_C: float) -> float:
        i = self.i_nom if self.ramp is None else self.ramp(t) * self.i_nom
        if self.k_dc != 0.0:
            i += self.k_dc * (self.u_ref - u_C)
        return i


class DCVoltageSource:
    """Constant dc voltage source ``u_dc`` (V) with series resistance ``R`` (ohm)."""

    def __init__(self, u_dc: float, R: float = 0.0) -> None:
        self.u_dc, self.R = float(u_dc), float(R)

    def terminal_voltage(self, i: float) -> float:
        return self.u_dc - self.R * i

    def dissipated_power(self, i: float) -> float:
        return self.R * i ** 2


class DCLink:
    """DC link made of an optional capacitor and a current source, voltage source or no source.

    Without a capacitor a voltage source is required; a voltage source with a
    capacitor requires ``R + R_esr > 0``. Named state: ``u_C`` (V) if a capacitor is present.
    """

    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.u_dc", "inp.i_dc", 1.0, -1.0),)
    outputs_need_inputs: ClassVar[bool] = True  # i_src depends on i_dc
    output_stages: ClassVar[tuple[OutputStage, ...]] = (
        OutputStage("set_stored_voltage", inputs=(), outputs=("u_C",)),
        OutputStage("set_outputs", inputs=("i_dc",), outputs=("u_dc", "i_src")),
    )

    def __init__(self, capacitor: DCCapacitor | None = None,
                 source: DCCurrentSource | DCVoltageSource | None = None) -> None:
        if capacitor is not None and not isinstance(capacitor, DCCapacitor):
            raise TypeError("capacitor must be a DCCapacitor or None")
        if source is not None and not isinstance(source, (DCCurrentSource, DCVoltageSource)):
            raise TypeError("source must be a DCCurrentSource, DCVoltageSource or None")
        if capacitor is None and not isinstance(source, DCVoltageSource):
            raise ValueError("a DC link without a capacitor requires a voltage source")
        if capacitor is not None and isinstance(source, DCVoltageSource) and source.R + capacitor.R_esr <= 0:
            raise ValueError("a voltage source with a capacitor requires positive source resistance or ESR")
        self.capacitor, self.source = capacitor, source
        self.state = capacitor.state if capacitor is not None else Empty()
        self.state_names = ("u_C",) if capacitor is not None else ()
        self.storage = ((StoragePort("u_C", capacitor.C, 1.0, (("inp.i_dc", -1.0),)),)
                        if capacitor is not None else ())
        self.has_source = source is not None
        u0 = capacitor.state.u_C if capacitor is not None else source.u_dc
        self.inp = _DCInp(i_dc=0.0)
        self.out = _DCOut(u_dc=u0, u_C=u0, i_src=0.0)
        self.tripped = False
        # The circuit topology cannot change after assembly. Bind its equations once instead of
        # rediscovering the capacitor/source combination at every Runge--Kutta stage.
        if type(self) is DCLink:
            if capacitor is None:
                self.set_outputs = self._set_voltage_source_without_capacitor
                self.rhs = self._rhs_without_capacitor
            elif source is None:
                self.set_outputs = self._set_capacitor_without_source
                self.rhs = self._rhs_with_capacitor
            elif isinstance(source, DCVoltageSource):
                self.set_outputs = self._set_capacitor_with_voltage_source
                self.rhs = self._rhs_with_capacitor
            else:
                self.set_outputs = self._set_capacitor_with_current_source
                self.rhs = self._rhs_with_capacitor

    def open_breaker(self) -> None:
        """Disconnect the source from the capacitor; a link without capacitor is unchanged."""
        self.tripped = True

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the unit's source; current sources honor the connection ramp."""
        if isinstance(self.source, DCCurrentSource):
            self.source.connect(on, t, ramp)

    def retune_source(self, i: float, k: float, v: float, r: float) -> None:
        """Apply new settings to the existing DC source without replacing its state."""
        source = self.source
        if isinstance(source, DCCurrentSource):
            source.i_nom, source.k_dc = float(i), float(k)
        elif isinstance(source, DCVoltageSource):
            source.u_dc, source.R = float(v), float(r)

    def set_stored_voltage(self, t: float) -> None:
        """Expose the capacitor voltage (or stiff source emf) without a dc-current dependency."""
        self.out.u_C = (self.capacitor.state.u_C if self.capacitor is not None
                        else self.source.u_dc)

    def set_outputs(self, t: float) -> None:
        """Generic topology dispatch retained for subclasses."""
        cap, source, i_dc = self.capacitor, self.source, self.inp.i_dc
        if cap is None:
            self.out.u_dc = self.out.u_C = source.terminal_voltage(i_dc)
            self.out.i_src = i_dc
            return
        if self.tripped or source is None:
            i_src = 0.0
        elif isinstance(source, DCVoltageSource):
            # from u_C + R_esr (i_src - i_dc) = u_dc - R i_src
            i_src = (source.u_dc - cap.state.u_C + cap.R_esr * i_dc) / (source.R + cap.R_esr)
        else:
            i_src = source(t, cap.state.u_C)
        self.out.i_src = i_src
        self.out.u_C = cap.state.u_C
        self.out.u_dc = cap.terminal_voltage(i_src - i_dc)

    def _set_voltage_source_without_capacitor(self, t: float) -> None:
        source, i_dc, out = self.source, self.inp.i_dc, self.out
        out.u_dc = out.u_C = source.u_dc - source.R * i_dc
        out.i_src = i_dc

    def _set_capacitor_without_source(self, t: float) -> None:
        cap, i_dc, out = self.capacitor, self.inp.i_dc, self.out
        out.i_src = 0.0
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C - cap.R_esr * i_dc

    def _set_capacitor_with_voltage_source(self, t: float) -> None:
        cap, source, i_dc, out = self.capacitor, self.source, self.inp.i_dc, self.out
        i_src = (0.0 if self.tripped else
                 (source.u_dc - cap.state.u_C + cap.R_esr * i_dc) /
                 (source.R + cap.R_esr))
        out.i_src = i_src
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C + cap.R_esr * (i_src - i_dc)

    def _set_capacitor_with_current_source(self, t: float) -> None:
        cap, source, i_dc, out = self.capacitor, self.source, self.inp.i_dc, self.out
        i_src = 0.0 if self.tripped else source(t, cap.state.u_C)
        out.i_src = i_src
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C + cap.R_esr * (i_src - i_dc)

    def rhs(self, t: float):
        if self.capacitor is None:
            return ()
        return (self.capacitor.derivative(self.out.i_src - self.inp.i_dc),)

    def _rhs_without_capacitor(self, t: float):
        return ()

    def _rhs_with_capacitor(self, t: float):
        return ((self.out.i_src - self.inp.i_dc) / self.capacitor.C,)

    def dissipated_power(self) -> float:
        loss = (self.capacitor.dissipated_power(self.out.i_src - self.inp.i_dc)
                if self.capacitor is not None else 0.0)
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            loss += self.source.dissipated_power(i_src)
        return loss

    def supplied_power(self) -> float:
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            return self.source.u_dc * i_src
        return self.out.u_dc * self.out.i_src


def make_dclink(cfg: Any) -> DCLink:
    """Build a :class:`DCLink` from SI parameters; source type ``"current"``, ``"voltage"`` or ``"none"``."""
    capacitor = None
    if cfg.capacitor is not None:
        capacitor = DCCapacitor(cfg.vdc_ref, cfg.capacitor.c, cfg.capacitor.r_esr)
    p = cfg.source
    if p.type == "current":
        source = DCCurrentSource(p.i, p.k, cfg.vdc_ref)
    elif p.type == "voltage":
        source = DCVoltageSource(p.v, p.r)
    elif p.type == "none":
        source = None
    else:
        raise ValueError(f"unknown DC source type {p.type!r}")
    return DCLink(capacitor, source)
