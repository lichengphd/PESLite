"""The converter's power stage: bridge models and the dc link.

Quantities are SI. Every bridge keeps the same AC/DC and held-modulation connections; a concrete
bridge also owns the timing which turns sampled controller commands into its modulation value.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, ClassVar, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex, smoothstep
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
            (ac_branch, "u_from"): (self, "u_c"),
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
    """Delayed ideal controlled voltage-source bridge with no PWM peripheral.

    A controller word sampled at ``t`` drives the source from the first equivalent load instant
    not earlier than ``t + computation``. Duty ratios and the enable travel together through the
    delay history so continuation restores every command still in flight.
    """

    state_prefix = "bridge"

    def __init__(self, period: float, load_period: float, offset: float,
                 computation: float) -> None:
        super().__init__()
        self.period = float(period)
        self.load_period = float(load_period)
        self.offset = float(offset)
        self.computation = float(computation)
        if not math.isfinite(self.period) or self.period <= 0.0:
            raise ValueError(f"the control period must be finite and positive, got {period}")
        if not math.isfinite(self.load_period) or self.load_period <= 0.0:
            raise ValueError(
                f"the equivalent update period must be finite and positive, got {load_period}"
            )
        if not math.isfinite(self.offset):
            raise ValueError(f"the timer offset must be finite, got {offset}")
        if not math.isfinite(self.computation) or self.computation < 0.0:
            raise ValueError(
                f"the computation time must be finite and nonnegative, got {computation}"
            )
        pending = [self.apply_time(self.interrupt(k)) - self.interrupt(k) for k in range(4)]
        self.history_length = max(1, int(math.ceil(max(pending) / self.period - 1e-12)))
        zero = np.zeros(4)
        self.active = zero.copy()
        self.history = [zero.copy() for _ in range(self.history_length)]
        self.k = 0
        self.t_interrupt = self.interrupt(0)
        self._queue: deque[tuple[float, int, NDArray[np.float64]]] = deque()
        self.t_load = self.load_end = math.inf
        self.modulator = None
        self.state_owner = self

    @staticmethod
    def _word(d_abc: NDArray[np.float64], on: bool) -> NDArray[np.float64]:
        word = np.empty(4, dtype=float)
        word[:3] = np.asarray(d_abc, dtype=float).ravel()
        word[3] = 1.0 if on else 0.0
        return word

    @property
    def on(self) -> bool:
        return bool(self.active[3])

    @property
    def duty(self) -> NDArray[np.float64]:
        return self.active[:3].copy()

    @property
    def event_periods(self) -> list[float]:
        return [self.period, self.load_period]

    def interrupt(self, k: int) -> float:
        return self.offset + k * self.period

    def after(self, t: float) -> int:
        """Index of the first control interrupt strictly after ``t``."""
        return int(math.floor((t - self.offset + TIME_EPS) / self.period)) + 1

    def control_count(self, t_a: float, t_b: float) -> int:
        first = math.ceil((t_a - self.offset - TIME_EPS) / self.period)
        return max(0, self.after(t_b) - first)

    def _load_at_or_after(self, t: float) -> float:
        j = int(math.ceil((t - self.offset - TIME_EPS) / self.load_period))
        return self.offset + j * self.load_period

    def apply_time(self, t: float) -> float:
        return self._load_at_or_after(t + self.computation)

    def delay(self, t: float) -> float:
        return self.apply_time(t) + 0.5 * self.load_period - t

    def describe(self) -> str:
        delays = [self.delay(self.interrupt(k)) for k in range(8)]
        lo, hi = min(delays), max(delays)
        if abs(hi - lo) <= TIME_EPS:
            return f"ideal averaging, delay {lo * 1e6:g} us"
        return f"ideal averaging, delay {lo * 1e6:g}..{hi * 1e6:g} us"

    @property
    def next_control_time(self) -> float:
        return self.t_interrupt

    @property
    def control_index(self) -> int:
        return self.k

    def previous_control_time(self) -> float:
        return self.interrupt(self.k - 1)

    def next_time(self) -> float:
        return min(self.t_interrupt, self.t_load)

    def control_due(self, t: float) -> bool:
        return abs(self.t_interrupt - t) < TIME_EPS

    def fast_edge(self, t: float) -> bool:
        due = abs(self.t_load - t) < TIME_EPS
        # With zero computation time the command which creates an immediate load does not exist
        # until sense() has run. Protection runs first, so derive that edge from the timer grid.
        immediate = (self.control_due(t)
                     and abs(self.apply_time(t) - t) < TIME_EPS)
        return due or immediate

    def reset(self, d_abc: NDArray[np.float64], on: bool = False) -> None:
        """Fill the output and delay history with one initial command word."""
        word = self._word(d_abc, on)
        self.active = word.copy()
        self.history = [word.copy() for _ in range(self.history_length)]
        self._queue.clear()
        self.t_load = self.load_end = math.inf

    def start(self, t: float, sync: tuple[float | None, float | None],
              continued: bool) -> complex:
        """Resume the command grid and rebuild delayed commands from saved history."""
        self.k = self.after(t)
        self.t_interrupt = self.interrupt(self.k)
        pending = []
        if continued:
            for age, word in enumerate(self.history):
                index = self.k - 1 - age
                due = self.apply_time(self.interrupt(index))
                if due > t + TIME_EPS:
                    pending.append((due, index, word))
        self._queue = deque()
        for due, index, word in sorted(pending, key=lambda item: item[:2]):
            self._enqueue(due, index, word)
        self.t_load = self.load_end = self._queue[0][0] if self._queue else math.inf
        return abc2complex(self.active[:3])

    def _enqueue(self, due: float, index: int, word: NDArray[np.float64]) -> None:
        """Keep only the last command when several interrupts feed one load instant."""
        item = (due, index, word)
        if self._queue and abs(self._queue[-1][0] - due) < TIME_EPS:
            self._queue[-1] = item
        else:
            self._queue.append(item)

    def write(self, t: float, d_abc: NDArray[np.float64], on: bool = True,
              theta: float | None = None, omega: float | None = None) -> None:
        """Snapshot and schedule one delayed controller command."""
        word = self._word(d_abc, on)
        if self.history_length == 1:
            self.history[0] = word
        else:
            self.history.insert(0, word)
            self.history.pop()
        self._enqueue(self.apply_time(t), self.k, word)
        self.k += 1
        self.t_interrupt = self.interrupt(self.k)
        self.t_load = self.load_end = self._queue[0][0]

    def load(self, t: float) -> complex:
        """Apply every delayed command due at ``t``."""
        while self._queue and self._queue[0][0] <= t + TIME_EPS:
            self.active = self._queue.popleft()[2]
        self.t_load = self.load_end = self._queue[0][0] if self._queue else math.inf
        return abc2complex(self.active[:3])

    def actuate(self, t: float, command: tuple[float, Any] | None,
                enabled: bool) -> list[complex]:
        if command is not None:
            at, output = command
            self.write(at, output.d_abc, bool(output.gates) and enabled,
                       output.theta, output.omega)
        return [self.load(t)] if abs(self.t_load - t) < TIME_EPS else []

    def get_state(self) -> dict[str, Any]:
        names = ("d_a", "d_b", "d_c", "on")
        out = {name: (bool(self.active[k]) if name == "on" else float(self.active[k]))
               for k, name in enumerate(names)}
        for age, word in enumerate(self.history):
            out.update({f"history.{age}.{name}":
                        (bool(word[k]) if name == "on" else float(word[k]))
                        for k, name in enumerate(names)})
        return out

    def set_state(self, values: Mapping[str, Any]) -> None:
        known = set(self.get_state())
        unknown = set(values) - known
        if unknown:
            raise KeyError(
                f"averaging bridge: no state(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        names = ("d_a", "d_b", "d_c", "on")
        for key, value in values.items():
            parts = key.split(".")
            target = self.active if len(parts) == 1 else self.history[int(parts[1])]
            name = parts[-1]
            target[names.index(name)] = (1.0 if value else 0.0) if name == "on" else float(value)


def make_bridge(cfg: Any, sim: Any = None,
                modulator: Modulator | None = None) -> Bridge:
    """Build the selected bridge implementation behind the common bridge boundary."""
    pwm = cfg.pwm
    if cfg.bridge.model == "averaging":
        if modulator is not None:
            raise ValueError("an averaging bridge does not use a PWM modulator")
        return AveragingBridge(
            cfg.ctrl.period, pwm.load_period, pwm.grid_offset, cfg.ctrl.computation)
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
