"""The controller of a converter unit as one block: its interface, and :class:`UniteType`.

At each PWM publication the controller takes an ADC sample (:class:`Measurement`, SI), scales it to
the unit's pu bases, runs its loop network (:class:`ControlGraph`: the loops of :mod:`.loops`
connected by typed ports) and its protection (:mod:`.protection`), and turns the voltage command
into duty ratios (:mod:`.modulation`); it returns them in a :class:`ControlOutput`. ``control.type``
(``gfl``, ``gfm``) chooses how its loops are wired by default.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..solver.model import ConfigError, gather, scatter
from .blocks import peak_abs
from .loops import (ANGLE, CURRENT, DC_VOLTAGE, FREQUENCY, I_AB, LOOP_TYPES, POWER_PU, V_AB, V_DQ, VOLTAGE)
from .modulation import CONFIGURED, OutputStage
from .protection import Protection

__all__ = ["Measurement", "ControlMeasurement", "ControlOutput", "Controller", "ControlGraph", "default_wiring",
           "UniteType",
           "make_controller"]


# ------------------------------------------------------------------ the interface

@dataclass
class Measurement:
    """SI sample at ``t``: peak-scaled alpha-beta AC vectors (V, A) and DC voltage (V).

    Windowed channels hold means over ``(t - T_avg, t]``; ``*_raw`` and ``i_abc`` are instantaneous.
    ``samples`` holds an oversampled update's samples, oldest first, this one last; otherwise empty.
    """

    t: float
    u_g: complex  # PCC voltage (V)
    i_c: complex  # converter current (A), positive out of the bridge
    u_dc: float  # dc-link voltage (V)
    i_abc: NDArray[np.float64]  # instantaneous phase currents (A)
    u_g_raw: Optional[complex] = None  # before window averaging
    i_c_raw: Optional[complex] = None
    u_dc_raw: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)
    samples: tuple["Measurement", ...] = ()


@dataclass
class ControlMeasurement:
    """Controller-side sample in pu: alpha-beta AC vectors on peak phase bases, DC voltage on ``vdc_ref``.

    ``t`` is in s.
    """

    t: float
    u_g: complex
    i_c: complex
    u_dc: float
    i_abc: NDArray[np.float64]


@dataclass
class ControlOutput:
    """Result of one controller call.

    ``theta`` (rad) and ``omega`` (rad/s): the controller's synchronization angle and frequency,
    required by synchronous modulators; ``None`` allows only asynchronous modulation.
    """

    d_abc: NDArray[np.float64]  # duty ratios of phases a, b, c in [0, 1]
    tripped: bool = False
    log: dict[str, float] | None = None
    theta: float | None = None  # synchronization angle (rad)
    omega: float | None = None  # synchronization frequency (rad/s)


@runtime_checkable
class Controller(Protocol):
    """Sampled controller ``(t, Measurement) -> ControlOutput``, called once per PWM publication.

    Optional: ``initial_sync() -> (theta, omega)`` seeds a synchronous carrier for the first period;
    ``next_event``, ``periods``, ``reset_clocks(t)`` and ``update(t, meas)`` for independently clocked
    loops; ``fast_check(t, i_abc)`` for an over-current check between samples; ``align_startup``.
    """

    T_s: float  # PWM publication interval (s), pwm.update_period

    def __call__(self, t: float, meas: Measurement) -> ControlOutput: ...

    def initial_duty(self) -> NDArray[np.float64]:
        """Duty ratios applied before the first control update."""


# ------------------------------------------------------------------ the loop network

# kinds of input source
_HELD, _MEASURED, _CONSTANT = 0, 1, 2
_MEASUREMENTS = {"measurement.u_g": "u_g", "measurement.i_c": "i_c", "measurement.u_dc": "u_dc"}
_REFERENCES = {"id_ref_pu": CURRENT, "iq_ref_pu": CURRENT, "theta": ANGLE, "omega": FREQUENCY,
               "p_ref_pu": POWER_PU, "q_ref_pu": POWER_PU, "v_ref_pu": VOLTAGE,
               "vdc_ref_pu": DC_VOLTAGE, "zero_v_pu": V_DQ}


class ControlGraph:
    """The loops of ``cfg.control.loops`` wired by typed ports, each run on its own clock.

    Loop inputs are held outputs of other loops, the pu measurement (``measurement.u_g``, ``.i_c``,
    ``.u_dc``) or the references (``references.<name>``); ``connections`` and ``outputs`` are the
    default wiring, which ``cfg.control.connections`` and ``.outputs`` override. Loops run in signal
    order; a loop without a period runs whenever the loops before it do.

    Retuned parameters are queued by :meth:`schedule` and take effect at the next update. Changed
    loops are rebuilt from their new parameters and continue from their named states.
    """

    def __init__(self, cfg, scenario, connections, outputs):
        self.cfg, self.scenario = cfg, scenario
        self._pending: list[Any] = []
        self.on_retune = None
        self.connections = {**connections, **cfg.control.connections}
        self.outputs = {**outputs, **cfg.control.outputs}
        self.nodes = {}
        self.periods = {}
        self.ticks = {}
        self.values: dict[str, Any] = {}
        self._out_specs: dict[str, tuple] = {}
        self.updated = set()
        refs = cfg.control.references
        self.references = {f.name: getattr(refs, f.name) for f in fields(refs)}
        types = {"measurement.u_g": V_AB, "measurement.i_c": I_AB, "measurement.u_dc": DC_VOLTAGE,
                 **{f"references.{k}": v for k, v in _REFERENCES.items()}}
        for name, loop in cfg.control.loops.items():
            cls = LOOP_TYPES.get(loop.type)
            if cls is None:
                raise ConfigError(f"control.loops.{name}: no loop type {loop.type!r} is registered")
            self.nodes[name] = node = cls(loop, cfg, scenario)
            self._out_specs[name] = tuple((port, f"{name}.{port}", spec.complex_value)
                                          for port, spec in node.outputs.items())
            self.periods[name] = loop.period
            self.ticks[name] = 1
            self._store(name, node.initial_outputs())
            types.update({f"{name}.{port}": spec for port, spec in node.outputs.items()})
        dependencies = {name: set() for name in self.nodes}
        targets = set()
        for name, node in self.nodes.items():
            for port, spec in node.inputs.items():
                target = f"{name}.{port}"
                targets.add(target)
                source = self.connections.get(target)
                if source not in types:
                    raise ConfigError(f"control.connections.{target}: missing or unknown source {source!r}")
                if types[source] != spec:
                    raise ConfigError(f"control.connections.{target}: incompatible signal {source!r}: "
                                      f"{types[source]} -> {spec}")
                owner = source.partition(".")[0]
                if owner in self.nodes and port not in node.delayed:
                    dependencies[name].add(owner)
        if set(self.connections) - targets:
            raise ConfigError(f"control.connections: unknown input ports {sorted(set(self.connections) - targets)}")
        for port, spec in {"u_dq": V_DQ, "theta": ANGLE, "omega": FREQUENCY}.items():
            source = self.outputs.get(port)
            if source not in types or types[source] != spec:
                raise ConfigError(f"control.outputs.{port}: missing or incompatible source {source!r}")
        if set(self.outputs) - {"u_dq", "theta", "omega"}:
            raise ConfigError("control.outputs: only u_dq, theta and omega are supported")
        self.order = []
        pending = dict(dependencies)
        while pending:
            ready = [n for n, deps in pending.items() if not deps]
            if not ready:
                raise ConfigError("control.connections: instantaneous cycle; insert an explicit unit_delay")
            for name in ready:
                self.order.append(name)
                pending.pop(name)
            for deps in pending.values():
                deps.difference_update(ready)
        self._rewire()
        self._all_clocked = all(T is not None for T in self.periods.values())
        self._clocked = tuple((n, T) for n, T in self.periods.items() if T)
        self._retime()

    def _rewire(self):
        """Resolve loop inputs and graph outputs against the current references."""
        self._constants = {f"references.{k}": complex(v) if k == "zero_v_pu" else v
                           for k, v in self.references.items()}
        self._wiring = {}
        self._immediate = {}
        self._port_source = {}
        for name, node in self.nodes.items():
            wiring = tuple((port,) + self._resolve(self.connections[f"{name}.{port}"]) for port in node.inputs)
            self._wiring[name] = wiring
            self._immediate[name] = tuple(w for w in wiring if w[0] not in node.delayed)
            self._port_source[name] = {w[0]: w[1:] for w in wiring}
        self._latching = frozenset(n for n, node in self.nodes.items() if node.delayed)
        self._output_source = {port: self._resolve(source) for port, source in self.outputs.items()}

    def schedule(self, cfg):
        """Queue unit parameters to take effect at the next control update."""
        self._pending.append(cfg)

    def _apply_pending(self):
        """Apply queued references and rebuild loops whose parameters changed."""
        while self._pending:
            cfg = self._pending.pop(0)
            control = cfg.control
            if control.references != self.cfg.control.references:
                self.references = {f.name: getattr(control.references, f.name)
                                   for f in fields(control.references)}
                self._rewire()
            changed = [name for name, params in control.loops.items()
                       if params != self.cfg.control.loops[name]]
            self.cfg = cfg
            for name in changed:
                self._retune(name)

    def _retune(self, name):
        """Rebuild one loop and continue from the old instance's state."""
        old = self.nodes[name]
        params = self.cfg.control.loops[name]
        fresh = LOOP_TYPES[params.type](params, self.cfg, self.scenario)
        try:
            fresh.set_state(old.get_state())
        except (KeyError, ValueError) as exc:
            raise ConfigError(
                f"control.loops.{name}: with the parameters of a set event it has other "
                f"states than before ({exc}), so it cannot continue from them") from None
        fresh.retuned(old)
        self.nodes[name] = fresh
        if self.on_retune is not None:
            self.on_retune(name)

    def _resolve(self, source):
        if source in _MEASUREMENTS:
            return _MEASURED, _MEASUREMENTS[source]
        if source in self._constants:
            return _CONSTANT, self._constants[source]
        return _HELD, source

    def _read(self, kind, key, meas):
        if kind == _HELD:
            return self.values[key]
        if kind == _MEASURED:
            return getattr(meas, key)
        return key

    def _gather(self, wiring, meas):
        values = self.values  # _read, inlined: this runs at every loop update
        out = {}
        for port, kind, key in wiring:
            if kind == _HELD:
                out[port] = values[key]
            elif kind == _MEASURED:
                out[port] = getattr(meas, key)
            else:
                out[port] = key
        return out

    def _store(self, name, outputs):
        specs = self._out_specs[name]
        if len(outputs) != len(specs) or any(port not in outputs for port, _key, _c in specs):
            raise ValueError(f"control loop {name}: outputs must be {sorted(self.nodes[name].outputs)}")
        values = self.values
        isfinite = math.isfinite
        for port, key, complex_value in specs:
            value = outputs[port]
            if complex_value:
                value = complex(value)
                finite = isfinite(value.real) and isfinite(value.imag)
            else:
                value = float(value)
                finite = isfinite(value)
            if not finite:
                raise FloatingPointError(f"control loop {name}.{port} returned a non-finite value")
            values[key] = value

    # ------------------------------------------------------------ clocks
    def reset_clocks(self, t):
        self.ticks = {name: int(math.floor(t / T + 1e-9)) + 1 if T else 1
                      for name, T in self.periods.items()}
        self._retime()

    def _retime(self):
        """Recompute :attr:`next_event` from the clocks."""
        ticks = self.ticks
        self._next_event = min([ticks[n] * T for n, T in self._clocked], default=math.inf)

    @property
    def next_event(self):
        """Earliest time (s) at which a clocked loop is due."""
        return self._next_event

    # ------------------------------------------------------------ running
    def input(self, name, port, meas):
        """Return the input ``port`` of the loop ``name`` for the pu measurement ``meas``."""
        return self._read(*self._port_source[name][port], meas)

    def update(self, t, meas, finalize=None):
        """Run the loops due at ``t`` on the pu measurement ``meas``; return their names.

        ``finalize(t, meas, updated)`` runs after them, before the delayed inputs are latched.
        """
        if self._pending:
            self._apply_pending()
        self.updated = updated = set()
        periods, ticks = self.periods, self.ticks
        due = {n for n, T in self._clocked if ticks[n] * T <= t + 1e-10}
        if not due and self._all_clocked:
            return updated
        nodes, gather_, immediate = self.nodes, self._gather, self._immediate
        for name in self.order:
            T = periods[name]
            if name not in due and T is not None:
                continue
            self._store(name, nodes[name].update(t, gather_(immediate[name], meas)))
            updated.add(name)
            if T:
                ticks[name] += 1
        if due:
            self._retime()
        if finalize is not None:
            finalize(t, meas, updated)
        for name in due:
            if name in self._latching:
                nodes[name].latch(gather_(self._wiring[name], meas))
        return updated

    def output(self, port, meas):
        """Return the graph output ``port`` (``u_dq``, ``theta`` or ``omega``)."""
        return self._read(*self._output_source[port], meas)

    def describe(self):
        return {"loops": {n: {"type": self.cfg.control.loops[n].type, "period": self.periods[n]}
                          for n in self.order}, "connections": dict(self.connections), "outputs": dict(self.outputs)}

    # ------------------------------------------------------------ states
    def get_state(self):
        state = gather(self.nodes)
        state.update({f"held.{self._state_port(k)}": v for k, v in self.values.items()})
        state.update({f"clock.{n}": k for n, k in self.ticks.items() if self.periods[n]})
        return state

    def set_state(self, values):
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"unknown control states {sorted(unknown)}; electrical held outputs and converted states use _pu")
        state_ports = {self._state_port(k): k for k in self.values}
        rest = {}
        for key, value in values.items():
            if key.startswith("held."):
                self.values[state_ports[key[5:]]] = value
            elif key.startswith("clock."):
                name = key[6:]
                if int(value) != value or value < 1:
                    raise ValueError(f"invalid control clock {key}: {value}")
                self.ticks[name] = int(value)
            else:
                rest[key] = value
        self._retime()
        scatter(self.nodes, rest)
        # held outputs that follow the loaded states, unless given themselves
        for name, node in self.nodes.items():
            if node.outputs_from_state:
                for port, value in node.initial_outputs().items():
                    if f"held.{self._state_port(name + '.' + port)}" not in values:
                        self.values[f"{name}.{port}"] = value

    def _state_port(self, key):
        name, port = key.split(".", 1)
        return key + "_pu" if self.nodes[name].outputs[port].unit.startswith("pu_") else key


# ------------------------------------------------------------------ the default wiring

def default_wiring(cfg, family):
    """Return the default ``(connections, outputs)`` for family ``"gfl"``, ``"gfm"`` or ``"custom"``.

    Each loop is wired by its role (:attr:`peslite.control.loops.Loop.role`), so a registered loop
    type takes the place of the built-in one of its role.
    """
    if family == "custom":
        return {}, {}
    loops = cfg.control.loops
    roles = {name: getattr(LOOP_TYPES.get(p.type), "role", None) for name, p in loops.items()}

    def find(role):
        found = [n for n, r in roles.items() if r == role]
        if len(found) > 1:
            raise ConfigError(f"control.loops: ambiguous {family} role {found}; use control.type = 'custom' "
                              "and explicit connections/outputs for multiple instances of this role")
        return found[0] if len(found) == 1 else None
    pll, sync = find("pll"), find("sync")
    angle = pll if family == "gfl" else sync
    frame = f"{angle}.frame" if angle else "references.theta"
    theta = f"{angle}.theta" if angle else "references.theta"
    omega = f"{angle}.omega" if angle else "references.omega"
    dc, cc, power = find("dc_voltage"), find("current"), find("power")
    va, vi, damping = find("admittance"), find("impedance"), find("damping")
    extra = f"{damping}.extra" if damping else "references.zero_v_pu"
    ports_of = {
        "pll": {"v": "measurement.u_g"},
        "dc_voltage": {"u_dc": "measurement.u_dc", "vdc_ref": "references.vdc_ref_pu"},
        "current": {"v": "measurement.u_g", "i": "measurement.i_c", "frame": frame, "omega": omega,
                    "extra": extra, "id_ref": f"{va or dc}.id_ref" if va or dc else "references.id_ref_pu",
                    "iq_ref": f"{va}.iq_ref" if va else "references.iq_ref_pu"},
        "power": {"v": "measurement.u_g", "i": "measurement.i_c"},
        "sync": {"p": f"{power}.p", "q": f"{power}.q", "v": "measurement.u_g", "i": "measurement.i_c",
                 "u_dc": "measurement.u_dc", "p_ref": "references.p_ref_pu", "q_ref": "references.q_ref_pu",
                 "v_ref": "references.v_ref_pu"},
        "impedance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "i": "measurement.i_c",
                      "extra": extra},
        "admittance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "v": "measurement.u_g"},
        "damping": {"i": "measurement.i_c", "frame": frame},
    }
    wires = {f"{name}.{port}": source for name, role in roles.items()
             for port, source in ports_of.get(role, {}).items()
             if port in LOOP_TYPES[loops[name].type].inputs}
    terminal = vi if family == "gfm" and vi else cc
    return wires, {"u_dq": f"{terminal}.u_dq", "theta": theta, "omega": omega}


# ------------------------------------------------------------------ the controller

class UniteType:
    """Configurable controller of one unit: its loop network, protection and output stage.

    ``cfg``: the unit's parameters; ``cfg.control.type``: ``"gfl"``, ``"gfm"`` or ``"custom"`` (no
    default wiring). ``scenario``: the unit's connection state and ramp over time.
    ``pwm_method``, ``limiter``: see :class:`~peslite.control.modulation.OutputStage`.
    ``update()`` advances the due loops; ``__call__`` is the PWM publication. Integrating loops
    freeze while the unit is disconnected or tripped.
    """

    def __init__(self, cfg, scenario, pwm_method=None, *, limiter=CONFIGURED):
        self.p = cfg
        self.scenario = scenario
        wires, outputs = default_wiring(cfg, cfg.control.type)
        self.graph = graph = ControlGraph(cfg, scenario, wires, outputs)
        self.periods = graph.periods
        self.T_s = cfg.pwm.update_period
        self.v_base, self.i_base, self.w0 = cfg.base.v_phase_peak, cfg.base.i_phase_peak, cfg.base.w0
        self.v_dc_base = cfg.dc_base.v
        self.protection = Protection(cfg.protection, self.T_s, scenario.armed)
        self.stage = OutputStage(cfg, complex(cfg.base.v_phase_peak), pwm_method=pwm_method, limiter=limiter)
        self.theta, self.omega = self.initial_sync()
        self.u_cmd = 1.0 + 0j
        self.command_theta = self.theta
        self.v_dq = self.i_dq = 0j
        self.last_log = {}
        # the loops the controller itself reads, by role
        role = {name: node.role for name, node in graph.nodes.items()}
        def first(r, port=None):
            return next((name for name, value in role.items()
                         if value == r and (port is None or port in graph.nodes[name].inputs)), None)

        self._terminal = graph.outputs["u_dq"].partition(".")[0]
        self._terminal_key = f"{self._terminal}.u_dq"
        self._terminal_in_graph = self._terminal in graph.nodes
        self._terminal_is_cc = self._terminal_in_graph and role[self._terminal] == "current"
        self._find_nodes()
        graph.on_retune = self._find_nodes
        self._frame_key = f"{graph.outputs['theta'].partition('.')[0]}.frame"
        self._dc, self._sync = first("dc_voltage"), first("sync")
        self._frame_cc, self._log_cc = first("current", "frame"), first("current", "id_ref")
        self._is_gfl = cfg.control.type == "gfl"
        self._p_key, self._q_key, self._v_ref_key = f"{first('power')}.p", f"{first('power')}.q", f"{self._sync}.v_ref"

    def _find_nodes(self, name=None):
        """Refresh loop-instance references after a loop is rebuilt."""
        self._freezable = tuple(node for node in self.graph.nodes.values()
                               if hasattr(node, "frozen"))
        self._terminal_node = self.graph.nodes.get(self._terminal)

    def describe(self):
        return self.graph.describe()

    def retune(self, cfg, paths, t):
        """Queue new control parameters and apply new protection settings."""
        if any(path.startswith("control.") for path in paths):
            self.graph.schedule(cfg)
        if any(path.startswith("protection.") for path in paths):
            self.protection.retune(cfg.protection)
        self.p = cfg

    @property
    def next_event(self):
        return self.graph.next_event

    def reset_clocks(self, t):
        self.graph.reset_clocks(t)

    def initial_sync(self):
        values = {**self.graph.values, **{f"references.{k}": v for k, v in self.graph.references.items()}}
        return float(values[self.graph.outputs["theta"]]), float(values[self.graph.outputs["omega"]])

    def initial_duty(self):
        return self.stage.initial_duty()

    def align_startup(self, u_ab, u_dc):
        """Start with the modulation that reproduces the terminal voltage ``u_ab`` (V) from ``u_dc`` (V)."""
        self.stage.align_startup(u_ab, u_dc)
        self.theta, self.omega = self.initial_sync()
        self.command_theta = self.theta
        self.u_cmd = u_ab / self.v_base * cmath.exp(-1j * self.command_theta)

    @property
    def tripped(self):
        return self.protection.tripped

    def fast_check(self, t, i_abc):
        """Check the instantaneous over-current criterion between samples; ``i_abc`` in A."""
        return self.protection.check_current(t, peak_abs(i_abc) / self.i_base)

    # ------------------------------------------------------------ one sample
    def _pu(self, meas: Measurement) -> ControlMeasurement:
        """The SI sample in the unit's pu bases."""
        if not isinstance(meas, Measurement):
            raise TypeError("the controller takes an SI Measurement, not an already normalized sample")
        return ControlMeasurement(meas.t, meas.u_g / self.v_base, meas.i_c / self.i_base,
                                  meas.u_dc / self.v_dc_base, meas.i_abc / self.i_base)

    def _raw_log(self, meas: Measurement, theta: float) -> dict[str, float]:
        """The pre-average sample in the ``theta`` frame as ``*_raw_pu`` log entries (empty if not averaged)."""
        out: dict[str, float] = {}
        if meas.u_g_raw is not None and (meas.u_g_raw != meas.u_g or meas.i_c_raw != meas.i_c):
            rot = cmath.exp(-1j * theta)
            v, i = meas.u_g_raw * rot, meas.i_c_raw * rot
            out.update({"vd_raw_pu": v.real / self.v_base, "vq_raw_pu": v.imag / self.v_base,
                        "id_raw_pu": i.real / self.i_base, "iq_raw_pu": i.imag / self.i_base})
        if meas.u_dc_raw is not None and meas.u_dc_raw != meas.u_dc:
            out["vdc_raw_pu"] = meas.u_dc_raw / self.v_dc_base
        return out

    def update(self, t, meas):
        """Run the loops due at ``t`` between publications, on the SI sample ``meas``."""
        return self._update_pu(t, self._pu(meas))

    def _in_service(self, t):
        return not self.tripped and self.scenario.connected(t)

    def _update_pu(self, t, control_meas):
        self.stage.new_instant()
        idle = not self._in_service(t)
        for node in self._freezable:
            node.frozen = idle
        return self.graph.update(t, control_meas, finalize=self._accept_command)

    def _accept_command(self, t, meas, updated):
        """Update the held angle, frequency and voltage command, applying anti-windup feedback."""
        graph = self.graph
        self.theta = float(graph.output("theta", meas))
        self.omega = float(graph.output("omega", meas))
        if self._terminal in updated or (updated and not self._terminal_in_graph):
            self.u_cmd = complex(graph.output("u_dq", meas))
            self.command_theta = self.theta
            cc = self._terminal_node if self._terminal_is_cc else None
            extra = cc.extra if cc is not None else 0j
            self.stage.modulate(t, self.u_cmd - extra, self.command_theta, meas.u_dc,
                                cc=cc, extra_dq=extra, count=False)
            if cc is not None and cc.antiwindup == "conditional":
                self.u_cmd = cc.command(*cc._last) + cc.extra
                graph.values[self._terminal_key] = self.u_cmd

    def __call__(self, t, meas):
        control_meas = self._pu(meas)
        self._update_pu(t, control_meas)
        frame = self.graph.values.get(self._frame_key, self.theta)
        if self._frame_cc is not None:
            frame = self.graph.input(self._frame_cc, "frame", control_meas)
        rot = cmath.exp(-1j * frame)
        self.v_dq, self.i_dq = control_meas.u_g * rot, control_meas.i_c * rot
        freq_dev = (self.omega - self.w0) / (2 * math.pi)
        refs, prot = self.graph.references, self.protection
        prot.check_current(t, peak_abs(control_meas.i_abc))
        prot.check_sampled(t, abs(self.v_dq), freq_dev,
                           control_meas.u_dc - refs["vdc_ref_pu"])
        tripped = prot.tripped
        duty = self.stage.modulate(t, 0j if tripped else self.u_cmd, self.command_theta, control_meas.u_dc)
        log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
               "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
               "vac_pu": abs(self.v_dq), "vdc_pu": control_meas.u_dc,
               "m_max": peak_abs(self.stage.m_abc), "in_service": 1.0 if self._in_service(t) else 0.0,
               **self._raw_log(meas, frame)}
        angle_rel = (self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi
        if self._is_gfl:
            id_ref = self.graph.input(self._log_cc, "id_ref", control_meas) if self._log_cc else 0.0
            log.update(id_ref_pu=id_ref, freq_dev=freq_dev, angle_rel=angle_rel)
        else:
            values = self.graph.values
            pr, qr, vr = self.scenario.setpoints(t, refs["p_ref_pu"], refs["q_ref_pu"], refs["v_ref_pu"])
            log.update(p_pu=values.get(self._p_key, 0.0), q_pu=values.get(self._q_key, 0.0),
                       p_ref_pu=pr, v_ref_pu=values.get(self._v_ref_key, 1.0),
                       freq_dev=freq_dev, angle_rel=angle_rel)
        self.last_log = log
        return ControlOutput(duty, tripped, log, theta=self.theta, omega=self.omega)

    # ------------------------------------------------------------ states and summary
    def get_state(self):
        return {**self.graph.get_state(), **gather({"prot": self.protection}), "command.u_dq_pu": self.u_cmd,
                "command.theta": self.command_theta, "command.omega": self.omega}

    def set_state(self, values):
        rest, prot = {}, {}
        for name, value in values.items():
            if name.startswith("prot."):
                prot[name] = value
            elif name == "command.u_dq_pu":
                self.u_cmd = complex(value)
            elif name == "command.theta":
                self.command_theta = float(value)
            elif name == "command.omega":
                self.omega = float(value)
            else:
                rest[name] = value
        self.graph.set_state(rest)
        scatter({"prot": self.protection}, prot)
        self.theta, self.omega = self.initial_sync()

    def summary(self):
        """Return this unit's summary; values for events which did not occur are ``None``."""
        st, trip, stage = self.protection.stats, self.protection.trip, self.stage
        summary = {
            "tripped": int(trip is not None),
            "trip_time": trip.t if trip and math.isfinite(trip.t) else None,
            "trip_cause": trip.cause if trip else None,
            "max_current_pu": st.max_current_pu,
            "modulation_saturation_fraction": stage.n_saturated / max(1, stage.n_updates),
            "modulation_saturation_first_t": stage.first_saturation_t,
            "rocof_max": st.rocof_max,
            "vac_min_pu": st.vac_min_pu,
            "vac_max_pu": st.vac_max_pu,
            **{f"{criterion}_first_t": getattr(st, f"{criterion}_first_t")
               for criterion in ("overcurrent", "undervoltage", "overvoltage", "frequency",
                                 "dc_voltage", "rocof")},
            "alarms": list(st.alarms),
        }
        if self._dc is not None:
            dc = self.graph.nodes[self._dc]
            summary["id_ref_limit_fraction"] = dc.n_clamped / max(1, dc.n_updates)
            summary["id_ref_limit_first_t"] = dc.first_clamp_t
        if self._sync is not None:
            summary["law"] = self.p.control.loops[self._sync].type
        return summary


def make_controller(cfg, scenario, **kwargs):
    """Build the :class:`UniteType` controller of a unit from ``cfg.control``."""
    return UniteType(cfg, scenario, **kwargs)
