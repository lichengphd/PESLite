"""A converter controller as one closed discrete-time block.

At each control interrupt the controller takes an ADC sample (:class:`Measurement`, SI), scales it to
the unit's pu bases, runs its loop network (:class:`ControlGraph`: the loops of :mod:`.loops`
connected by typed ports) and its start-up sequencer, and turns the voltage command into duty
ratios. It sees only its samples, host commands and its own state. Protection and physical trip
actions belong to the unit.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..solver.model import ConfigError, gather, scatter
from .blocks import peak_abs, smoothstep
from .loops import (ANGLE, CURRENT, DC_VOLTAGE, FREQUENCY, I_AB, LOOP_TYPES, POWER_PU, V_AB, V_DQ, VOLTAGE)
from .modulation import CONFIGURED, OutputStage

__all__ = ["Measurement", "ControlMeasurement", "ControlOutput", "Controller", "Sequencer",
           "ControlGraph", "default_wiring", "UniteType", "make_controller"]


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

    ``d_abc`` and ``gates`` go through the PWM registers. ``startup_complete`` reports that the
    controller's start-up ramp has finished; the owning unit decides how to use it.
    ``theta`` (rad) and ``omega`` (rad/s): the controller's synchronization angle and frequency,
    required by synchronous modulators; ``None`` allows only asynchronous modulation.
    """

    d_abc: NDArray[np.float64]  # duty ratios of phases a, b, c in [0, 1]
    log: dict[str, float] | None = None
    theta: float | None = None  # synchronization angle (rad)
    omega: float | None = None  # synchronization frequency (rad/s)
    gates: bool = True
    startup_complete: bool = True


@runtime_checkable
class Controller(Protocol):
    """Sampled controller ``(t, Measurement) -> ControlOutput``.

    Optional host-facing methods are ``command``, ``track``, ``retune``, state access and
    ``summary``; the required target interface is only the sampled call. Protection is deliberately
    outside this interface.
    """

    def __call__(self, t: float, meas: Measurement) -> ControlOutput: ...


# ------------------------------------------------------------------ the start-up sequence

class Sequencer:
    """Start-up sequence advanced only by the controller's own interrupts."""

    def __init__(self, T: float, run: bool = True, ramp: float = 0.0) -> None:
        self.T = float(T)
        self.run = bool(run)
        self.ramp = float(ramp)
        self.steps = 0
        self.running = False
        self.ramp_value = 0.0
        self.complete = False

    def command(self, run: bool, ramp: float = 0.0) -> None:
        """Take a host run/stop command; every new run has its own ramp."""
        self.run = bool(run)
        if self.run:
            self.ramp = float(ramp)
        else:
            self.steps = 0

    def step(self) -> None:
        self.running = self.run
        if not self.running:
            self.steps, self.ramp_value, self.complete = 0, 0.0, False
            return
        progress = 1.0 if self.ramp <= 0.0 else min(1.0, self.steps * self.T / self.ramp)
        self.ramp_value = smoothstep(progress)
        self.complete = progress >= 1.0
        if not self.complete:
            self.steps += 1

    def setpoints(self, p_ref: float, q_ref: float, v_ref: float) -> tuple[float, float, float]:
        return self.ramp_value * p_ref, q_ref, v_ref

    def get_state(self) -> dict[str, Any]:
        return {"run": self.run, "ramp": self.ramp, "steps": self.steps}

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"run", "ramp", "steps"}
        if unknown:
            raise KeyError(f"the start-up sequence has no state(s) {sorted(unknown)}")
        if "run" in values:
            self.run = bool(values["run"])
        if "ramp" in values:
            self.ramp = float(values["ramp"])
        if "steps" in values:
            steps = float(values["steps"])
            if steps < 0.0 or steps != int(steps):
                raise ValueError(f"the start-up sequence: steps must be a whole number >= 0, got {values['steps']}")
            self.steps = min(int(steps), self._end())

    def _end(self) -> int:
        if self.ramp <= 0.0:
            return 0
        n = max(0, math.ceil(self.ramp / self.T) - 2)
        while n * self.T / self.ramp < 1.0:
            n += 1
        return n


# ------------------------------------------------------------------ the loop network

# kinds of input source
_HELD, _MEASURED, _CONSTANT = 0, 1, 2
_MEASUREMENTS = {"meas.u_g": "u_g", "meas.i_c": "i_c", "meas.u_dc": "u_dc"}
_REFERENCES = {"id_ref_pu": CURRENT, "iq_ref_pu": CURRENT, "theta": ANGLE, "omega": FREQUENCY,
               "p_ref_pu": POWER_PU, "q_ref_pu": POWER_PU, "v_ref_pu": VOLTAGE,
               "vdc_ref_pu": DC_VOLTAGE, "zero_v_pu": V_DQ}


class ControlGraph:
    """The loops of ``cfg.ctrl.loops`` wired by typed ports and run by the control interrupt.

    Loop inputs are held outputs of other loops, the pu measurement (``meas.u_g``, ``.i_c``,
    ``.u_dc``) or the references (``references.<name>``); ``connections`` and ``outputs`` are the
    default wiring, which ``cfg.ctrl.connections`` and ``.outputs`` override. Loops run in signal
    order, every interrupt or every n-th interrupt according to their periods.

    Retuned parameters are queued by :meth:`schedule` and take effect at the next update. Changed
    loops are rebuilt from their new parameters and continue from their named states.
    """

    def __init__(self, cfg, sequencer, connections, outputs):
        self.cfg, self.sequencer = cfg, sequencer
        self._pending: list[Any] = []
        self.on_retune = None
        self.connections = {**connections, **cfg.ctrl.connections}
        self.outputs = {**outputs, **cfg.ctrl.outputs}
        self.nodes = {}
        self.periods = {}
        self._T, self._t0 = float(cfg.ctrl.period), cfg.pwm.grid_offset
        self.values: dict[str, Any] = {}
        self._out_specs: dict[str, tuple] = {}
        self.updated = set()
        refs = cfg.ctrl.references
        self.references = {f.name: getattr(refs, f.name) for f in fields(refs)}
        types = {"meas.u_g": V_AB, "meas.i_c": I_AB, "meas.u_dc": DC_VOLTAGE,
                 **{f"references.{k}": v for k, v in _REFERENCES.items()}}
        for name, loop in cfg.ctrl.loops.items():
            cls = LOOP_TYPES.get(loop.type)
            if cls is None:
                raise ConfigError(f"ctrl.loops.{name}: no loop type {loop.type!r} is registered")
            self.nodes[name] = node = cls(loop, cfg, sequencer)
            self._out_specs[name] = tuple((port, f"{name}.{port}", spec.complex_value)
                                          for port, spec in node.outputs.items())
            self.periods[name] = loop.period
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
                    raise ConfigError(f"ctrl.connections.{target}: missing or unknown source {source!r}")
                if types[source] != spec:
                    raise ConfigError(f"ctrl.connections.{target}: incompatible signal {source!r}: "
                                      f"{types[source]} -> {spec}")
                owner = source.partition(".")[0]
                if owner in self.nodes and port not in node.delayed:
                    dependencies[name].add(owner)
        if set(self.connections) - targets:
            raise ConfigError(f"ctrl.connections: unknown input ports {sorted(set(self.connections) - targets)}")
        for port, spec in {"u_dq": V_DQ, "theta": ANGLE, "omega": FREQUENCY}.items():
            source = self.outputs.get(port)
            if source not in types or types[source] != spec:
                raise ConfigError(f"ctrl.outputs.{port}: missing or incompatible source {source!r}")
        if set(self.outputs) - {"u_dq", "theta", "omega"}:
            raise ConfigError("ctrl.outputs: only u_dq, theta and omega are supported")
        self.order = []
        pending = dict(dependencies)
        while pending:
            ready = [n for n, deps in pending.items() if not deps]
            if not ready:
                raise ConfigError("ctrl.connections: instantaneous cycle; insert an explicit unit_delay")
            for name in ready:
                self.order.append(name)
                pending.pop(name)
            for deps in pending.values():
                deps.difference_update(ready)
        self._rewire()
        self._every = tuple((name, max(1, int(round(T / self._T))))
                            for name, T in self.periods.items())

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
            ctrl = cfg.ctrl
            if ctrl.references != self.cfg.ctrl.references:
                self.references = {f.name: getattr(ctrl.references, f.name)
                                   for f in fields(ctrl.references)}
                self._rewire()
            changed = [name for name, params in ctrl.loops.items()
                       if params != self.cfg.ctrl.loops[name]]
            self.cfg = cfg
            for name in changed:
                self._retune(name)

    def _retune(self, name):
        """Rebuild one loop and continue from the old instance's state."""
        old = self.nodes[name]
        params = self.cfg.ctrl.loops[name]
        fresh = LOOP_TYPES[params.type](params, self.cfg, self.sequencer)
        try:
            fresh.set_state(old.get_state())
        except (KeyError, ValueError) as exc:
            raise ConfigError(
                f"ctrl.loops.{name}: with the parameters of a set event it has other "
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

    def due(self, t):
        """Return the loops that run at the control interrupt at ``t``."""
        k = int(round((t - self._t0) / self._T))
        return {name for name, every in self._every if k % every == 0}

    # ------------------------------------------------------------ running
    def track(self, meas):
        """Pre-synchronise every loop that supports ``track(inputs)``."""
        done = set()
        for name in self.order:
            track = getattr(self.nodes[name], "track", None)
            out = track(self._gather(self._immediate[name], meas)) if track is not None else None
            if out is not None:
                self._store(name, out)
                done.add(name)
        return done

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
        due = self.due(t)
        if not due:
            return updated
        nodes, gather_, immediate = self.nodes, self._gather, self._immediate
        for name in self.order:
            if name not in due:
                continue
            self._store(name, nodes[name].update(t, gather_(immediate[name], meas)))
            updated.add(name)
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
        return {"loops": {n: {"type": self.cfg.ctrl.loops[n].type, "period": self.periods[n]}
                          for n in self.order}, "connections": dict(self.connections), "outputs": dict(self.outputs)}

    # ------------------------------------------------------------ states
    def get_state(self):
        state = gather(self.nodes)
        state.update({f"held.{self._state_port(k)}": v for k, v in self.values.items()})
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
            else:
                rest[key] = value
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
    loops = cfg.ctrl.loops
    roles = {name: getattr(LOOP_TYPES.get(p.type), "role", None) for name, p in loops.items()}

    def find(role):
        found = [n for n, r in roles.items() if r == role]
        if len(found) > 1:
            raise ConfigError(f"ctrl.loops: ambiguous {family} role {found}; use ctrl.type = 'custom' "
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
        "pll": {"v": "meas.u_g"},
        "dc_voltage": {"u_dc": "meas.u_dc", "vdc_ref": "references.vdc_ref_pu"},
        "current": {"v": "meas.u_g", "i": "meas.i_c", "frame": frame, "omega": omega,
                    "extra": extra, "id_ref": f"{va or dc}.id_ref" if va or dc else "references.id_ref_pu",
                    "iq_ref": f"{va}.iq_ref" if va else "references.iq_ref_pu"},
        "power": {"v": "meas.u_g", "i": "meas.i_c"},
        "sync": {"p": f"{power}.p", "q": f"{power}.q", "v": "meas.u_g", "i": "meas.i_c",
                 "u_dc": "meas.u_dc", "p_ref": "references.p_ref_pu", "q_ref": "references.q_ref_pu",
                 "v_ref": "references.v_ref_pu"},
        "impedance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "i": "meas.i_c",
                      "extra": extra},
        "admittance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "v": "meas.u_g"},
        "damping": {"i": "meas.i_c", "frame": frame},
    }
    wires = {f"{name}.{port}": source for name, role in roles.items()
             for port, source in ports_of.get(role, {}).items()
             if port in LOOP_TYPES[loops[name].type].inputs}
    terminal = vi if family == "gfm" and vi else cc
    return wires, {"u_dq": f"{terminal}.u_dq", "theta": theta, "omega": omega}


# ------------------------------------------------------------------ the controller

class UniteType:
    """Configurable closed controller block with loops, start-up and output stage.

    The host gives it only ``command(run, ramp)`` and parameter updates. At each interrupt it sees
    one :class:`Measurement` and returns duty ratios, PWM enable and synchronization estimates.
    """

    def __init__(self, cfg, pwm_method=None, *, limiter=CONFIGURED):
        self.p = cfg
        self.sequencer = Sequencer(cfg.ctrl.period)
        wires, outputs = default_wiring(cfg, cfg.ctrl.type)
        self.graph = graph = ControlGraph(cfg, self.sequencer, wires, outputs)
        self.periods = graph.periods
        self.v_base, self.i_base, self.w0 = cfg.base.v_phase_peak, cfg.base.i_phase_peak, cfg.base.w0
        self.v_dc_base = cfg.dc_base.v
        self.stage = OutputStage(cfg, pwm_method=pwm_method, limiter=limiter)
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
        self._find_terminal()
        graph.on_retune = self._find_terminal
        self._frame_key = f"{graph.outputs['theta'].partition('.')[0]}.frame"
        self._dc, self._sync = first("dc_voltage"), first("sync")
        self._frame_cc, self._log_cc = first("current", "frame"), first("current", "id_ref")
        self._is_gfl = cfg.ctrl.type == "gfl"
        self._p_key, self._q_key, self._v_ref_key = f"{first('power')}.p", f"{first('power')}.q", f"{self._sync}.v_ref"

    def _find_terminal(self, name=None):
        """Refresh the terminal loop reference after a loop is rebuilt."""
        self._terminal_node = self.graph.nodes.get(self._terminal)

    def describe(self):
        return self.graph.describe()

    def retune(self, cfg, paths, t):
        """Queue new controller parameters for its next interrupt."""
        if any(path.startswith("ctrl.") for path in paths):
            self.graph.schedule(cfg)
        self.p = cfg

    def initial_sync(self):
        values = {**self.graph.values, **{f"references.{k}": v for k, v in self.graph.references.items()}}
        return float(values[self.graph.outputs["theta"]]), float(values[self.graph.outputs["omega"]])

    def command(self, run, ramp=0.0):
        """Take the host's run/stop command."""
        self.sequencer.command(run, ramp)

    def track(self, meas):
        """Pre-synchronise grid-forming laws to an SI terminal sample."""
        if self.graph.track(self._pu(meas)):
            self.theta, self.omega = self.initial_sync()
            self.command_theta = self.theta

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

    def log_names(self) -> tuple[str, ...]:
        """Return the fixed controller-log schema before the first interrupt."""
        names = ["id_pu", "iq_pu", "vd_pu", "vq_pu", "vac_pu", "vdc_pu",
                 "m_max", "in_service"]
        if self.p.meas.average == "window":
            names += ["vd_raw_pu", "vq_raw_pu", "id_raw_pu", "iq_raw_pu"]
        if self.p.meas.u_dc == "window":
            names.append("vdc_raw_pu")
        names += (["id_ref_pu", "freq_dev", "angle_rel"] if self._is_gfl else
                  ["p_pu", "q_pu", "p_ref_pu", "v_ref_pu", "freq_dev", "angle_rel"])
        return tuple(names)

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
        sequence = self.sequencer
        sequence.step()
        self.stage.new_instant()
        self.graph.update(t, control_meas, finalize=self._accept_command)
        frame = self.graph.values.get(self._frame_key, self.theta)
        if self._frame_cc is not None:
            frame = self.graph.input(self._frame_cc, "frame", control_meas)
        rot = cmath.exp(-1j * frame)
        self.v_dq, self.i_dq = control_meas.u_g * rot, control_meas.i_c * rot
        freq_dev = (self.omega - self.w0) / (2 * math.pi)
        refs = self.graph.references
        gates = sequence.running
        duty = self.stage.modulate(t, self.u_cmd, self.command_theta, control_meas.u_dc, count=gates)
        log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
               "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
               "vac_pu": abs(self.v_dq), "vdc_pu": control_meas.u_dc,
               "m_max": peak_abs(self.stage.m_abc), "in_service": 1.0 if gates else 0.0,
               **self._raw_log(meas, frame)}
        angle_rel = (self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi
        if self._is_gfl:
            id_ref = self.graph.input(self._log_cc, "id_ref", control_meas) if self._log_cc else 0.0
            log.update(id_ref_pu=id_ref, freq_dev=freq_dev, angle_rel=angle_rel)
        else:
            values = self.graph.values
            pr, qr, vr = sequence.setpoints(refs["p_ref_pu"], refs["q_ref_pu"], refs["v_ref_pu"])
            log.update(p_pu=values.get(self._p_key, 0.0), q_pu=values.get(self._q_key, 0.0),
                       p_ref_pu=pr, v_ref_pu=values.get(self._v_ref_key, 1.0),
                       freq_dev=freq_dev, angle_rel=angle_rel)
        self.last_log = log
        return ControlOutput(duty, log=log, theta=self.theta, omega=self.omega,
                             gates=gates, startup_complete=sequence.complete)

    # ------------------------------------------------------------ states and summary
    def get_state(self):
        return {**self.graph.get_state(), "command.u_dq_pu": self.u_cmd,
                "command.theta": self.command_theta, "command.omega": self.omega,
                **gather({"sequence": self.sequencer})}

    def set_state(self, values):
        rest, sequence = {}, {}
        for name, value in values.items():
            if name.startswith("sequence."):
                sequence[name] = value
            elif name == "command.u_dq_pu":
                self.u_cmd = complex(value)
            elif name == "command.theta":
                self.command_theta = float(value)
            elif name == "command.omega":
                self.omega = float(value)
            else:
                rest[name] = value
        self.graph.set_state(rest)
        scatter({"sequence": self.sequencer}, sequence)
        self.theta, self.omega = self.initial_sync()

    def summary(self):
        """Return controller and modulation observations."""
        stage = self.stage
        summary = {
            "modulation_saturation_fraction": stage.n_saturated / max(1, stage.n_updates),
            "modulation_saturation_first_t": stage.first_saturation_t,
        }
        if self._dc is not None:
            dc = self.graph.nodes[self._dc]
            summary["id_ref_limit_fraction"] = dc.n_clamped / max(1, dc.n_updates)
            summary["id_ref_limit_first_t"] = dc.first_clamp_t
        if self._sync is not None:
            summary["law"] = self.p.ctrl.loops[self._sync].type
        return summary


def make_controller(cfg, **kwargs):
    """Build the :class:`UniteType` controller of a unit from ``cfg.ctrl``."""
    return UniteType(cfg, **kwargs)
