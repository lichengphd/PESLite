"""The run of a simulation: the system and solver built from its parameters, the initial states, the
event loop, and the command line (the ``peslite`` command).

Between events the solver integrates the model; the events are those of the file, each unit's ADC
samples, loop updates, PWM publications, averaging-window openings and switching instants, and the
snapshots. Order at a coincident instant: file events, over-current check, ADC samples, loop updates,
PWM publication, window opening, switching instants, snapshot.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sysconfig
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..assembly.events import EVENT_TYPES, SWITCHING, connected_at, switching_schedule
from ..assembly.params import Change, Params, dump, dumps, load
from ..assembly.system import System
from ..assembly.unit import Unit
from ..components.pwm import Delay, Modulator
from ..control.blocks import abc2complex, complex2abc
from ..control.controller import Controller
from .integrators import Solver
from .model import ConfigError, expand_aliases, flatten, gather, resolve, scatter
from .multirate import make_solver
from .splitbound import split_error_bound

__all__ = ["Simulation", "SimulationResult", "Recorder", "main", "read_csv", "align", "window_ptp",
           "pointwise_errors"]


_EPS = 1e-10  # seconds; intervals shorter than this are not integrated separately


def _finite(y: np.ndarray) -> bool:
    """Return True when every entry of ``y`` is finite."""
    return all(map(math.isfinite, y.tolist()))


# ------------------------------------------------------------------ what a run produces

@dataclass
class SimulationResult:
    params: Params
    t: np.ndarray
    plant: dict[str, np.ndarray]
    control: dict[str, np.ndarray]
    states: dict[str, np.ndarray] = field(default_factory=dict)
    energy: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    n_rhs: int = 0

    @property
    def tripped(self) -> bool:
        return bool(self.summary.get("tripped", 0))

    def columns(self) -> dict[str, np.ndarray]:
        """Return flat real columns: plant quantities in SI and controller logs in pu.

        Names: ``<bus>.v_a``, ``<unit>.i_conv_a``, ``<branch>.i_a``, ``<source>.i_a``, ``<source>.angle``,
        ``<unit>.u_dc``/``<unit>.i_dc`` (V/A) and the controllers' logged signals.
        """
        cols: dict[str, np.ndarray] = {"t": self.t}
        for key, vec in self.plant.items():
            head, _, what = key.rpartition(".")
            if what in ("u_g", "i_c", "i", "u") and len(vec) and np.iscomplexobj(vec):
                stem = {"u_g": "v", "i_c": "i_conv", "i": "i", "u": "v"}[what]
                abc = np.array([complex2abc(z) for z in vec])
                for k, ph in enumerate("abc"):
                    cols[f"{head}.{stem}_{ph}"] = abc[:, k]
        for name in self.params.units:
            if f"{name}.u_dc" in self.plant:
                cols[f"{name}.u_dc"] = self.plant[f"{name}.u_dc"]
            if f"{name}.i_dc" in self.plant:
                cols[f"{name}.i_dc"] = self.plant[f"{name}.i_dc"]
        for name in self.params.sources:
            if f"{name}.angle" in self.plant:
                cols[f"{name}.angle"] = self.plant[f"{name}.angle"]
        for key, arr in self.plant.items():
            if key.startswith("ctrl."):
                cols[key[5:]] = arr
        return cols

    def to_csv(self, path: str | Path) -> None:
        _write(path, self.columns())

    def control_to_csv(self, path: str | Path) -> None:
        """Write one control-log CSV per unit, ``<path stem>.<unit>.csv``; return the paths."""
        path = Path(path)
        written = []
        for name in self.params.units:
            head = f"{name}."
            cols = {"t": self.control[f"{name}.t"]} if f"{name}.t" in self.control else {}
            cols.update({k[len(head):]: v for k, v in self.control.items()
                         if k.startswith(head) and k != f"{name}.t"})
            if len(cols) > 1:
                out = path.with_name(f"{path.stem}.{name}{path.suffix}")
                _write(out, cols)
                written.append(out)
        return written

    # ------------------------------------------------------------ states
    def final_states(self) -> dict[str, float]:
        """Return the last row of the state table (``t`` included)."""
        return {k: float(v[-1]) for k, v in self.states.items()}

    def states_to_csv(self, path: str | Path) -> None:
        """Write the state table at full precision, so a row read back reproduces the state exactly."""
        keys = list(self.states)
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            columns = [self.states[k].tolist() for k in keys]
            for row in zip(*columns):
                w.writerow([repr(v) for v in row])

    def energy_to_csv(self, path: str | Path) -> None:
        keys = list(self.energy)
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for row in zip(*(self.energy[k] for k in keys)):
                w.writerow([f"{v:.10g}" for v in row])

    def save(self, out_dir: str | Path, info: dict[str, Any] | None = None) -> list[Path]:
        """Write the configured output files into ``out_dir`` and return their paths.

        info: extra entries for ``summary.json``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        output = self.params.simulation.output
        if output.states:
            self.states_to_csv(out / "states.csv")
            written.append(out / "states.csv")
        if output.signals:
            self.to_csv(out / "plant.csv")
            written += [out / "plant.csv"] + self.control_to_csv(out / "control.csv")
        if output.energy and self.energy:
            self.energy_to_csv(out / "energy.csv")
            written.append(out / "energy.csv")
        summary = {**(info or {}), "wall_time": self.wall_time, "n_rhs": self.n_rhs, **self.summary}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        dump(self.params, out / "simulation.pes")
        written += [out / "summary.json", out / "simulation.pes"]
        return written

def _write(path: str | Path, cols: dict) -> None:
    keys = list(cols)
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in zip(*(cols[k] for k in keys)):
            w.writerow([f"{v:.10g}" for v in row])


class Recorder:
    """Collects snapshots during a run. ``keep_states=False`` keeps only the last state row."""

    def __init__(self, keep_states: bool = True) -> None:
        self.t: list[float] = []
        self.plant: dict[str, list] = {}
        self.ctrl_t: dict[str, list[float]] = {}   # one time grid per unit
        self.ctrl: dict[str, list] = {}
        self.last_ctrl_log: dict[str, dict[str, float]] = {}
        self.keep_states = keep_states
        self.state_keys: list[str] | None = None
        self.state_t: list[float] = []
        self.state_rows: list[list[float]] = []
        self.energy_t: list[float] = []
        self.energy_rows: dict[str, list] = {}

    def energy_row(self, t: float, columns: dict[str, float]) -> None:
        self.energy_t.append(t)
        for k, v in columns.items():
            self.energy_rows.setdefault(k, []).append(v)

    def state_row(self, t: float, row: dict[str, float]) -> None:
        if self.state_keys is None:
            self.state_keys = list(row)
        elif len(row) != len(self.state_keys):
            raise RuntimeError("the set of named states changed during the run")
        if not self.keep_states:
            self.state_t.clear()
            self.state_rows.clear()
        self.state_t.append(t)
        self.state_rows.append(list(row.values()))

    def plant_snapshot(self, t: float, signals: dict[str, Any]) -> None:
        n_before = len(self.t)
        self.t.append(t)
        for k, v in signals.items():
            self.plant.setdefault(k, []).append(v)
        for unit, log in self.last_ctrl_log.items():
            for k, v in log.items():
                key = f"ctrl.{unit}.{k}"
                if key not in self.plant:  # controller signal appearing after the first snapshots
                    self.plant[key] = [math.nan] * n_before
                self.plant[key].append(v)

    def control_sample(self, unit: str, t: float, log: dict[str, float]) -> None:
        """Record one controller's log at one of its sampling instants."""
        self.ctrl_t.setdefault(unit, []).append(t)
        for k, v in log.items():
            self.ctrl.setdefault(f"{unit}.{k}", []).append(v)

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        plant = {k: np.asarray(v) for k, v in self.plant.items()}
        ctrl = {f"{unit}.t": np.asarray(times) for unit, times in self.ctrl_t.items()}
        ctrl.update({k: np.asarray(v) for k, v in self.ctrl.items()})
        states: dict[str, np.ndarray] = {"t": np.asarray(self.state_t, dtype=float)}
        if self.state_keys:
            table = np.asarray(self.state_rows, dtype=float).reshape(len(self.state_rows), len(self.state_keys))
            states.update({k: table[:, j] for j, k in enumerate(self.state_keys)})
        energy: dict[str, np.ndarray] = {}
        if self.energy_t:
            energy = {"t": np.asarray(self.energy_t, dtype=float)}
            energy.update({k: np.asarray(v, dtype=float) for k, v in self.energy_rows.items()})
        return np.asarray(self.t), plant, ctrl, states, energy


# ------------------------------------------------------------------ reading results back

def read_csv(path: str | Path, skip_header: int = 0) -> np.ndarray:
    """Read a CSV file with a header row into a structured array, keeping column names verbatim (dots included)."""
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=skip_header, deletechars="")


def align(a: np.ndarray, b: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``a`` and ``b`` whose times fall on the same multiple of ``period``."""
    ia = np.round(a["t"] / period).astype(int)
    ib = np.round(b["t"] / period).astype(int)
    _common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    return a[ka], b[kb]


def window_ptp(d: np.ndarray, t0: float, t1: float, key: str) -> float:
    """Peak-to-peak of ``d[key]`` on ``[t0, t1)`` (NaN with fewer than 11 samples)."""
    m = (d["t"] >= t0) & (d["t"] < t1)
    return float(np.ptp(d[key][m])) if m.sum() > 10 else float("nan")


def pointwise_errors(a: np.ndarray, b: np.ndarray, keys: Iterable[str],
                     scale: Optional[Mapping[str, float]] = None) -> dict[str, dict[str, float]]:
    """Max and RMS of ``|a[k] - b[k]| / scale[k]`` for aligned arrays, per key present in both."""
    scale = scale or {}
    out = {}
    for key in keys:
        if key not in a.dtype.names or key not in b.dtype.names:
            continue
        s = scale.get(key, 1.0)
        x, y = a[key] / s, b[key] / s
        e = np.abs(x - y)
        out[key] = {"max": float(e.max()), "rms": float(np.sqrt(np.mean(e * e))),
                    "t_max": float(a["t"][int(e.argmax())]), "rms_ref": float(np.sqrt(np.mean(y * y)))}
    return out


# ------------------------------------------------------------------ the run

class Simulation:
    """Assemble the system and solver from parameters and run the event loop once."""

    def __init__(self, p: Params, system: Any = None, solver: Solver | None = None,
                 parts: Mapping[str, Callable[..., Any]] | None = None,
                 instances: Mapping[str, Any] | None = None) -> None:
        """Build every part from ``p`` unless replaced through ``parts`` or ``instances``.

        parts: factories keyed ``"system"`` or ``"solver"`` (called with ``p``), or ``"<unit>"``,
        ``"<unit>.ctrl"``, ``"<unit>.modulator"``, ``"<unit>.delay"`` (called with the unit's section).
        instances: ready objects under the same keys (also ``system=``, ``solver=``); with any
        instance the split error bound is not computed.
        """
        self.p = p
        self._factories: dict[str, Callable[..., Any]] = dict(parts or {})
        given: dict[str, Any] = dict(instances or {})
        if system is not None:
            given["system"] = system
        if solver is not None:
            given["solver"] = solver
        known = {"system", "solver"}
        for name in p.units:
            known |= {name, f"{name}.ctrl", f"{name}.modulator", f"{name}.delay"}
        for label, what in (("parts", self._factories), ("instances", given)):
            unknown = set(what) - known
            if unknown:
                raise ConfigError(f"{label}: unknown part(s) {sorted(unknown)}; known: {sorted(known)}")
        both = set(self._factories) & set(given)
        if both:
            raise ConfigError(f"parts: {sorted(both)} given both as a factory and as an instance")
        self._rebuildable = not given

        unit_parts = {}
        for name, cfg in p.units.items():
            for key in (name, f"{name}.ctrl", f"{name}.modulator", f"{name}.delay"):
                if key in self._factories:
                    unit_parts[key] = self._factories[key](cfg)
                elif key in given:
                    unit_parts[key] = given[key]
        if "system" in self._factories:
            self.system = self._factories["system"](p)
        elif "system" in given:
            self.system = given["system"]
        else:
            self.system = System(p, parts=unit_parts)
        if "solver" in self._factories:
            self.solver = self._factories["solver"](p)
        elif "solver" in given:
            self.solver = given["solver"]
        else:
            self.solver = make_solver(p.simulation.solver, model=self.system.model,
                                      ratings=self._ratings())
        for unit in self.system.units.values():
            for obj, proto in ((unit.ctrl, Controller), (unit.pwm.modulator, Modulator), (unit.pwm.delay, Delay)):
                if not isinstance(obj, proto):
                    raise TypeError(f"{unit.name}: {type(obj).__name__} does not satisfy the "
                                    f"{proto.__name__} protocol")
        check_events = getattr(self.system, "check_events", None)
        if check_events is not None:
            check_events(p)
        for needed, what in (("switch", switching_schedule(p.events)), ("apply", p.changes)):
            if what and not hasattr(self.system, needed):
                raise ConfigError(f"the events need a system with {needed}(); "
                                  f"{type(self.system).__name__} has none")
        if not isinstance(self.solver, Solver):
            raise TypeError(f"{type(self.solver).__name__} does not satisfy the Solver protocol")
        self.result: SimulationResult | None = None
        self.energy_problems: list[str] = []
        self.ph_report = None  # port-Hamiltonian structure report
        self._check_energy()

    @property
    def units(self) -> dict[str, Unit]:
        return self.system.units

    def unit(self, name: Optional[str] = None) -> Unit:
        """Return the named unit, or the only unit when ``name`` is omitted."""
        self.p.unit(name)  # checks the name
        return self.system.units[name if name is not None else next(iter(self.p.units))]

    def _actions(self) -> list[tuple[float, int, Any]]:
        """Return file-event actions as ``(time, file_order, action)``."""
        order = {name: index for index, name in enumerate(self.p.events)}
        actions = [(change.t, order[change.event], change) for change in self.p.changes]
        actions += [(event.t, order[name], event) for name, event in self.p.events.items()
                    if event.type != "set"]
        return sorted(actions, key=lambda action: action[:2])

    def _act(self, what: Any, t: float) -> None:
        """Apply one file event; the caller repacks the solver vector afterwards."""
        if isinstance(what, Change):
            self.system.apply(what)
        else:
            EVENT_TYPES[what.type].apply(what, self.system, t)

    def _ratings(self):
        """Return a function giving the rated (effort, flow) of a storage, or ``None``."""
        base = self.p.base
        system_ac = (base.v_phase_peak, base.i_phase_peak)
        units = self.p.units

        def rating(name: str, storage: Any) -> Optional[tuple[float, float]]:
            cfg = units.get(name.partition(".")[0])
            if storage.scale == 1.0:  # dc quantities: that unit's dc base
                if cfg is None:
                    return None
                dc = cfg.dc_base
                return (dc.v, dc.i) if dc.v > 0 else None
            return (cfg.base.v_phase_peak, cfg.base.i_phase_peak) if cfg is not None else system_ac

        return rating

    def _check_energy(self) -> None:
        """Build the structure report and apply ``simulation.energy_check`` (warn, strict or off)."""
        mode = self.p.simulation.energy_check
        model = getattr(self.system, "model", None)
        if model is None or not hasattr(model, "ph_report"):
            return
        solver = self.solver
        groups = getattr(solver, "groups", None)
        if groups is not None and len(groups) < 2:  # not split
            groups = None
        hold = None
        if hasattr(solver, "window") and hasattr(solver, "dt"):
            hold = {"step": solver.dt, "window": solver.window}
        zoh = {unit.zoh: 0.7 + 0.2j for unit in self.system.units.values()}
        self.ph_report = report = model.ph_report(zoh=zoh, groups=groups, hold=hold)
        problems = list(report.problems)
        if report.defaulted:
            problems.append(f"default energy declaration (black box, net power inferred from the neighbours) "
                            f"applied to: {report.defaulted}")
        bad_cuts = [c for c in report.cuts if not c.admissible]
        if bad_cuts:
            problems.append("subsystems stepped on their own whose interface cannot be accounted for: "
                            + "; ".join(f"{c.group} {c.members} holds {c.unaccounted}" for c in bad_cuts))
        problems += report.warnings
        self.energy_problems = problems
        if mode == "off":
            return
        if problems and mode == "strict":
            raise ConfigError("energy check: " + "; ".join(problems))
        for msg in problems:
            warnings.warn("energy check: " + msg, stacklevel=3)

    # ------------------------------------------------------------ states
    def _parts(self) -> dict[str, Any]:
        """The parts that have named states, by their prefix in the state table, in its order."""
        parts: dict[str, Any] = {"plant": self.system}
        for name, unit in self.system.units.items():
            parts[f"ctrl.{name}"] = unit.ctrl
            parts[f"pwm.{name}"] = unit.pwm
            parts[f"delay.{name}"] = unit.pwm.delay
            if unit.adc.averaging:
                parts[f"meas.{name}"] = unit.adc
        parts["solver"] = self.solver
        return parts

    def _start_duty(self) -> None:
        """Put the controllers' start-up duty ratios in force and fill the delay pipelines with them."""
        for unit in self.system.units.values():
            unit.pwm.d = np.array(unit.ctrl.initial_duty(), dtype=float)
            unit.pwm.delay.reset(unit.pwm.d)

    def watch_values(self, t: float,
                     ctrl_logs: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Return values addressable by ``simulation.progress.watch`` at time ``t``.

        Names include state-table columns, plant-table columns, state aliases, and complex states
        without their ``.re``/``.im`` suffix, in which case the magnitude is returned.
        """
        system = self.system
        plant = {key: np.asarray([value]) for key, value in system.signals(t).items()}
        plant.update({f"ctrl.{unit}.{key}": np.asarray([value])
                      for unit, log in ctrl_logs.items() for key, value in log.items()})
        row = SimulationResult(params=self.p, t=np.asarray([t]), plant=plant, control={}).columns()
        values = {key: float(value[0]) for key, value in row.items() if key != "t"}
        state = gather(self._parts())
        for alias, name in getattr(system, "state_aliases", {}).items():
            if f"plant.{name}" in state:
                state[f"plant.{alias}"] = state[f"plant.{name}"]
        values.update({key: abs(value) for key, value in state.items() if isinstance(value, complex)})
        values.update(flatten(state))
        return values

    def state_names(self) -> list[str]:
        """Return the state-table column names after ``t``, i.e. the valid ``initial.states`` keys."""
        if self.result is None:
            self._start_duty()
        return list(flatten(gather(self._parts())))

    def _apply_initial(self, t0: float) -> np.ndarray:
        """Load ``initial.states`` into the parts; return the solver vector."""
        system = self.system
        aliases = {f"plant.{a}": f"plant.{c}" for a, c in getattr(system, "state_aliases", {}).items()}
        presets_of = getattr(system, "state_presets", None)
        presets = presets_of(t0) if presets_of is not None else {}
        if callable(presets):  # state-dependent keywords: strip the "plant." prefix
            inner = presets
            presets = lambda key, word: inner(key[6:] if key.startswith("plant.") else key, word)  # noqa: E731
        self._start_duty()  # so that the duty ratios and delay pipelines have their states
        template = gather(self._parts())
        try:
            given = expand_aliases(self.p.simulation.initial.states, aliases)
            if callable(presets):
                for bus in getattr(system, "buses", {}):
                    key = f"plant.{bus}.u_C"
                    if key not in template or any(name == key or name.startswith(key + ".") for name in given):
                        continue
                    try:
                        given[key] = presets(key, "source")
                    except ValueError:
                        pass
            values = resolve(template, given, presets)
        except (KeyError, ValueError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            if aliases:
                msg += f". Aliases: {', '.join(f'{a} -> {c}' for a, c in aliases.items())}"
            raise ConfigError(msg) from None
        # 1. power stage (unset states stay zero), events up to t0, then the solver vector
        scatter({"plant": system}, values)
        if hasattr(system, "switch"):
            for name, steps in switching_schedule(self.p.events).items():
                if not connected_at(steps, t0):
                    system.switch(name, False, t0)
        for event_t, _, what in self._actions():
            if event_t <= t0 + _EPS and getattr(what, "type", None) not in SWITCHING:
                self._act(what, event_t)
        y = system.model.get_initial_values()
        # 2. start-up modulation matching each unit's terminal voltage at t0, or the given duty ratios
        system.model.sync(t0, y)
        for name, unit in system.units.items():
            align = getattr(unit.ctrl, "align_startup", None)
            if align is not None:
                align(unit.ports.u_g(), unit.ports.u_dc())
            unit.adc.seed()  # averaging window starts at t0
            unit.pwm.d = np.array(unit.ctrl.initial_duty(), dtype=float)
            scatter({f"pwm.{name}": unit.pwm}, values)
        # 3. controllers, delay pipelines, measurement windows, solver
        for name, unit in system.units.items():
            scatter({f"ctrl.{name}": unit.ctrl}, values)
            unit.pwm.delay.reset(unit.pwm.d)
            scatter({f"delay.{name}": unit.pwm.delay}, values)
            if unit.adc.averaging:
                scatter({f"meas.{name}": unit.adc}, values)
        scatter({"solver": self.solver}, values)
        return y

    # ------------------------------------------------------------ the loop
    def run(self, t_end: float | None = None) -> SimulationResult:
        if self.result is not None:
            raise RuntimeError("a Simulation runs once; build a new one for another run")
        p, system = self.p, self.system
        mdl, solver = system.model, self.solver
        settle = getattr(solver, "settle", None)
        parameters_changed = getattr(solver, "parameters_changed", None)
        t_end = p.simulation.t_end if t_end is None else t_end
        output = p.simulation.output
        log_period = output.period
        control_every = max(1, output.control_every)
        stop_on_trip = p.simulation.stop_on_trip
        progress = p.simulation.progress
        progress_every = progress.period if progress.enable else 0.0
        watch = list(progress.watch) if progress.enable else []
        rec = Recorder(keep_states=output.states)
        wall0 = time.time()

        units = list(system.units.values())
        periods = [u.pwm.period for u in units]
        t_start = min(round(p.simulation.initial.t / T) * T for T in periods)
        if t_end <= t_start + _EPS:
            raise ValueError(f"t_end = {t_end} must be after the start time {t_start}")
        # the run ends at the earliest period boundary at or after t_end
        t_final = min(math.ceil((t_end - _EPS) / T) * T for T in periods)
        # trips already handled, by unit (a trip loaded with the initial states is handled at the first sample)
        tripped = {u.name: bool(getattr(u.ctrl, "tripped", False)) or u.tripped for u in units}
        for unit in units:
            pwm = unit.pwm
            pwm.k = int(round(t_start / pwm.period))
            pwm.start = pwm.t_next
            pwm.sync = unit.ctrl.initial_sync() if hasattr(unit.ctrl, "initial_sync") else (None, None)
            if hasattr(unit.ctrl, "reset_clocks"):
                unit.ctrl.reset_clocks(t_start)
        averaging = [u for u in units if u.adc.averaging]  # units whose ADC averages
        y = self._apply_initial(t_start)
        if parameters_changed is not None:
            parameters_changed()
        for unit in units:
            unit.adc.latest = unit.adc.measure(t_start)
        n_log = max(0, math.ceil(t_start / log_period - 1e-9))
        actions = deque(action for action in self._actions() if action[0] > t_start + _EPS)
        next_report = t_start + progress_every if progress_every > 0 else math.inf
        t_local = t_start
        n_rhs0 = getattr(solver, "n_rhs", 0)

        mode = p.simulation.energy_check
        energy_on = mode != "off" and hasattr(mdl, "energy_balance")
        worst = {"tellegen": 0.0, "balance": 0.0}
        stop_reason = ""
        coarse_reported = False
        stopped = False
        parts = self._parts()

        def snapshot(t_now: float) -> None:
            try:
                with np.errstate(over="raise"):
                    mdl.sync(t_now, y)
                    rec.plant_snapshot(t_now, system.signals(t_now))
                    rec.state_row(t_now, flatten(gather(parts)))
                    if energy_on and _finite(y):
                        rep = mdl.energy_balance(t_now, y)
                        rec.energy_row(t_now, rep.columns())
                        scale = rep.scale
                        worst["tellegen"] = max(worst["tellegen"], abs(rep.tellegen) / scale)
                        worst["balance"] = max(worst["balance"], rep.max_residual / scale)
            except (OverflowError, FloatingPointError):
                pass  # diverged state: not recorded

        def watched(t_now: float) -> str:
            """Format the quantities appended to a progress line."""
            mdl.sync(t_now, y)
            values = self.watch_values(t_now, rec.last_ctrl_log)
            unknown = [name for name in watch if name not in values]
            if unknown:
                raise ConfigError(f"simulation.progress.watch: unknown name(s) {unknown}; "
                                  f"known: {sorted(values)}")
            return "".join(f"   {name} = {values[name]:.6g}" for name in watch)

        @np.errstate(over="raise")  # overflow counts as divergence
        def integrate(t_a: float, t_b: float, y: np.ndarray, event: bool) -> np.ndarray:
            y = solver(mdl.rhs, t_a, t_b, y).y
            return settle(t_b, y) if event and settle is not None else y

        def advance(t_a: float, t_b: float, event: bool = False) -> bool:
            """Integrate over ``[t_a, t_b]``; ``event`` means the model changes at ``t_b``."""
            nonlocal y, t_local, stop_reason, coarse_reported
            try:
                y = integrate(t_a, t_b, y, event)
            except (OverflowError, FloatingPointError):
                stop_reason = f"diverged: the states overflowed between t = {t_a:.6g} s and {t_b:.6g} s"
                t_local = t_a
                return True
            t_local = t_b
            finite = _finite(y)
            if averaging and finite:
                mdl.sync(t_b, y)
                for unit in averaging:
                    unit.adc.accumulate(t_b - t_a)
            if not finite:
                stop_reason = f"diverged: non-finite states at t = {t_b:.6g} s"
                return True
            coarse = getattr(solver, "coarse_hold", None)
            if coarse is not None and not coarse_reported:
                coarse_reported = True
                t_c, label, rel = coarse
                msg = (f"the window holding {label} (closed at t = {t_c:.6g} s) bounds its change by {rel:.2f} of "
                       f"the rated effort: the window is too long for this storage")
                if mode == "strict":
                    stop_reason = "coarse window: " + msg
                    return True
                if mode == "warn":
                    warnings.warn("energy check: " + msg, stacklevel=3)
            return False

        def settle_now() -> None:
            """Finish deferred integration with the current model before it changes now."""
            nonlocal y
            if settle is not None:
                y = settle(t_local, y)
            mdl.set_states(y)

        def do_trip(unit: Unit) -> None:
            nonlocal y
            tripped[unit.name] = True
            settle_now()
            unit.trip()
            y = mdl.get_initial_values()

        def publish(unit: Unit, t_k: float) -> bool:
            """Sample, run the controller and publish its duty ratios; return True if a new trip stops the run."""
            mdl.sync(t_k, y)
            out = unit.ctrl(t_k, unit.adc.sample(t_k))
            if out.log is not None:
                rec.last_ctrl_log[unit.name] = out.log
                if unit.pwm.k % control_every == 0:
                    rec.control_sample(unit.name, t_k, out.log)
            new_trip = out.tripped and not tripped[unit.name]
            if new_trip:
                do_trip(unit)
            unit.pwm.publish(out.d_abc, out.theta, out.omega)
            return new_trip and stop_on_trip

        def modulate(unit: Unit, t_k: float) -> None:
            """Start the unit's PWM period at ``t_k``."""
            mdl.set_zoh_input(unit.zoh, unit.pwm.modulate(t_k))

        # ---------------------------------------------------------------- the event grid
        for unit in units:  # first period: start-up duty ratios, no controller call
            modulate(unit, unit.pwm.t_next)
        while True:
            t_log = n_log * log_period
            t_stop = t_log if t_log < t_final - _EPS else t_final  # the earliest event of all
            for unit in units:
                pwm, adc = unit.pwm, unit.adc
                t_c = pwm.t_next
                t_stop = min(t_stop, t_c if t_c < t_final - _EPS else t_final, pwm.next_switch,
                             adc.t_window(t_c), adc.t_sample(pwm.start), getattr(unit.ctrl, "next_event", math.inf))
            if actions:
                t_stop = min(t_stop, actions[0][0])
            event_due = bool(actions and actions[0][0] <= t_stop + _EPS)
            advancing = t_stop > t_local + _EPS
            if advancing and advance(t_local, t_stop, event_due):
                stopped = True
                snapshot(t_local)
                break
            # File events run before the coincident protection/control/PWM events.
            if event_due:
                if advancing:  # advance() already settled at t_stop
                    mdl.set_states(y)
                else:
                    settle_now()
                while actions and actions[0][0] <= t_stop + _EPS:
                    self._act(actions.popleft()[2], t_stop)
                if parameters_changed is not None:
                    parameters_changed()
                y = mdl.get_initial_values()
            # 0. over-current check of units whose switching interval ends here
            for unit in units:
                pwm = unit.pwm
                ends = abs(pwm.next_switch - t_stop) < _EPS or abs(pwm.end - t_stop) < _EPS
                check = getattr(unit.ctrl, "fast_check", None)
                if ends and check is not None and not tripped[unit.name]:
                    mdl.set_states(y)
                    if check(t_local, unit.adc.phase_currents()):
                        do_trip(unit)
                        if stop_on_trip:
                            stopped = True
            if stopped:
                snapshot(t_local)
                break
            # ADC samples due here, before any loop update
            for unit in units:
                if abs(unit.adc.t_sample(unit.pwm.start) - t_stop) < _EPS:
                    mdl.sync(t_local, y)
                    unit.adc.peek(t_local)
            # loop-only updates: latest sample, no PWM/delay advance
            for unit in units:
                if (getattr(unit.ctrl, "next_event", math.inf) <= t_stop + _EPS
                        and abs(unit.pwm.t_next - t_stop) >= _EPS and t_stop < t_final - _EPS):
                    unit.ctrl.update(t_stop, unit.adc.latest)
            # 1. PWM publications due here (with any due loop updates)
            for unit in units:
                if abs(unit.pwm.t_next - t_stop) < _EPS and t_stop < t_final - _EPS:
                    if publish(unit, t_stop):
                        stopped = True
                    modulate(unit, t_stop)
            if stopped:
                snapshot(t_local)
                break
            # 1b. an averaging window shorter than the PWM update period opens here
            for unit in units:
                if abs(unit.adc.t_window(unit.pwm.t_next) - t_stop) < _EPS:
                    mdl.sync(t_local, y)
                    unit.adc.open()
            # 2. switching instants due here
            for unit in units:
                for q in unit.pwm.switches(t_stop, _EPS):
                    mdl.set_zoh_input(unit.zoh, q)
            # 3. snapshots
            if abs(t_log - t_stop) < _EPS and t_stop < t_final - _EPS:
                snapshot(t_local)
                n_log += 1
            if t_stop >= t_final - _EPS:
                break
            if t_local >= next_report:
                suffix = watched(t_local) if watch else ""
                print(f"  t = {t_local:8.4f} s   wall {time.time() - wall0:8.1f} s{suffix}",
                      flush=True)
                next_report += progress_every

        if not stopped:
            # final instant: only the loop/PWM events due here
            for unit in units:
                if abs(unit.pwm.t_next - t_local) < _EPS:
                    publish(unit, t_local)
                elif getattr(unit.ctrl, "next_event", math.inf) <= t_local + _EPS:
                    unit.ctrl.update(t_local, unit.adc.latest)
        if not rec.t or rec.t[-1] < t_local - _EPS:
            snapshot(t_local)
        if stop_reason:
            warnings.warn(f"simulation stopped at t = {t_local:.6g} s: {stop_reason}", stacklevel=2)
        t_arr, plant_arrays, ctrl_arrays, state_arrays, energy_arrays = rec.arrays()
        summary: dict[str, Any] = {}
        for name, unit in system.units.items():
            if hasattr(unit.ctrl, "summary"):
                summary.update({f"{name}.{k}": v for k, v in unit.ctrl.summary().items()})
        summary["tripped"] = int(any(tripped.values()))
        summary["t_start"] = t_start
        summary["t_stop"] = float(t_arr[-1]) if len(t_arr) else t_start
        if stop_reason:
            summary["stop_reason"] = stop_reason
        coarse = getattr(solver, "coarse_hold", None)
        if coarse is not None:
            summary["interface_coarse_hold"] = f"{coarse[1]} at t = {coarse[0]:.6g} s: {coarse[2]:.3g} of rating"
        if self.ph_report is not None:
            summary["ph_verdict"] = self.ph_report.verdict
            summary["ph_defaulted"] = list(self.ph_report.defaulted)
            summary["ph_report"] = self.ph_report.to_dict()
        if energy_on:
            summary["energy_tellegen_max_rel"] = worst["tellegen"]
            summary["energy_balance_max_rel"] = worst["balance"]
            summary["energy_problems"] = list(self.energy_problems)
        interface = getattr(solver, "interface", None)
        if interface:  # split-interface indicators
            summary.update({f"interface_{k}": v for k, v in interface.items()})
        if getattr(solver, "window_log", None) and p.simulation.solver.linearisations > 0:
            if self._rebuildable:
                loop = SystemLoop(self)
                if loop.period is None:  # no common control period
                    summary["split_bound"] = (f"the converters' control periods {loop.periods} have no "
                                              f"common multiple within 64 updates, so the loop has no "
                                              f"period to linearise: no bound")
                else:
                    summary.update(split_error_bound(
                        solver, loop, t_arr, state_arrays, p.simulation.solver.linearisations,
                        labels=mdl.state_labels(), prefix="plant."))
            else:  # parts given as instances
                summary["split_bound"] = ("the loop has custom parts given as instances, so it cannot be "
                                          "rebuilt for the linearisation: pass them as factories of the "
                                          "parameter tree for a bound")
        self.result = SimulationResult(
            params=p, t=t_arr, plant=plant_arrays, control=ctrl_arrays, states=state_arrays,
            energy=energy_arrays, summary=summary, wall_time=time.time() - wall0,
            n_rhs=getattr(solver, "n_rhs", 0) - n_rhs0)
        return self.result


_LOOPMAP_EPS = 1e-9

def macro_period(periods: list[float], limit: int = 64) -> float | None:
    """Return the smallest common multiple of ``periods`` (s), or ``None`` beyond ``limit`` longest periods."""
    longest = max(periods)
    for n in range(1, limit + 1):
        T = n * longest
        if all(abs(T / p - round(T / p)) <= 1e-9 * max(1.0, T / p) for p in periods):
            return T
    return None


class SystemLoop:
    """Closed-loop map of the whole system over one common control period, rebuilt from the parameters."""

    def __init__(self, sim: Any) -> None:

        self._cls = Simulation
        self._factories = dict(sim._factories)
        self.periods = [float(u.pwm.period) for u in sim.units.values()]
        for u in sim.units.values():
            self.periods.extend(T for T in getattr(u.ctrl, "periods", {}).values() if T)
        self.period = macro_period(self.periods)
        self.w0 = 2.0 * math.pi * sim.p.base.f0
        self._flags = {k for k, v in gather(sim._parts()).items() if isinstance(v, bool)}
        rated = sim.solver.rated_effort
        self._rated = ({lab: float(r) for lab, r in zip(sim.system.model.state_labels(), rated)}
                       if rated is not None else {})
        p = sim.p
        sp = p.simulation
        self._p = dataclasses.replace(
            p,
            simulation=dataclasses.replace(
                sp, energy_check="off", stop_on_trip=False,
                progress=dataclasses.replace(sp.progress, enable=False),
                                           output=dataclasses.replace(sp.output, period=self.period or 1.0,
                                                                      states=True, energy=False, signals=False),
                                           solver=dataclasses.replace(sp.solver, subsystems={},
                                                                      linearisations=0)))

    # ------------------------------------------------------------ coordinates
    def coordinates(self, names: list[str]) -> list[tuple[str, list[str], float]]:
        """Group state-table columns into map coordinates ``(kind, names, scale)``."""
        cols: list[tuple[str, list[str], float]] = []
        used: set[str] = set()
        for c in names:
            if c in used or c == "t" or c.startswith("solver.") or ".clock." in c:
                continue  # solver bookkeeping, not a loop state
            if c in self._flags:
                cols.append(("frozen", [c], 1.0))
                used.add(c)
            elif c.endswith(".re") and c[:-3] + ".im" in names:
                base = c[:-3]
                turns = base.startswith("plant.")  # alpha-beta; a controller vector is already in dq
                scale = self._rated.get(base[len("plant."):] + ".re", 0.0) if turns else 0.0
                cols.append(("vector" if turns else "fixed", [c, base + ".im"], scale if scale > 0 else 1.0))
                used.update(cols[-1][1])
            elif c.endswith(".d_a") and c[:-4] + ".d_b" in names and c[:-4] + ".d_c" in names:
                cols.append(("triple", [c[:-4] + f".d_{ph}" for ph in "abc"], 1.0))
                used.update(cols[-1][1])
            else:
                scale = self._rated.get(c[len("plant."):], 0.0) if c.startswith("plant.") else 0.0
                cols.append(("scalar", [c], scale if scale > 0 else 1.0))
                used.add(c)
        return cols

    def to_frame(self, row: dict[str, float], t: float, cols: list) -> np.ndarray:
        theta = self.w0 * t
        rot = complex(math.cos(theta), -math.sin(theta))
        out: list[float] = []
        for kind, names, _s in cols:
            if kind == "vector":
                z = complex(row[names[0]], row[names[1]]) * rot
                out += [z.real, z.imag]
            elif kind == "fixed":
                out += [row[names[0]], row[names[1]]]
            elif kind == "triple":
                d = np.array([row[k] for k in names])
                z = abc2complex(d) * rot
                out += [z.real, z.imag, float(d.mean())]
            else:
                out.append(row[names[0]])
        return np.array(out)

    def from_frame(self, x: np.ndarray, t: float, cols: list, row: dict[str, float]) -> dict[str, float]:
        theta = self.w0 * t
        rot = complex(math.cos(theta), math.sin(theta))
        out = dict(row)
        i = 0
        for kind, names, _s in cols:
            if kind == "vector":
                z = complex(x[i], x[i + 1]) * rot
                out[names[0]], out[names[1]] = z.real, z.imag
            elif kind == "fixed":
                out[names[0]], out[names[1]] = float(x[i]), float(x[i + 1])
            elif kind == "triple":
                d = complex2abc(complex(x[i], x[i + 1]) * rot) + x[i + 2]
                for k, name in enumerate(names):
                    out[name] = float(d[k])
            elif kind == "scalar":
                out[names[0]] = float(x[i])
            i += 2 if kind in ("vector", "fixed") else 3 if kind == "triple" else 1
        return out

    # --------------------------------------------------------- one period of it
    def advance(self, row: dict[str, float], t: float,
                view: tuple[str, float, str] | None = None) -> dict[str, float]:
        """Return the state row one period after ``t``.

        view: optional ``(plant state label, offset, channel)``; the offset is seen by the
        integration (``"b"``) or by the end-of-period sample (``"c"``).
        """
        initial = dataclasses.replace(
            self._p.simulation.initial, t=t,
            states={k: v for k, v in row.items() if not k.startswith("solver.")})
        p = dataclasses.replace(self._p, simulation=dataclasses.replace(
            self._p.simulation, t_end=t + self.period, initial=initial))
        sim = self._cls(p, parts=self._factories)
        model = sim.system.model
        if view is not None:
            label, off, channel = view
            vec = np.zeros(model.n_states)
            vec[list(model.state_labels()).index(label)] = off
            t_end = t + self.period
            if channel == "b":  # what the integration sees
                rhs0 = model.rhs
                model.rhs = lambda tt, y: rhs0(tt, y + vec)  # type: ignore[method-assign]
            else:  # what the end-of-period sample reads
                sync0 = model.sync
                model.sync = lambda tt, y: sync0(tt, y + vec if tt >= t_end - _LOOPMAP_EPS else y)  # type: ignore[method-assign]
        res = sim.run()
        out = {k: float(v[-1]) for k, v in res.states.items() if k != "t"}
        if view is not None and view[2] == "c":
            out["plant." + view[0]] -= view[1]
        return out


# --------------------------------------------------------- command-line entry

def _example_configs_dir() -> Path:
    """Return the root examples directory, from a checkout or an installed wheel."""
    source = Path(__file__).resolve().parents[3] / "examples"
    if source.is_dir():
        return source
    try:
        dist = distribution("peslite")
    except PackageNotFoundError:
        return source
    # data-files live below the installation prefix, outside the import package.  A ``--target``
    # installation puts them below the metadata root; a normal environment uses sysconfig's data root.
    roots = (Path(dist.locate_file("")).resolve(), Path(sysconfig.get_path("data")).resolve())
    for root in roots:
        candidate = root / "share" / "peslite" / "examples"
        if candidate.is_dir():
            return candidate
    return source


_EXAMPLE_CONFIGS = _example_configs_dir()
_RESULTS = Path.cwd() / "output"
_CONFIG_SUFFIXES = (".pes", ".yaml", ".yml", ".json")


def _config_path(value: str | None) -> Path:
    if value is None:
        configs = sorted(
            p for p in _EXAMPLE_CONFIGS.iterdir()
            if p.is_file() and p.suffix.lower() in _CONFIG_SUFFIXES
        )
        if not configs:
            raise FileNotFoundError("no bundled example configurations available")
        return configs[0]
    path = Path(value).expanduser()
    candidates = [path]
    if not path.suffix:
        candidates += [path.with_suffix(s) for s in _CONFIG_SUFFIXES]
    example = _EXAMPLE_CONFIGS / path
    candidates.append(example)
    if not path.suffix:
        candidates += [example.with_suffix(s) for s in _CONFIG_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"configuration not found: {value}")

def parse_override(text: str):
    key, _, value = text.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError("expected key.path=value")
    for cast in (int, float):
        try:
            v = cast(value)
            if cast is int and str(v) != value:
                continue
            return key, v
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    if value.lower() in ("null", "none"):
        return key, None
    return key, value


def main(argv=None) -> int:
    """Run a simulation file and save its states, summary and configured signals."""
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?",
                    help="simulation file path or bundled example name; default: first example by filename")
    ap.add_argument("--set", action="append", default=[], type=parse_override, metavar="PATH=VALUE",
                    help="override a dotted parameter path (repeatable)")
    ap.add_argument("--initial", default=None, metavar="FILE",
                    help="initial values: a states.csv row or a simulation file's simulation.initial block")
    ap.add_argument("--initial-time", type=float, default=None, metavar="T",
                    help="with a states.csv: start from the row at time T instead of the last row")
    ap.add_argument("--averaging", action="store_true",
                    help="run every unit with averaging enabled; each unit keeps its configured "
                         "averaging.over value, and this option wins over --set")
    ap.add_argument("--out", default=None,
                    help="output directory (default: output/<file name>, or "
                         "output/<file name>-averaging when --averaging changes a unit)")
    ap.add_argument("--progress", type=float, default=None, metavar="SECONDS",
                    help="print a progress line every SECONDS of simulated time")
    ap.add_argument("--watch", action="append", default=None, metavar="NAME",
                    help="append a state or plant/control value to each progress line; repeatable, "
                         "or use NAME,NAME")
    ap.add_argument("--list-states", action="store_true",
                    help="print the state names generated for this configuration and exit")
    ap.add_argument("--ph-report", action="store_true",
                    help="print the port-Hamiltonian structure report of the system (always built) and exit")
    ap.add_argument("--resolved", action="store_true",
                    help="print the complete resolved simulation file and exit")
    args = ap.parse_args(argv)

    try:
        config = _config_path(args.config)
    except FileNotFoundError as exc:
        ap.error(str(exc))

    overrides = dict(args.set)
    if args.progress is not None:
        overrides["simulation.progress.enable"] = 1
        overrides["simulation.progress.period"] = args.progress
    if args.watch is not None:
        overrides["simulation.progress.watch"] = [name for value in args.watch
                                                   for name in value.split(",") if name]
    p = load(config, initial=args.initial,
                    initial_time=args.initial_time, **overrides)
    averaged = args.averaging and any(not unit.averaging.enable for unit in p.units.values())
    if args.averaging:
        p = p.replace(**{f"units.{name}.averaging.enable": 1 for name in p.units})
    if args.watch and not p.simulation.progress.enable:
        ap.error("--watch prints on the progress lines: add --progress SECONDS")
    if args.resolved:
        print(dumps(p), end="")
        return 0
    sim = Simulation(p)
    if args.list_states:
        print("\n".join(sim.state_names()))
        return 0
    if args.ph_report:
        print(sim.ph_report)
        return 0
    def bridge_model(unit):
        return (f"averaging over the {unit.averaging.over.replace('_', ' ')}"
                if unit.averaging.enable else "switching")

    units = ", ".join(f"{n} ({u.control.type}, {bridge_model(u)}, "
                      f"PWM {1e-3 / u.pwm.update_period:.0f} kHz, delay {u.delay.steps})"
                      for n, u in p.units.items())
    print(f"peslite: {config}  units={units}  "
          f"solver={p.simulation.solver.type}/{p.simulation.solver.method}  "
          f"t = {p.simulation.initial.t} .. {p.simulation.t_end} s  "
          f"({len(p.simulation.initial.states)} initial values given)")
    if sim.ph_report is not None:
        print(f"structure: {sim.ph_report.verdict} (state coverage {sim.ph_report.coverage:.0%}"
              f"{'; ' + '; '.join(sim.energy_problems) if sim.energy_problems else ''})")
    r = sim.run()
    s = r.summary
    print(f"done: wall {r.wall_time:.1f} s, rhs evaluations {r.n_rhs}, stop at {s.get('t_stop', 0):.4f} s, "
          f"tripped={bool(s.get('tripped'))}")
    for name in p.units:
        print(f"  {name}: max |i| = {s.get(f'{name}.max_current_pu', 0):.4f} pu, "
              f"trip: {s.get(f'{name}.trip_cause') or 'none'}, "
              f"alarms: {', '.join(s.get(f'{name}.alarms') or []) or 'none'}")
    if "stop_reason" in s:
        print(f"stopped: {s['stop_reason']}")
    if "interface_kappa" in s:
        print(f"split: measured hold error {s.get('interface_max_error_rated', s['interface_max_rel_error']):.2e}, "
              f"bound kappa {s['interface_kappa']:.2e} of rating, within bound: {bool(s['interface_within_bound'])}"
              + (f"; coarse window: {s['interface_coarse_hold']}" if "interface_coarse_hold" in s else ""))
    if "split_error_bound_rated" in s:
        print(f"split error, from the linearised closed loop: <= {s['split_error_bound_rated']:.2e} of rating "
              f"({s['split_bound']})")
    out = Path(args.out) if args.out else _RESULTS / (f"{config.stem}-averaging" if averaged else config.stem)
    written = r.save(out, info={"config": str(config)})
    print("wrote", ", ".join(str(w) for w in written))
    final = r.final_states()
    if final:
        width = max(map(len, final))
        print(f"final values (last row of states.csv, t = {final['t']:.6g} s):")
        for name, value in final.items():
            if name != "t":
                print(f"  {name:<{width}}  {value: .10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
