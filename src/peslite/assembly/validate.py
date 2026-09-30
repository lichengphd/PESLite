"""Checks across the sections of a constructed parameter tree (read-only).

Covers topology, dc link, control, timing, solver settings and initial values; the construction
(:func:`peslite.assembly.params.from_dict`) has checked each field on its own.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import fields, is_dataclass

from typing import TYPE_CHECKING

from ..components.network import element_buses
from ..solver.integrators import ADAPTIVE_METHODS, FIXED_METHODS
from ..solver.model import ConfigError
from ..solver.multirate import parse_step
from .events import NAMED, SWITCHABLE, SWITCHING

if TYPE_CHECKING:
    from .params import Params

__all__ = ["validate"]


def _whole(r: float) -> bool:
    """Is ``r`` a whole number of at least one?"""
    n = round(r)
    return n >= 1 and abs(r - n) <= 1e-9 * max(1.0, r)


def _misaligned(name: str, r: float, where: str, both_ways: bool = False) -> None:
    """Warn if a ratio of periods is not a whole number (optionally in either direction)."""
    if _whole(r) or (both_ways and _whole(1.0 / r)):
        return
    warnings.warn(f"{where}: {name} = {r:.6g} is not a whole number"
                  f"{' in either direction' if both_ways else ''}, so the two grids do not line "
                  f"up; the periods are used as configured", stacklevel=3)


def _switched(section, where: str) -> None:
    """Validate a section controlled by an ``enable`` switch."""
    if not section.enable:
        return
    for f in fields(section):
        value = getattr(section, f.name)
        if f.name == "enable" or not f.init:
            continue
        if value is None:
            raise ConfigError(f"{where}.{f.name}: required when enable is 1")
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
            raise ConfigError(f"{where}.{f.name} must be finite and positive, got {value}")


def _dclink(cfg, where):
    cap, source = cfg.capacitor, cfg.source
    if not math.isfinite(cfg.vdc_ref) or cfg.vdc_ref <= 0:
        raise ConfigError(f"{where}.vdc_ref must be finite and positive")
    if cap is not None:
        if not math.isfinite(cap.c) or cap.c <= 0:
            raise ConfigError(f"{where}.capacitor.c must be finite and positive")
        if not math.isfinite(cap.r_esr) or cap.r_esr < 0:
            raise ConfigError(f"{where}.capacitor.r_esr must be finite and nonnegative")
    for key in ("i", "k", "v", "r"):
        if not math.isfinite(getattr(source, key)):
            raise ConfigError(f"{where}.source.{key} must be finite")
    if source.v < 0 or source.r < 0:
        raise ConfigError(f"{where}.source.v and r must be nonnegative")
    allowed = {"current": {"i", "k"}, "voltage": {"v", "r"}, "none": set()}[source.type]
    for key, default in {"i": 0.0, "k": 0.0, "v": 0.0, "r": 0.0}.items():
        if key not in allowed and getattr(source, key) != default:
            raise ConfigError(f"{where}.source.{key} is not used by source.type = {source.type!r}")
    if cap is None and source.type != "voltage":
        raise ConfigError(f"{where}: without a capacitor, a voltage source is required")
    if cap is not None and source.type == "voltage" and cap.r_esr + source.r <= 0:
        raise ConfigError(f"{where}: a voltage source with a capacitor requires positive source resistance or ESR")


def _pwm(w, base, where: str) -> None:
    """Check switching frequency, update period and synchronous pulse ratio."""
    if not math.isfinite(w.f_sw) or w.f_sw <= 0.0:
        raise ConfigError(f"{where}.f_sw must be finite and positive, got {w.f_sw}")
    T_pwm = w.effective_update_period
    if not math.isfinite(T_pwm) or T_pwm <= 0:
        raise ConfigError(f"{where}.update_period must be finite and positive")
    if w.sync == "synchronous":
        ratio = w.f_sw / base.f0
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio):
            raise ConfigError(f"{where}.sync = 'synchronous' needs an integer pulse ratio "
                              f"pwm.f_sw / base.f0, got {w.f_sw} / {base.f0} = {ratio}")
    _misaligned("pwm.switching_period / pwm.update_period", w.switching_period / T_pwm,
                f"{where}.update_period", both_ways=True)


def _control(c, dclink, where: str) -> None:
    """Check typed control loops and their requirements on the DC side."""
    if not c.loops:
        raise ConfigError(f"{where}.loops must contain at least one loop")
    for name, cfg in c.loops.items():
        if name in {"measurement", "references", "held", "clock", "command", "prot"}:
            raise ConfigError(f"{where}.loops.{name}: reserved loop name")
        T = cfg.period  # None where the loop type allows it: runs on upstream updates
        if T is not None and (not math.isfinite(T) or T <= 0):
            raise ConfigError(f"{where}.loops.{name}.period must be finite and positive")
        for key, value in vars(cfg).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ConfigError(f"{where}.loops.{name}.{key} must be finite")
            if is_dataclass(value) and hasattr(value, "enable"):
                _switched(value, f"{where}.loops.{name}.{key}")
        if cfg.type == "matching" and dclink.capacitor is None:
            raise ConfigError(f"{where}: matching control needs dclink.capacitor")
        if cfg.type == "matching" and cfg.k_theta_pu is None and c.references.vdc_ref_pu <= 0:
            raise ConfigError(f"{where}.references.vdc_ref_pu must be positive for automatic matching gain")
    for name, value in vars(c.references).items():
        if name == "omega" and value is None:
            continue
        if not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{where}.references: expected finite scalar references")


def _sampling(c, m, T_pwm: float, where: str) -> None:
    """Check the ADC sampling period and averaging window."""
    if c.sampling_period is not None and (not math.isfinite(c.sampling_period) or c.sampling_period <= 0):
        raise ConfigError(f"{where}.control.sampling_period must be positive")
    if c.sampling_period is not None and c.sampling_period > T_pwm * (1 + 1e-9):
        raise ConfigError(f"{where}.control.sampling_period is longer than pwm.update_period")
    sampling_period = c.sampling_period if c.sampling_period is not None else T_pwm
    _misaligned("pwm.update_period / control.sampling_period", T_pwm / sampling_period,
                f"{where}.control.sampling_period")
    if m.window is not None and m.window <= 0.0:
        raise ConfigError(f"{where}.measurement.window must be > 0, got {m.window}")
    if m.average == "window" or m.u_dc == "window":
        window = m.window if m.window is not None else T_pwm
        if window > T_pwm * (1.0 + 1e-9):
            raise ConfigError(
                f"{where}.measurement.window = {window} is longer than the PWM update period {T_pwm}")
        if sampling_period < T_pwm * (1.0 - 1e-9):
            raise ConfigError(
                f"{where}: measurement.average = 'window' cannot be combined with a sampling period "
                f"shorter than the PWM update period")
    elif m.window is not None:
        warnings.warn(f"{where}.measurement.window has no effect: no quantity is window-averaged",
                      stacklevel=4)


def _unit(u, base, where: str) -> None:
    """Check one converter unit."""
    if u.s_base is not None and (not math.isfinite(u.s_base) or u.s_base <= 0.0):
        raise ConfigError(f"{where}.s_base must be finite and positive, got {u.s_base}")
    if u.ac_filter.l_f <= 0.0:
        raise ConfigError(f"{where}.ac_filter.l_f must be > 0")
    _dclink(u.dclink, f"{where}.dclink")
    _pwm(u.pwm, base, f"{where}.pwm")
    _control(u.control, u.dclink, f"{where}.control")
    _sampling(u.control, u.measurement, u.pwm.effective_update_period, where)
    if u.delay.steps < 0:
        raise ConfigError(f"{where}.delay.steps must be >= 0")
    protection = u.protection
    for name in ("overcurrent", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof"):
        _switched(getattr(protection, name), f"{where}.protection.{name}")
    if not math.isfinite(protection.hold) or protection.hold < 0:
        raise ConfigError(f"{where}.protection.hold must be finite and >= 0, got {protection.hold}")
    if not math.isfinite(protection.rocof.window) or protection.rocof.window <= 0:
        raise ConfigError(f"{where}.protection.rocof.window must be finite and positive, "
                          f"got {protection.rocof.window}")


def _network(p: Params) -> None:
    """Check element identities, network values and bus references."""
    if not p.buses:
        raise ConfigError("buses: a system needs at least one bus")
    if not p.units:
        raise ConfigError("units: a system needs at least one converter")
    seen: dict[str, str] = {}
    for kind in NAMED:
        for name in getattr(p, kind):
            if name in seen:
                raise ConfigError(f"{kind}.{name}: the name is already used by {seen[name]}; element "
                                  f"names are unique across buses, branches, sources, units and elements")
            seen[name] = f"{kind}.{name}"
    for name, bus in p.buses.items():
        if bus.c <= 0.0:
            raise ConfigError(f"buses.{name}.c must be > 0: the shunt capacitor is what makes the "
                              f"bus voltage a state (a bus without one would need a node-voltage solver)")
    def _bus_of(where: str, bus: str) -> None:
        if bus not in p.buses:
            raise ConfigError(f"{where}: unknown bus {bus!r}; known: {sorted(p.buses)}")
    for name, br in p.branches.items():
        _bus_of(f"branches.{name}.from_bus", br.from_bus)
        _bus_of(f"branches.{name}.to_bus", br.to_bus)
        if br.from_bus == br.to_bus:
            raise ConfigError(f"branches.{name}: both ends are {br.from_bus!r}")
        if br.l <= 0.0:
            raise ConfigError(f"branches.{name}.l must be > 0")
    for name, src in p.sources.items():
        _bus_of(f"sources.{name}.bus", src.bus)
        if src.f is not None and (not math.isfinite(src.f) or src.f <= 0.0):
            raise ConfigError(f"sources.{name}.f must be finite and positive, got {src.f}")
        if not math.isfinite(src.angle):
            raise ConfigError(f"sources.{name}.angle must be finite")
        if src.l <= 0.0:
            raise ConfigError(f"sources.{name}.l must be > 0: a source is an emf behind an impedance")
    for name, u in p.units.items():
        _bus_of(f"units.{name}.bus", u.bus)
    for name, element in p.elements.items():
        for key, bus in element_buses(element).items():
            _bus_of(f"elements.{name}.{key}", bus)
        for key, value in vars(element).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"elements.{name}.{key} must be finite")


def _solver(s, time_step_averaging: list[str]) -> None:
    """Check solver methods, bridge compatibility and subsystem step ratios."""
    if s.type == "fixed" and s.method not in FIXED_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {FIXED_METHODS}")
    if s.type == "adaptive" and s.method not in ADAPTIVE_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {ADAPTIVE_METHODS}")
    if time_step_averaging and s.type != "fixed":
        name = time_step_averaging[0]
        raise ConfigError(f"units.{name}.averaging.over = 'time_step' requires the fixed-step "
                          f"solver (simulation.solver.type = 'fixed')")
    if s.sweeps < 1:
        raise ConfigError("simulation.solver.sweeps must be >= 1")
    if s.linearisations < 0:
        raise ConfigError("simulation.solver.linearisations must be >= 0")
    for name, value in s.subsystems.items():
        where = f"simulation.solver.subsystems.{name}"
        if s.type != "fixed":
            raise ConfigError(f"{where}: subsystems on their own step need the fixed-step solver")
        if isinstance(value, dict) and (set(value) - {"step", "method"} or "step" not in value):
            raise ConfigError(f"{where}: expected {{step, method}} (step required), got {sorted(value)}")
        try:
            _ratio, method = parse_step(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        if method is not None and method not in FIXED_METHODS + ADAPTIVE_METHODS:
            raise ConfigError(f"{where}.method: {method!r} is not one of {FIXED_METHODS + ADAPTIVE_METHODS}")


def _events(p: Params) -> None:
    """Check event targets, times, and each target's connect/disconnect sequence."""
    switched: dict[str, tuple[str, bool]] = {}
    at: dict[tuple[str, float], str] = {}
    for name, event in sorted(p.events.items(), key=lambda item: item[1].t):
        where = f"events.{name}"
        for key, value in vars(event).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"{where}.{key} must be finite")
        target = getattr(event, "target", None)
        section = p.section_of(target) if isinstance(target, str) else None
        if target is not None and section is None:
            known = sorted(name for named in NAMED for name in getattr(p, named))
            raise ConfigError(f"{where}.target: unknown name {target!r}; known: {known}")
        if event.type not in SWITCHING:
            continue
        if section not in SWITCHABLE:
            raise ConfigError(f"{where}.target: {target!r} is a bus, which has no breaker; {event.type} "
                              f"switches a unit, source, branch or element")
        on = event.type == "connect"
        previous = switched.get(target)
        if previous is not None and previous[1] == on:
            raise ConfigError(f"{where}: {target!r} is already {event.type}ed by events.{previous[0]}")
        if (target, event.t) in at:
            raise ConfigError(f"{where}: {target!r} is also switched at t = {event.t} "
                              f"by events.{at[(target, event.t)]}")
        at[(target, event.t)] = name
        switched[target] = (name, on)


def _initial(p: Params) -> None:
    """Check the start/end times and the shape of initial state values."""
    t0 = p.simulation.initial.t
    if t0 < 0.0:
        raise ConfigError("simulation.initial.t must be >= 0")
    for name, u in p.units.items():  # t0 must be on every unit's PWM update grid
        T = u.pwm.effective_update_period
        if abs(round(t0 / T) * T - t0) > 1e-9 * max(1.0, t0):
            raise ConfigError(f"simulation.initial.t = {t0} is not on the PWM update grid of {name!r} (a multiple of "
                              f"units.{name}.pwm.update_period = {T})")
    if p.simulation.t_end <= t0:
        raise ConfigError(f"simulation.t_end = {p.simulation.t_end} must be after simulation.initial.t = {t0}")
    for key, value in p.simulation.initial.states.items():
        if isinstance(value, str):
            continue  # a keyword, resolved when the plant is built
        elif isinstance(value, (list, tuple)):
            if len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                raise ConfigError(f"simulation.initial.states.{key}: expected [re, im], got {value!r}")
        elif not isinstance(value, (int, float)):
            raise ConfigError(f"simulation.initial.states.{key}: expected a number, [re, im], a flag or a keyword, "
                              f"got {value!r}")


def _progress(progress) -> None:
    """Check progress settings; watched names are resolved at the first progress line."""
    _switched(progress, "simulation.progress")
    watch = progress.watch
    if not isinstance(watch, list) or not all(isinstance(name, str) and name for name in watch):
        raise ConfigError(f"simulation.progress.watch: expected a list of quantity names, got {watch!r}")


def validate(p: Params) -> Params:
    """Check a :class:`Params` tree and return it unchanged; raise ConfigError if invalid."""
    _network(p)
    time_step = [name for name, unit in p.units.items()
                 if unit.averaging.enable and unit.averaging.over == "time_step"]
    for name, unit in p.units.items():
        _unit(unit, p.base, f"units.{name}")
        if unit.pwm.sync == "synchronous" and name in time_step:
            raise ConfigError(f"units.{name}.pwm.sync = 'synchronous' is not available with time-step "
                              f"averaging (units.{name}.averaging.over = 'time_step'), which has an "
                              f"asynchronous carrier")
    _solver(p.simulation.solver, time_step)
    _events(p)
    _progress(p.simulation.progress)
    _initial(p)
    return p
