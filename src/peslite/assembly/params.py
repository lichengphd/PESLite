"""The parameters of a simulation file: their classes, the pu bases, and their construction from a mapping.

Electrical inputs without suffix are SI values; ``_pu`` names are per unit. Plant quantities are
stored in SI, controller quantities in pu. A field listed in a class's ``_quantities`` may be given
in SI or pu. :func:`from_dict` builds the checked :class:`Params` from a mapping in one pass: it
converts each section's inputs on the bases in force there (the system base, or a unit's own), checks
fields, types and choices, and builds the loops with the parameters of their registered types (a
loop type declares its own, :class:`peslite.control.loops.Loop`); :func:`to_dict` converts back.

pu bases, AC: Vb = sqrt(2/3)*v_ll_rms (phase peak), Ib = S/(1.5*Vb), Zb = Vb/Ib, w0 = 2*pi*f0;
DC: Vdc = vdc_ref, Idc = S/Vdc, Zdc = Vdc**2/S. L base = Z/w0, C base = 1/(w0*Z).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections.abc import Mapping as _Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from ..control.loops import LOOP_TYPES
from ..solver.model import ConfigError
from .validate import validate

__all__ = [
    "ConfigError", "load", "dump", "read_tree", "read_initial", "from_dict", "to_dict", "BaseValues", "DCBase",
    "BusParams", "BranchParams", "SourceParams", "SourceEventParams",
    "ACFilterParams", "DCLinkParams", "DCCapacitorParams", "DCSourceParams", "PWMParams", "MeasurementParams",
    "ReferenceParams", "ControlParams", "DelayParams", "ProtectionParams",
    "StartupParams", "SetpointStepParams", "PLLGainStepParams", "UnitEventParams", "UnitParams",
    "SolverParams", "LogParams", "SimulationParams", "InitialParams", "OutputParams", "Params",
]


# ------------------------------------------------------------------ pu bases and SI/pu inputs

@dataclass(frozen=True)
class DCBase:
    """DC-side bases derived from the rated DC voltage and the power base."""

    v: float  # rated dc voltage (V)
    i: float  # s_base / v (A)
    z: float  # v / i (ohm)
    w0: float

    def C(self, c_pu: float) -> float:
        """pu -> F."""
        return c_pu / (self.w0 * self.z)


@dataclass(frozen=True)
class BaseValues:
    """System base (``s_base`` VA, ``v_ll_rms`` V, ``f0`` Hz) and derived phase-peak SI bases."""

    s_base: float
    v_ll_rms: float
    f0: float
    v_phase_peak: float = field(init=False)
    i_phase_peak: float = field(init=False)
    z_base: float = field(init=False)
    w0: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("s_base", "v_ll_rms", "f0"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be finite and positive")
        v = math.sqrt(2.0 / 3.0) * self.v_ll_rms
        i = self.s_base / (1.5 * v)
        object.__setattr__(self, "v_phase_peak", v)
        object.__setattr__(self, "i_phase_peak", i)
        object.__setattr__(self, "z_base", v / i)
        object.__setattr__(self, "w0", 2.0 * math.pi * self.f0)

    # pu -> SI
    def L(self, x_pu: float) -> float:
        return x_pu * self.z_base / self.w0

    def R(self, r_pu: float) -> float:
        return r_pu * self.z_base

    def C(self, c_pu: float) -> float:
        return c_pu / (self.w0 * self.z_base)

    def dc(self, vdc_ref: float) -> DCBase:
        """Return the DC base for the rated DC voltage ``vdc_ref`` (V)."""
        i = self.s_base / vdc_ref if vdc_ref > 0.0 else 0.0
        return DCBase(v=vdc_ref, i=i, z=vdc_ref / i if i > 0.0 else 0.0, w0=self.w0)

    def parameter_scales(self, vdc_ref: float = 1.0) -> dict[str, float]:
        """Return the SI value of one pu for each quantity name."""
        dc = self.dc(vdc_ref)
        return {"1": 1.0, "voltage": self.v_phase_peak, "current": self.i_phase_peak,
                "power": self.s_base, "frequency": self.w0,
                "resistance": self.z_base, "inductance": self.L(1.0), "capacitance": self.C(1.0),
                "dc_voltage": dc.v, "dc_current": dc.i, "dc_resistance": dc.z,
                "dc_capacitance": dc.C(1.0)}


def quantity_metadata(cls, name: str) -> dict:
    """Merge the class attribute ``name`` over the class hierarchy into one dict."""
    result = {}
    for parent in reversed(cls.__mro__):
        result.update(parent.__dict__.get(name, {}))
    return result


def quantity_fields(cls, values: dict | None = None) -> dict:
    quantities = quantity_metadata(cls, "_quantities")
    if hasattr(cls, "_signal_quantities"):
        quantities.update(cls._signal_quantities(values or {}))
    return quantities


def quantity_name(cls, name: str, values: dict | None = None) -> str:
    """Return the stored field name for the SI or ``_pu`` spelling of a parameter."""
    name = quantity_metadata(cls, "_input_aliases").get(name, name)
    for target in quantity_fields(cls, values):
        si = target.removesuffix("_pu")
        if name in (si, si + "_pu"):
            return target
    return name


def convert_quantities(data: dict, cls, scales: dict[str, float], where: str) -> dict:
    """Convert SI/pu inputs of one section to its storage units and fill pu defaults.

    Raises ConfigError if both the SI and the pu spelling of a parameter are given.
    """
    aliases = quantity_metadata(cls, "_input_aliases")
    quantities = quantity_fields(cls, data)
    out, seen = {}, {}
    for name, value in data.items():
        spelling = aliases.get(name, name)
        target = quantity_name(cls, name, data)
        if target in seen:
            raise ConfigError(f"{where}.{target}: both {seen[target]!r} and {name!r} specify the same parameter")
        seen[target] = name
        if target in quantities and value is not None:
            if isinstance(value, bool):
                raise ConfigError(f"{where}.{name}: expected a number")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{where}.{name}: expected a number, got {value!r}") from None
            if spelling.endswith("_pu") != target.endswith("_pu"):
                scale = _scale(quantities[target], scales)
                value = value * scale if spelling.endswith("_pu") else value / scale
        out[target] = value
    for name, value in quantity_metadata(cls, "_defaults_pu").items():
        if name not in out:
            out[name] = value if name.endswith("_pu") else value * _scale(quantities[name], scales)
    return out


def _scale(expression: str, scales: dict[str, float]) -> float:
    numerator, *denominators = expression.split("/")
    value = math.prod(scales[name] for name in numerator.split("*"))
    for denominator in denominators:
        value /= math.prod(scales[name] for name in denominator.split("*"))
    return value


# ------------------------------------------------------------------ the network

@dataclass(frozen=True)
class BusParams:
    """AC network node with a shunt capacitor to ground.

    ``c``: capacitance, F (must be > 0); ``r_d``: series damping resistor, ohm.
    pu inputs use the system base.
    """

    c: float
    r_d: float = 0.0

    _quantities = {"c": "capacitance", "r_d": "resistance"}


@dataclass(frozen=True)
class BranchParams:
    """Series R-L branch between two buses; current positive from ``from_bus`` to ``to_bus``.

    ``l`` in H, ``r`` in ohm; pu inputs use the system base. ``breaker``: protection may open it.
    """

    from_bus: str
    to_bus: str
    l: float
    r: float = 0.0
    breaker: bool = False

    _quantities = {"l": "inductance", "r": "resistance"}
    _input_aliases = {"x_pu": "l_pu"}


@dataclass(frozen=True)
class SourceParams:
    """Three-phase emf behind a series R-L impedance, connected to ``bus``.

    ``l`` in H, ``r`` in ohm, ``v`` in phase-peak V (default 1 pu); pu inputs use the system base.
    """

    bus: str
    l: float
    r: float = 0.0
    v: float = 0.0  # emf magnitude, phase-peak V
    events: "SourceEventParams" = field(default_factory=lambda: SourceEventParams())

    _quantities = {"l": "inductance", "r": "resistance", "v": "voltage"}
    _input_aliases = {"x_pu": "l_pu"}
    _defaults_pu = {"v": 1.0}


@dataclass(frozen=True)
class SourceEventParams:
    """Events applied to one source; times in s, ``t < 0`` disables a step."""

    freq_step_t: float = -1.0
    freq_step_hz: float = 0.0
    phase_jump_t: float = -1.0
    phase_jump_rad: float = 0.0
    voltage_step_t: float = -1.0
    voltage_step: float = 0.0  # phase-peak emf magnitude after the step, V

    _quantities = {"voltage_step": "voltage"}
    _defaults_pu = {"voltage_step": 1.0}


# ------------------------------------------------------------------ a converter unit

@dataclass(frozen=True)
class ACFilterParams:
    """Series R-L branch of the converter's AC filter (``l_f`` H, ``r_f`` ohm).

    pu inputs use the unit's AC base. The filter capacitor is set on the bus (:class:`BusParams`).
    """

    l_f: float
    r_f: float = 0.0

    _quantities = {"l_f": "inductance", "r_f": "resistance"}
    _input_aliases = {"x_f_pu": "l_f_pu"}


@dataclass(frozen=True)
class DCCapacitorParams:
    """Capacitance (F) and series ESR (ohm); pu inputs use the unit DC base."""

    c: float
    r_esr: float = 0.0

    _quantities = {"c": "dc_capacitance", "r_esr": "dc_resistance"}


@dataclass(frozen=True)
class DCSourceParams:
    """DC supply; ``type`` selects which fields apply (pu inputs use the unit DC base)."""

    type: str = "none"  # "none" | "current" | "voltage"
    i: float = 0.0  # current source, A, positive into the DC link
    k: float = 0.0  # current-source voltage droop, A/V
    v: float = 0.0  # voltage-source emf, V
    r: float = 0.0  # voltage-source series resistance, ohm

    _choices = {"type": ("none", "current", "voltage")}

    _quantities = {"i": "dc_current", "k": "dc_current/dc_voltage",
                   "v": "dc_voltage", "r": "dc_resistance"}

    @staticmethod
    def _scaled_defaults(data: dict, scales: dict[str, float]) -> dict:
        """A voltage source's emf defaults to the rated dc voltage."""
        return {"v": scales["dc_voltage"]} if data.get("type") == "voltage" else {}


@dataclass(frozen=True)
class DCLinkParams:
    """DC link of one unit: rated voltage ``vdc_ref`` (V), optional capacitor and supply."""

    vdc_ref: float
    capacitor: Optional[DCCapacitorParams] = None
    source: DCSourceParams = field(default_factory=DCSourceParams)


@dataclass(frozen=True)
class PWMParams:
    """PWM settings: carrier, modulation method and duty-cycle update period.

    ``sync``: ``"asynchronous"`` (carrier on absolute time) or ``"synchronous"`` (carrier
    locked to the controller angle; requires an integer ``f_sw / base.f0``).
    """

    f_sw: float  # carrier frequency, Hz
    modulation_limit: float = 1.0
    carrier_phase: float = 0.0  # carrier position at t = 0, in carrier periods
    method: str = "spwm"  # "spwm" | "svpwm"
    sync: str = "asynchronous"  # "asynchronous" | "synchronous"
    update_period: Optional[float] = None  # s; default: one carrier period

    _choices = {"method": ("spwm", "svpwm"), "sync": ("asynchronous", "synchronous")}

    @property
    def switching_period(self) -> float:
        return 1.0 / self.f_sw

    @property
    def effective_update_period(self) -> float:
        """Return ``update_period`` (s), or one carrier period if it is not set."""
        return self.update_period if self.update_period is not None else self.switching_period


@dataclass(frozen=True)
class MeasurementParams:
    """ADC measurement settings.

    ``average`` (AC) and ``u_dc`` (DC): ``"instantaneous"`` or ``"window"`` (mean over
    ``window_s``, s, at most the PWM update period).
    """

    average: str = "instantaneous"
    window_s: Optional[float] = None  # s; default: the PWM update period
    u_dc: str = "instantaneous"

    _choices = {"average": ("instantaneous", "window"), "u_dc": ("instantaneous", "window")}


@dataclass(frozen=True)
class ReferenceParams:
    """Controller references, all electrical quantities in this unit's pu bases."""

    id_ref_pu: float = 0.0
    iq_ref_pu: float = 0.0
    p_ref_pu: float = 0.0
    q_ref_pu: float = 0.0
    v_ref_pu: float = 1.0
    vdc_ref_pu: float = 1.0
    theta: float = 0.0
    omega: Optional[float] = None  # rad/s; defaults to the system frequency
    zero_v_pu: float = 0.0

    _quantities = {"id_ref_pu": "current", "iq_ref_pu": "current", "p_ref_pu": "power",
                   "q_ref_pu": "power", "v_ref_pu": "voltage", "vdc_ref_pu": "dc_voltage",
                   "zero_v_pu": "voltage"}


@dataclass(frozen=True)
class ControlParams:
    """Control configuration: named loops, their signal connections and references.

    ``type`` selects the default wiring; ``connections`` and ``outputs`` override it.
    Each ``loops`` entry is validated against the schema registered for its ``type``.
    """

    type: str  # "gfl" | "gfm" | "custom"
    loops: dict[str, Any] = field(metadata={"entries": "loop"})  # each built with the parameters of its type
    connections: dict = field(default_factory=dict)  # input port -> output port
    outputs: dict = field(default_factory=dict)  # u_dq, theta, omega -> output port
    references: ReferenceParams = field(default_factory=ReferenceParams)
    sampling_period: Optional[float] = None  # ADC sampling period, s; default: pwm update period
    samples_per_update: int = field(init=False, default=1)

    _choices = {"type": ("gfl", "gfm", "custom")}


@dataclass(frozen=True)
class DelayParams:
    steps: int = 0  # one step is one PWM output update


@dataclass(frozen=True)
class ProtectionParams:
    """Trip and alarm thresholds; ``<= 0`` disables a criterion; times in s."""

    i_alarm_pu: float = -1.0  # instantaneous phase current, trips immediately
    vac_uv_pu: float = -1.0
    vac_ov_pu: float = -1.0
    freq_band_hz: float = -1.0
    vdc_band_pu: float = -1.0
    hold_s: float = 0.02
    rocof_alarm_hz_s: float = -1.0  # alarm only
    rocof_window_s: float = 0.1

    _quantities = {"i_alarm_pu": "current", "vac_uv_pu": "voltage",
                   "vac_ov_pu": "voltage", "vdc_band_pu": "dc_voltage"}


@dataclass(frozen=True)
class StartupParams:
    start: float = 0.0
    duration: float = 0.0  # s; ramp of the DC source and of id0


@dataclass(frozen=True)
class SetpointStepParams:
    """Step of a controller setpoint (GFM ``p_ref`` / ``q_ref`` / ``v_ref``)."""

    t: float = -1.0
    p_ref_pu: Optional[float] = None
    q_ref_pu: Optional[float] = None
    v_ref_pu: Optional[float] = None

    _quantities = {"p_ref_pu": "power", "q_ref_pu": "power", "v_ref_pu": "voltage"}


@dataclass(frozen=True)
class PLLGainStepParams:
    t: float = -1.0
    kp_after_pu: float = 0.0

    _quantities = {"kp_after_pu": "1/voltage"}


@dataclass(frozen=True)
class UnitEventParams:
    """Events applied to one converter unit."""

    startup: StartupParams = field(default_factory=StartupParams)
    setpoint: SetpointStepParams = field(default_factory=SetpointStepParams)
    pll_gain: PLLGainStepParams = field(default_factory=PLLGainStepParams)


@dataclass(frozen=True)
class UnitParams:
    """One converter unit: rating, plant components, control, protection and events.

    ``bus``: terminal bus. Plant fields are stored in SI; controller parameters are
    pu on the unit's rating ``s_base`` (VA) and DC base (``dclink.vdc_ref``).
    """

    bus: str
    ac_filter: ACFilterParams
    dclink: DCLinkParams
    pwm: PWMParams
    control: ControlParams
    delay: DelayParams = field(default_factory=DelayParams)
    s_base: Optional[float] = None  # rating (VA); default: the system base
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    protection: ProtectionParams = field(default_factory=ProtectionParams)
    events: UnitEventParams = field(default_factory=UnitEventParams)
    base: BaseValues = field(init=False, default=None)  # set by resolved()

    @staticmethod
    def _bases(data: dict, base: BaseValues, where: str) -> tuple[BaseValues, dict[str, float]]:
        """The unit's own base (its ``s_base``, default the system's) and its pu scales (dc: ``dclink.vdc_ref``)."""
        if data.get("s_base") is not None:
            base = _build(BaseValues, {"s_base": data["s_base"], "v_ll_rms": base.v_ll_rms, "f0": base.f0},
                          where, None, None)
        dc = data.get("dclink")
        if not isinstance(dc, dict) or "vdc_ref" not in dc:
            raise ConfigError(f"{where}.dclink.vdc_ref: required value missing")
        vdc_ref = _numeric_string(dc["vdc_ref"])
        if (isinstance(vdc_ref, bool) or not isinstance(vdc_ref, (int, float))
                or not math.isfinite(vdc_ref) or vdc_ref <= 0):
            raise ConfigError(f"{where}.dclink.vdc_ref must be finite and positive")
        return base, base.parameter_scales(vdc_ref)

    def resolved(self, system_base: BaseValues) -> UnitParams:
        """Return a copy with ``base``, timing defaults and ``omega`` filled in (after validation)."""
        pwm = replace(self.pwm, update_period=self.pwm.effective_update_period)
        references = self.control.references
        if references.omega is None:
            references = replace(references, omega=system_base.w0)
        control = replace(self.control, references=references)
        sampling_period = control.sampling_period if control.sampling_period is not None else pwm.update_period
        object.__setattr__(control, "samples_per_update",
                           max(1, int(round(pwm.update_period / sampling_period))))
        unit = replace(self, pwm=pwm, control=control)
        base = system_base if self.s_base is None else BaseValues(
            self.s_base, system_base.v_ll_rms, system_base.f0)
        object.__setattr__(unit, "base", base)
        return unit

    @property
    def dc_base(self) -> DCBase:
        """This unit's DC base (:class:`DCBase`)."""
        return self.base.dc(self.dclink.vdc_ref)


# ------------------------------------------------------------------ the run

@dataclass(frozen=True)
class SolverParams:
    """Integrator settings.

    ``subsystems`` (fixed-step only) maps subsystem names to a step relative to ``dt``
    (integer N, or 1/N) or to ``{step, method}``, e.g. ``{dclink: 10, network: 0.1}``.
    """

    type: str = "fixed"  # "fixed" | "adaptive"
    method: str = "rk4"  # fixed: euler|heun|rk4 ; adaptive: RK45|DOP853|Radau|BDF|LSODA|RK23|DP45
    dt: float = 1e-6  # s; fixed: maximum system step
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = math.inf
    warm_start: bool = True  # adaptive: reuse the last step size across intervals
    subsystems: dict = field(default_factory=dict)  # fixed only: {name: step relative to dt | {step, method}}
    sweeps: int = 2  # coupling sweeps for sub-stepped subsystems (>= 1)
    linearisations: int = 3  # linearisation points for the split error bound (0: none)

    _choices = {"type": ("fixed", "adaptive")}


@dataclass(frozen=True)
class LogParams:
    plant_period: float = 5e-4  # plant snapshot period, s
    control_every: int = 1  # keep every n-th controller sample


@dataclass(frozen=True)
class SimulationParams:
    t_end: float
    bridge: str = "switching"  # "switching" | "averaged" | "step_averaged"
    solver: SolverParams = field(default_factory=SolverParams)
    log: LogParams = field(default_factory=LogParams)
    stop_on_trip: bool = True
    progress_every: float = 0.0  # s of simulated time between progress lines; 0: silent
    energy_check: str = "warn"  # "warn" | "strict" | "off": verify the energy declarations before the run

    _choices = {"bridge": ("switching", "averaged", "step_averaged"), "energy_check": ("warn", "strict", "off")}


@dataclass(frozen=True)
class InitialParams:
    """Start time ``t`` (s) and overrides of initial state values.

    ``states`` maps dotted state names to a number, ``[re, im]`` or ``.re``/``.im`` entries,
    or a keyword: ``source`` (bus source voltage at ``t``) or ``rated`` (unit DC reference).
    ``t`` must lie on every unit's PWM update grid.
    """

    t: float = 0.0
    states: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputParams:
    """Files written by ``SimulationResult.save``."""

    states: bool = True  # states.csv
    signals: bool = False  # plant.csv and control.csv
    energy: bool = False  # energy.csv


@dataclass(frozen=True)
class Params:
    """Complete configuration: system base, network elements, units and run settings.

    ``buses``, ``branches``, ``sources`` and ``units`` map names to elements; names
    are unique across all four and prefix the state names.
    """

    base: BaseValues  # system base, used for network pu inputs
    buses: dict[str, BusParams]
    units: dict[str, UnitParams]
    simulation: SimulationParams
    sources: dict[str, SourceParams] = field(default_factory=dict)
    branches: dict[str, BranchParams] = field(default_factory=dict)
    initial: InitialParams = field(default_factory=InitialParams)
    output: OutputParams = field(default_factory=OutputParams)
    meta: dict = field(default_factory=dict)

    def unit(self, name: Optional[str] = None) -> UnitParams:
        """Return the unit ``name``, or the only unit if ``name`` is None."""
        if name is None:
            if len(self.units) != 1:
                raise ConfigError(f"this system has {len(self.units)} units {sorted(self.units)}: "
                                  f"name the one you mean")
            name = next(iter(self.units))
        if name not in self.units:
            raise ConfigError(f"unknown unit {name!r}; known: {sorted(self.units)}")
        return self.units[name]

    @property
    def elements(self) -> dict[str, Any]:
        """All named elements (buses, branches, sources, units) in assembly order."""
        return {**self.buses, **self.branches, **self.sources, **self.units}

    @staticmethod
    def _bases(data: dict, base: Any, where: str) -> tuple[Any, Optional[dict[str, float]]]:
        """The system base and its pu scales, for the network (``None`` without a base: reported as missing)."""
        if not isinstance(data.get("base"), dict):
            return None, None
        base = _build(BaseValues, data["base"], "base", None, None)
        return base, base.parameter_scales()

    def replace(self, **path_values: Any) -> "Params":
        """Return a copy with dotted paths replaced, e.g. ``replace(**{"units.vsc.control.loops.pll.kp_pu": 20})``."""
        d = to_dict(self)
        assigned = set()
        free = ("initial.states.", "simulation.solver.subsystems.")  # keys that contain dots themselves
        for path, value in path_values.items():
            head = next((h for h in free if path.startswith(h)), None)
            if head is not None:  # remainder is one key
                node = d
                for k in head.rstrip(".").split("."):
                    node = node.setdefault(k, {})
                node[path[len(head):]] = value
                continue
            node = d
            obj = self
            keys = path.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
                obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
            values = vars(obj) if hasattr(obj, "__dict__") else {}
            canonical = quantity_name(type(obj), keys[-1], values) if obj is not None else keys[-1]
            identity = (*keys[:-1], canonical)
            if identity in assigned:
                raise ConfigError(f"{path}: both actual and pu values specify the same parameter")
            assigned.add(identity)
            if canonical != keys[-1]:
                node.pop(canonical, None)
            node[keys[-1]] = value
        return from_dict(d)


# ------------------------------------------------------------------ construction from mappings

def from_dict(data: dict) -> Params:
    """Build, check and resolve a :class:`Params` tree from a mapping (the input is not modified)."""
    p = _build(Params, _normalize(data), "", None, None)
    validate(p)
    return replace(p, units={name: unit.resolved(p.base) for name, unit in p.units.items()})


def to_dict(p: Any) -> dict:
    """Convert a dataclass tree to mappings, omitting non-init (derived) fields."""
    if is_dataclass(p) and not isinstance(p, type):
        return {f.name: to_dict(getattr(p, f.name)) for f in fields(p) if f.init}
    if isinstance(p, dict):
        return {k: to_dict(v) for k, v in p.items()}
    return p


def _numeric_string(value: Any) -> Any:
    """Convert a numeric string to float; return other values unchanged."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _flatten_states(states: Any, prefix: str = "") -> dict:
    """Flatten nested state mappings to dotted keys; lists are kept as values."""
    if not isinstance(states, dict):
        raise ConfigError(f"initial.states{'.' + prefix if prefix else ''}: expected a mapping")
    out: dict = {}
    for key, value in states.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_states(value, name))
        else:
            out[name] = value
    return out


def _normalize(data: dict) -> dict:
    """A copy of the mapping with nested initial states flattened to dotted names and numeric strings read."""
    data = deepcopy(data)
    if not isinstance(data, dict):
        return data  # type error reported by the construction
    init = data.get("initial")
    if isinstance(init, dict) and init.get("states") is not None:
        init["states"] = {key: _numeric_string(value)
                          for key, value in _flatten_states(init["states"]).items()}
    simulation = data.get("simulation")
    solver = simulation.get("solver") if isinstance(simulation, dict) else None
    subsystems = solver.get("subsystems") if isinstance(solver, dict) else None
    if isinstance(subsystems, dict):
        for name, value in subsystems.items():
            if isinstance(value, dict):
                if "step" in value:
                    value["step"] = _numeric_string(value["step"])
            else:
                subsystems[name] = _numeric_string(value)
    return data


def _is_optional(tp) -> tuple[bool, Any]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _loop_class(data: Any, path: str) -> type:
    """The parameter class of a ``control.loops`` entry: that of its registered type."""
    kind = data.get("type") if isinstance(data, dict) else None
    if kind not in LOOP_TYPES:
        raise ConfigError(f"{path}.type: unknown loop type {kind!r}; known: {sorted(LOOP_TYPES)}")
    return LOOP_TYPES[kind].Params


def _build_named(item_cls, data: Any, path: str, base: Any, scales: Optional[dict]) -> dict:
    """Build a name -> entry mapping; names must be non-empty and contain no dots.

    ``item_cls``: the class of the entries, ``"loop"`` for loop entries, or anything else for raw values.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of names to entries, got {type(data).__name__}")
    out = {}
    for name, item in data.items():
        name = str(name)
        if not name or "." in name:
            raise ConfigError(f"{path}: {name!r} is not a usable name (non-empty, no '.')")
        where = f"{path}.{name}"
        if item_cls == "loop":
            out[name] = _build(_loop_class(item, where), item, where, base, scales)
        else:
            out[name] = _build(item_cls, item, where, base, scales) if is_dataclass(item_cls) else item
    return out


def _build(cls, data: Any, path: str, base: Any, scales: Optional[dict]):
    """Build the dataclass ``cls`` from a mapping at ``path``: convert its inputs with ``scales`` (SI
    values of one pu; ``None``: no conversion), then check its fields strictly and build its sections.

    A class with ``_bases(data, base, path)`` sets the base and scales of its sections.
    """
    if data is None:
        raise ConfigError(f"{path}: section is required")
    if is_dataclass(data):
        data = to_dict(data)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    if hasattr(cls, "_bases"):
        base, scales = cls._bases(data, base, path)
    if scales is not None:
        data = convert_quantities(data, cls, scales, path)
        if hasattr(cls, "_scaled_defaults"):
            data = {**cls._scaled_defaults(data, scales), **data}
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls) if f.init}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    kwargs = {}
    for f in fields(cls):
        if not f.init:
            continue
        key = f"{path}.{f.name}" if path else f.name
        tp = hints[f.name]
        optional, inner = _is_optional(tp)
        if f.name in data:
            value = data[f.name]
            if value is None:
                if not optional:
                    raise ConfigError(f"{key}: null is not allowed")
                kwargs[f.name] = None
            elif is_dataclass(inner) and isinstance(inner, type):
                kwargs[f.name] = _build(inner, value, key, base, scales)
            elif get_origin(inner) in (dict, _Mapping):
                kwargs[f.name] = _build_named(f.metadata.get("entries", get_args(inner)[1]), value, key, base, scales)
            elif inner is float:
                if isinstance(value, str):  # YAML 1.1 reads "2.0e6" as a string
                    try:
                        value = float(value)
                    except ValueError:
                        raise ConfigError(f"{key}: expected a number, got {value!r}") from None
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(f"{key}: expected a number, got {value!r}")
                kwargs[f.name] = float(value)
            elif inner is int:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                    raise ConfigError(f"{key}: expected an integer, got {value!r}")
                kwargs[f.name] = int(value)
            elif inner is bool:
                if isinstance(value, bool):
                    kwargs[f.name] = value
                elif isinstance(value, (int, float)) and value in (0, 1):
                    kwargs[f.name] = bool(value)
                else:
                    raise ConfigError(f"{key}: expected a boolean, got {value!r}")
            elif inner is str:
                kwargs[f.name] = str(value)
            elif inner is dict:
                kwargs[f.name] = dict(value)
            else:
                kwargs[f.name] = value
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            # an omitted section: built from nothing, so that its pu defaults are converted too
            kwargs[f.name] = (_build(inner, {}, key, base, scales) if is_dataclass(inner) and not optional
                              else f.default_factory())  # type: ignore[misc]
        else:
            raise ConfigError(f"{key}: required value missing")
    try:
        obj = cls(**kwargs)
    except ConfigError as exc:  # checks of the class itself
        raise ConfigError(f"{path}.{exc}" if path else str(exc)) from None
    for name, choices in getattr(cls, "_choices", {}).items():
        if getattr(obj, name) not in choices:
            raise ConfigError(f"{path or cls.__name__}.{name}: {getattr(obj, name)!r} not in {choices}")
    return obj


# ------------------------------------------------------------------ files

def load(path: str | Path, initial: str | Path | None = None,
         initial_time: Optional[float] = None, **overrides: Any) -> Params:
    """Load a YAML/JSON configuration, optionally replace its initial state, and apply overrides.

    ``initial``: states CSV (last row, or the row at ``initial_time`` in s) or YAML/JSON.
    ``overrides``: dotted paths, e.g. ``load("case.yaml", **{"simulation.t_end": 5.0})``.
    """
    p = from_dict(read_tree(path))
    if initial is not None:
        init = read_initial(initial, initial_time)
        merged = to_dict(p)
        merged["initial"] = init
        if merged["simulation"]["t_end"] <= init.get("t", 0.0):
            # keep the configured duration after the new start time
            merged["simulation"]["t_end"] = init.get("t", 0.0) + p.simulation.t_end
        p = from_dict(merged)
    if overrides:
        p = p.replace(**overrides)
    return p


def _read(path: Path) -> Any:
    """The content of a ``.json`` file, or of a YAML file."""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ConfigError(f"{path}: reading YAML needs PyYAML (pip install PyYAML), or write the "
                          f"configuration as .json") from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_initial(path: str | Path, t: Optional[float] = None) -> dict:
    """Read an ``initial`` mapping (``t``, ``states``) from YAML/JSON or a states CSV.

    CSV: last row, or the row at ``t`` (s); empty/NaN cells are omitted.
    YAML/JSON: the mapping itself or an ``initial`` block; ``t`` overrides its time.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2 or rows[0][0] != "t":
            raise ConfigError(f"{path}: not a states table (first column must be 't')")
        header, body = rows[0], rows[1:]
        if t is None:
            row = body[-1]
        else:
            times = [float(r[0]) for r in body]
            k = min(range(len(times)), key=lambda i: abs(times[i] - t))
            if abs(times[k] - t) > 1e-9 * max(1.0, abs(t)):
                raise ConfigError(f"{path}: no row at t = {t} (nearest: {times[k]})")
            row = body[k]
        states = {}
        for name, cell in zip(header[1:], row[1:]):
            if cell.strip() == "":
                continue
            value = float(cell)
            if not math.isnan(value):
                states[name] = value
        return {"t": float(row[0]), "states": states}
    data = _read(path)
    if isinstance(data, dict) and "initial" in data:
        data = data["initial"]
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an 'initial' mapping")
    if t is not None:
        data = {**data, "t": t}
    return data


def read_tree(path: str | Path) -> dict:
    """Return the top-level mapping of a ``.yaml``/``.yml``/``.json`` configuration file."""
    data = _read(Path(path))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def dump(p: Any, path: str | Path) -> None:
    """Write ``p`` as a configuration file (YAML if PyYAML is installed, else JSON)."""
    path = Path(path)
    d = to_dict(p)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    except ImportError:  # pragma: no cover
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
