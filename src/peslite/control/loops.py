"""What each control loop computes: the loop types, their typed signals and their registry.

A loop type is one class: its parameters (``Params``, the schema of its ``ctrl.loops`` entry),
its typed input and output ports, its role in the default wiring, and its discrete-time update.
How loops are connected is not theirs to know (:mod:`.controller`).
Signals are pu; time s, angles rad, frequencies rad/s.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, is_dataclass
from typing import Any, ClassVar, Mapping, Optional

from ..solver.model import ConfigError, assign, gather, scatter
from .blocks import HighPass1, LowPass1, clamp

__all__ = ["SignalType", "V_AB", "I_AB", "V_DQ", "VOLTAGE", "DC_VOLTAGE", "CURRENT", "ANGLE", "FREQUENCY",
           "POWER_PU", "ROLES", "Loop", "LOOP_TYPES", "register_loop_type", "PowerFilterParams",
           "CurrentLimitParams",
           "SRFPLL", "CurrentLoop", "DCVoltageLoop", "PowerLoop", "SyncLaw", "PSC", "Droop", "VSG", "DVOC",
           "Matching", "VirtualImpedance", "VirtualAdmittance", "ActiveDamping", "UnitDelay"]


# ------------------------------------------------------------------ signals

@dataclass(frozen=True)
class SignalType:
    unit: str
    frame: str = "scalar"
    complex_value: bool = False


V_AB = SignalType("pu_voltage_ac", "alpha_beta", True)
I_AB = SignalType("pu_current_ac", "alpha_beta", True)
V_DQ = SignalType("pu_voltage_ac", "dq", True)
VOLTAGE = SignalType("pu_voltage_ac")
DC_VOLTAGE = SignalType("pu_voltage_dc")
CURRENT = SignalType("pu_current_ac")
ANGLE = SignalType("rad")
FREQUENCY = SignalType("rad/s")
POWER_PU = SignalType("pu_power")


# ------------------------------------------------------------------ the base class and the registry

# the parts a loop can play in the default gfl/gfm wiring (see peslite.control.graph.default_wiring)
ROLES = ("pll", "sync", "current", "dc_voltage", "power", "impedance", "admittance", "damping")


class Loop:
    """A control loop: a discrete-time block with typed ports, built as ``cls(cfg, unit, startup)``.

    A loop type sets ``type`` (its name in ``ctrl.loops``), ``Params`` (a frozen dataclass with
    ``type`` and ``period`` fields; its ``_quantities`` name the fields given in SI or pu),
    ``inputs`` / ``outputs`` (port name -> :class:`SignalType`), ``delayed`` (inputs read after the
    update by ``latch(inputs)``, which break instantaneous cycles) and ``role`` (one of
    :data:`ROLES`, or ``None``: wired only by ``ctrl.connections``). With ``outputs_from_state``
    its held outputs are set from its states (``initial_outputs()``) when states are loaded.
    It provides ``initial_outputs()`` and ``update(t, inputs) -> outputs``; its named states are the
    attributes in ``state_names`` unless it overrides ``get_state`` / ``set_state``. A loop rebuilt
    after retuning continues from those states and from attributes listed in ``carried``.
    ``cfg``: the loop's parameters; ``unit``: the unit's parameters; ``startup``: the controller's
    current run/ramp state.
    """

    type: ClassVar[str]
    Params: ClassVar[type]
    inputs: ClassVar[Mapping[str, SignalType]] = {}
    outputs: ClassVar[Mapping[str, SignalType]] = {}
    delayed: ClassVar[frozenset[str]] = frozenset()
    role: ClassVar[Optional[str]] = None
    outputs_from_state: ClassVar[bool] = False  # held outputs = initial_outputs() after loading states
    state_names: tuple[str, ...] | Mapping[str, str] = ()
    carried: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: Any, unit: Any, startup: Any) -> None:
        self.cfg, self.unit, self.startup = cfg, unit, startup

    def initial_outputs(self) -> dict[str, Any]:
        raise NotImplementedError

    def update(self, t: float, inputs: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        names = self.state_names
        attrs = names if isinstance(names, Mapping) else {name: name for name in names}
        return {name: getattr(self, attr) for name, attr in attrs.items()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self.state_names)

    def retuned(self, old: "Loop") -> None:
        """Take non-state runtime attributes from the loop instance being replaced."""
        for name in self.carried:
            setattr(self, name, getattr(old, name))


LOOP_TYPES: dict[str, type] = {}


def register_loop_type(cls: type) -> type:
    """Register the loop type ``cls`` (see :class:`Loop`) under ``cls.type``, so that simulation files
    can use it; its parameters are then checked like those of the built-in loops. Returns ``cls``,
    so it serves as a class decorator."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("a loop type needs a name: the class attribute 'type'")
    if name in LOOP_TYPES:
        raise ValueError(f"loop type {name!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or not {"type", "period"} <= set(params.__dataclass_fields__):
        raise TypeError(f"loop type {name!r}: Params must be a dataclass with type and period")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"loop type {name!r}: Params.type must default to {name!r}")
    if cls.role is not None and cls.role not in ROLES:
        raise ValueError(f"loop type {name!r}: role {cls.role!r} is not one of {ROLES}")
    LOOP_TYPES[name] = cls
    return cls


def _into(frame: float):
    """The rotation from alpha-beta into the dq frame of angle ``frame`` (rad)."""
    return cmath.exp(-1j * frame)


@dataclass(frozen=True)
class PowerFilterParams:
    """Optional first-order low-pass filter on measured active and reactive power."""

    enable: bool = False
    bandwidth: Optional[float] = None  # Hz


@dataclass(frozen=True)
class CurrentLimitParams:
    """Optional current-reference magnitude limit."""

    enable: bool = False
    limit_pu: Optional[float] = None

    _quantities = {"limit_pu": "current"}


# ------------------------------------------------------------------ grid following

@register_loop_type
class SRFPLL(Loop):
    """SRF-PLL: ``omega = w0 + kp*eps + ki*integral``, ``theta += T*omega``.

    Input: the voltage, rotated into the frame of the previous angle. ``normalisation``:
    ``"rated"`` (eps = v_q) or ``"amplitude"`` (eps = v_q / u_g, u_g a filtered v_d estimate).
    Named states: ``theta`` (rad, unwrapped), ``integral`` (pu*s), ``u_g`` (pu, amplitude only).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp_pu: float
        ki_pu: float
        normalisation: str = "rated"
        period: Optional[float] = None
        type: str = "srf_pll"

        _choices = {"normalisation": ("rated", "amplitude")}
        _quantities = {"kp_pu": "1/voltage", "ki_pu": "1/voltage"}

    type = "srf_pll"
    role = "pll"
    outputs_from_state = True
    inputs = {"v": V_AB}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.kp, self.ki, self.w0, self.T = cfg.kp_pu, cfg.ki_pu, unit.base.w0, cfg.period
        self.theta, self.omega, self.integral = 0.0, self.w0, 0.0
        self.u_g = 1.0 if cfg.normalisation == "amplitude" else None
        self.state_names = {"theta": "theta", "integral_pu": "integral",
                            **({} if self.u_g is None else {"u_g_pu": "u_g"})}

    def initial_outputs(self):
        return {"theta": self.theta, "frame": self.theta, "omega": self.omega}

    def update(self, t, inputs):
        frame = self.theta
        v = inputs["v"] * _into(frame)
        u_g = self.u_g
        if u_g is None:                                    # rated
            eps = v.imag
        else:                                              # amplitude
            eps = v.imag / u_g if u_g > 0.0 else 0.0
        self.integral += self.T * eps
        self.omega = self.w0 + self.kp * eps + self.ki * self.integral
        self.theta += self.T * self.omega
        if u_g is not None:
            self.u_g = u_g + self.T * self.kp * (v.real - u_g)
        return {"theta": self.theta, "frame": frame, "omega": self.omega}


@register_loop_type
class CurrentLoop(Loop):
    """dq PI current loop: ``u = u_ff + (r_pu + j*omega/w0*x_pu)*i + kp*e + ki*integral`` (+ ``extra``).

    Gains default from ``bandwidth`` (Hz) and the unit's filter. Anti-windup: the
    firmware uses ``rollback`` (conditional) or ``backcalculate`` when the modulation saturates.
    Named state: ``integral_pu`` (complex dq, pu*s).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bandwidth: float  # Hz
        kp_pu: Optional[float] = None  # pu voltage / pu current
        ki_pu: Optional[float] = None  # pu voltage / (pu current * s)
        decoupling: bool = True
        feedforward: bool = True
        antiwindup: str = "conditional"  # "conditional" | "backcalc" | "none"
        period: Optional[float] = None
        type: str = "dq_current_pi"

        _choices = {"antiwindup": ("conditional", "backcalc", "none")}
        _quantities = {"kp_pu": "resistance", "ki_pu": "resistance"}

    type = "dq_current_pi"
    role = "current"
    inputs = {"id_ref": CURRENT, "iq_ref": CURRENT, "v": V_AB, "i": I_AB, "frame": ANGLE, "omega": FREQUENCY,
              "extra": V_DQ}
    outputs = {"u_dq": V_DQ, "extra": V_DQ}
    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        x = unit.ac_filter.l_f * unit.base.w0 / unit.base.z_base
        r = unit.ac_filter.r_f / unit.base.z_base
        self.kp = cfg.kp_pu if cfg.kp_pu is not None else x / unit.base.w0 * 2 * math.pi * cfg.bandwidth
        self.ki = cfg.ki_pu if cfg.ki_pu is not None else r * 2 * math.pi * cfg.bandwidth
        self.x_pu, self.r_pu, self.w0, self.T = x, r, unit.base.w0, cfg.period
        self.decoupling, self.feedforward, self.antiwindup = cfg.decoupling, cfg.feedforward, cfg.antiwindup
        self.integral = self._prev_integral = 0j
        self._last: tuple[complex, complex, complex, float] | None = None
        self.extra = 0j

    def initial_outputs(self):
        return {"u_dq": 0j, "extra": 0j}

    def update(self, t, inputs):
        rot = _into(inputs["frame"])
        self.extra = inputs["extra"]
        i, u_ff, omega = inputs["i"] * rot, inputs["v"] * rot, inputs["omega"]
        self._prev_integral = self.integral
        if not self.startup.active:
            e = 0j
        else:
            e = complex(inputs["id_ref"], inputs["iq_ref"]) - i
            self.integral = self.integral + self.T * e
        self._last = (e, i, u_ff, omega)
        return {"u_dq": self.command(e, i, u_ff, omega) + self.extra, "extra": self.extra}

    def command(self, e: complex, i: complex, u_ff: complex, omega: float) -> complex:
        """Voltage command (without ``extra``) from the integrator state (no integration)."""
        u = self.kp * e + self.ki * self.integral
        if self.feedforward:
            u += u_ff
        if self.decoupling:
            u += (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i
        return u

    def rollback(self) -> complex:
        """Undo the last integration and return the recomputed command (without ``extra``)."""
        self.integral = self._prev_integral
        assert self._last is not None
        return self.command(*self._last)

    def backcalculate(self, u_cmd: complex, u_limited: complex) -> None:
        if self.kp != 0.0:
            self.integral = self.integral + self.T * (u_limited - u_cmd) / self.kp

    def get_state(self):
        return {"integral_pu": self.integral}

    def set_state(self, values):
        unknown = set(values) - {"integral_pu"}
        if unknown:
            raise KeyError(f"dq_current_pi has no state(s) {sorted(unknown)}")
        if "integral_pu" in values:
            self.integral = self._prev_integral = complex(values["integral_pu"])


@register_loop_type
class DCVoltageLoop(Loop):
    """PI dc-voltage loop returning ``id_ref = ff + kp*e + ki*int(e)`` clamped to ``[floor, limit]``.

    ``e = u_dc - vdc_ref`` in dc pu; ``id_ref`` in ac current pu (positive = export); the
    feed-forward ``id0_export_pu`` follows the connection ramp. ``frozen`` (set while tripped) zeroes
    the output and holds the integrator. Named states: ``integral`` (pu*s) and, with conditional
    anti-windup, ``clamped``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp_pu: float
        ki_pu: float
        id0_export_pu: float = 0.0  # d-axis current feed-forward, pu
        bidirectional: bool = False  # False: id_ref limited to >= 0
        period: Optional[float] = None
        limit_pu: float = 1.0
        antiwindup: str = "none"
        type: str = "dc_voltage_pi"

        _choices = {"antiwindup": ("none", "conditional")}
        _quantities = {"kp_pu": "current/dc_voltage", "ki_pu": "current/dc_voltage",
                       "id0_export_pu": "current", "limit_pu": "current"}

    type = "dc_voltage_pi"
    role = "dc_voltage"
    inputs = {"u_dc": DC_VOLTAGE, "vdc_ref": DC_VOLTAGE}
    outputs = {"id_ref": CURRENT}
    carried = ("n_updates", "n_clamped", "n_reverse", "first_clamp_t")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.kp, self.ki, self.T = cfg.kp_pu, cfg.ki_pu, cfg.period
        self.limit = cfg.limit_pu
        self.floor = -cfg.limit_pu if cfg.bidirectional else 0.0
        self.antiwindup = cfg.antiwindup
        self.integral = self.error = self.raw = 0.0
        self.clamped = False
        self.n_updates = self.n_clamped = 0
        self.n_reverse = 0  # updates asking for reverse (import) current
        self.first_clamp_t: float | None = None
        self.state_names = {"integral_pu": "integral",
                            **({"clamped": "clamped"} if cfg.antiwindup == "conditional" else {})}

    def initial_outputs(self):
        return {"id_ref": 0.0}

    def update(self, t, inputs):
        self.n_updates += 1
        ff = self.startup.value * self.cfg.id0_export_pu
        self.error = inputs["u_dc"] - inputs["vdc_ref"]
        if not self.startup.active:
            self.raw = 0.0
        else:
            if not (self.antiwindup == "conditional" and self.clamped):
                self.integral += self.T * self.error
            self.raw = ff + self.kp * self.error + self.ki * self.integral
        id_ref = clamp(self.raw, self.floor, self.limit)
        self.clamped = id_ref != self.raw
        if self.clamped:
            self.n_clamped += 1
            if self.first_clamp_t is None:
                self.first_clamp_t = t
        if self.raw < 0.0:
            self.n_reverse += 1
        return {"id_ref": id_ref}


# ------------------------------------------------------------------ grid forming

@register_loop_type
class PowerLoop(Loop):
    """Power feedback ``p + j q = v conj(i)``, optionally low-pass filtered.

    Named states: filtered ``p`` and ``q`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        filter: PowerFilterParams = field(default_factory=PowerFilterParams)
        type: str = "power"

    type = "power"
    role = "power"
    inputs = {"v": V_AB, "i": I_AB}
    outputs = {"p": POWER_PU, "q": POWER_PU}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        bandwidth = cfg.filter.bandwidth if cfg.filter.enable else 0.0
        self.lpf_p = LowPass1(bandwidth, cfg.period, 0.0, init_on_first=False)
        self.lpf_q = LowPass1(bandwidth, cfg.period, 0.0, init_on_first=False)

    def initial_outputs(self):
        return {"p": 0.0, "q": 0.0}

    def update(self, t, inputs):
        s = inputs["v"] * inputs["i"].conjugate()
        return {"p": self.lpf_p.update(s.real), "q": self.lpf_q.update(s.imag)}

    def get_state(self):
        return gather({"p_pu": self.lpf_p, "q_pu": self.lpf_q})

    def set_state(self, values):
        scatter({"p_pu": self.lpf_p, "q_pu": self.lpf_q}, values)


class SyncLaw(Loop):
    """Base of the grid-forming synchronization laws: angle (rad), frequency (rad/s) and voltage
    magnitude reference (pu) from the power feedback, the setpoints and the measured voltage.

    A law implements ``step(T, p, q, v_mag, v_dc, p_ref, q_ref, v_ref, i_dq)``, which advances
    ``theta``, ``omega`` and ``v_mag`` by one period ``T``; its inputs are in pu (the voltage
    magnitude on the ac base, ``v_dc`` on the dc base) and ``i_dq`` is the current in the law's frame.
    The active-power setpoint follows the controller's start-up ramp. Before it runs, a grid-forming
    law follows the measured terminal voltage so that it starts in phase with the grid.
    """

    role = "sync"
    outputs_from_state = True
    inputs = {"p": POWER_PU, "q": POWER_PU, "v": V_AB, "i": I_AB, "u_dc": DC_VOLTAGE,
              "p_ref": POWER_PU, "q_ref": POWER_PU, "v_ref": VOLTAGE}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY, "v_ref": VOLTAGE}
    state_names = ("theta",)

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.w0, self.T = unit.base.w0, cfg.period
        self.theta, self.omega = 0.0, self.w0
        self.v_mag = unit.ctrl.references.v_ref_pu

    def initial_outputs(self):
        return {"theta": float(self.theta), "frame": float(self.theta), "omega": float(self.omega),
                "v_ref": self.unit.ctrl.references.v_ref_pu}

    def update(self, t, inputs):
        if not self.startup.active:
            out = self.track(inputs)
            if out is not None:
                return out
            return {"theta": self.theta, "frame": self.theta, "omega": self.omega,
                    "v_ref": self.v_mag}
        frame = float(self.theta)
        rot = _into(frame)
        v, i = inputs["v"] * rot, inputs["i"] * rot
        p_ref = self.startup.value * inputs["p_ref"]
        q_ref, v_ref = inputs["q_ref"], inputs["v_ref"]
        self.step(self.T, inputs["p"], inputs["q"], abs(v), inputs["u_dc"], p_ref, q_ref, v_ref, i)
        return {"theta": self.theta, "frame": frame, "omega": self.omega, "v_ref": self.v_mag}

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq) -> None:
        raise NotImplementedError

    def track(self, inputs):
        """Hold the law on a usable measured voltage before start-up."""
        voltage = inputs["v"]
        if abs(voltage) <= 0.1:
            return None
        theta = float(self.theta)
        theta += (cmath.phase(voltage) - theta + math.pi) % (2.0 * math.pi) - math.pi
        self.follow(theta, abs(voltage), inputs["v_ref"])
        return {"theta": theta, "frame": theta, "omega": self.omega, "v_ref": self.v_mag}

    def follow(self, theta: float, v_mag_pu: float, v_ref_pu: float) -> None:
        self.theta, self.omega, self.v_mag = theta, self.w0, v_mag_pu


@register_loop_type
class PSC(SyncLaw):
    """Power-synchronization control ``w = w0 + k_p (P* - P)`` with a PI voltage-magnitude loop.

    Named states: ``theta`` (rad), ``v_int`` (pu*s).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        k_p_pu: float  # rad/s per pu
        k_v: float = 0.0  # AVR proportional gain, pu/pu (0: open-loop magnitude)
        k_vi: float = 0.0  # AVR integral gain, 1/s
        type: str = "psc"

        _quantities = {"k_p_pu": "1/power"}

    type = "psc"
    state_names = {"theta": "theta", "v_int_pu": "v_int"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.k_p, self.k_v, self.k_vi = cfg.k_p_pu, cfg.k_v, cfg.k_vi
        self.v_int = 0.0

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.k_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        e_v = v_ref_pu - v_mag_pu
        self.v_int += T * e_v
        self.v_mag = v_ref_pu + self.k_v * e_v + self.k_vi * self.v_int

    def follow(self, theta, v_mag_pu, v_ref_pu):
        super().follow(theta, v_mag_pu, v_ref_pu)
        error = v_ref_pu - v_mag_pu
        self.v_int = ((v_mag_pu - v_ref_pu - self.k_v * error) / self.k_vi
                      if self.k_vi > 0.0 else 0.0)


@register_loop_type
class Droop(SyncLaw):
    """P-f / Q-V droop control. Named state: ``theta`` (rad)."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        m_p_pu: float  # rad/s per pu (P-f droop)
        n_q_pu: float  # pu/pu (Q-V droop)
        period: Optional[float] = None
        type: str = "droop"

        _quantities = {"m_p_pu": "1/power", "n_q_pu": "voltage/power"}

    type = "droop"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.m_p, self.n_q = cfg.m_p_pu, cfg.n_q_pu

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.m_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.n_q * (q_ref_pu - q_pu)


@register_loop_type
class VSG(SyncLaw):
    """Virtual synchronous generator: ``2H dw/dt = p* - p - d_p (w - w0)/w0`` and a Q-V droop.

    Named states: ``theta`` (rad), ``dw_pu`` and, if ``t_q > 0``, ``v_mag`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        h: float  # inertia constant, s
        d_p_pu: float  # damping, pu power per pu speed deviation
        k_q_pu: float  # Q-V droop, pu/pu
        t_q: float = 0.0  # voltage-magnitude lag time constant, s (0: none)
        period: Optional[float] = None
        type: str = "vsg"

        _quantities = {"d_p_pu": "power/frequency", "k_q_pu": "voltage/power"}

    type = "vsg"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.h, self.d_p, self.k_q, self.t_q = cfg.h, cfg.d_p_pu, cfg.k_q_pu, cfg.t_q
        self.dw_pu = 0.0
        self.state_names = {"theta": "theta", "dw_pu": "dw_pu",
                            **({"v_mag_pu": "v_mag"} if self.t_q > 0.0 else {})}

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.dw_pu += T / (2.0 * self.h) * (p_ref_pu - p_pu - self.d_p * self.dw_pu)
        self.omega = self.w0 * (1.0 + self.dw_pu)
        self.theta += T * self.omega
        v_cmd = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        if self.t_q > 0.0:
            self.v_mag += T / self.t_q * (v_cmd - self.v_mag)
        else:
            self.v_mag = v_cmd

    def follow(self, theta, v_mag_pu, v_ref_pu):
        super().follow(theta, v_mag_pu, v_ref_pu)
        self.dw_pu = 0.0


@register_loop_type
class DVOC(SyncLaw):
    """Dispatchable virtual oscillator control, integrated in polar form.

    ``dv/dt = j w0 v + eta e^{j kappa} ((p* - j q*) v / V*^2 - i) + eta alpha (1 - |v|^2 / V*^2) v``
    with oscillator voltage ``v`` and current ``i`` in pu. Named states: ``theta`` (rad), ``v_mag`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        eta_pu: float  # rad/s
        alpha_pu: float  # magnitude regulation gain
        kappa: float  # rotation of the current error, rad
        period: Optional[float] = None
        type: str = "dvoc"

        _quantities = {"eta_pu": "resistance", "alpha_pu": "1/resistance"}

    type = "dvoc"
    state_names = {"theta": "theta", "v_mag_pu": "v_mag"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.eta, self.alpha = cfg.eta_pu, cfg.alpha_pu
        self.rot = cmath.exp(1j * cfg.kappa)

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        V = self.v_mag
        i_star = complex(p_ref_pu, -q_ref_pu) * V / (v_ref_pu * v_ref_pu)
        w = self.eta * self.rot * (i_star - i_dq) + self.eta * self.alpha * (1.0 - V * V / (v_ref_pu * v_ref_pu)) * V
        self.omega = self.w0 + w.imag / V
        self.v_mag = V + T * w.real
        self.theta += T * self.omega


@register_loop_type
class Matching(SyncLaw):
    """Matching control ``w = k_theta * v_dc``; ``k_theta_pu`` defaults to ``w0 / vdc_ref_pu``.

    Named state: ``theta`` (rad).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        k_theta_pu: Optional[float] = None  # rad/s per pu dc voltage (dc base)
        k_q_pu: float = 0.0  # Q-V droop, pu/pu
        period: Optional[float] = None
        type: str = "matching"

        _quantities = {"k_theta_pu": "1/dc_voltage", "k_q_pu": "voltage/power"}

    type = "matching"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        vdc_ref_pu = unit.ctrl.references.vdc_ref_pu
        self.k_theta = cfg.k_theta_pu if cfg.k_theta_pu is not None else self.w0 / vdc_ref_pu
        self.k_q = cfg.k_q_pu

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.k_theta * v_dc_pu
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.k_q * (q_ref_pu - q_pu)


# ------------------------------------------------------------------ voltage-reference shaping

@register_loop_type
class VirtualImpedance(Loop):
    """Voltage command ``v_ref - (r_pu + j omega/w0 x_pu) i`` (+ ``extra``) behind a virtual impedance.

    Without a ``period`` it runs whenever the loops before it update.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        r_v_pu: float = 0.0
        x_v_pu: float = 0.0
        period: Optional[float] = None  # None: runs on upstream updates
        type: str = "virtual_impedance"

        _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance"}

    type = "virtual_impedance"
    role = "impedance"
    inputs = {"v_ref": VOLTAGE, "i": I_AB, "frame": ANGLE, "omega": FREQUENCY, "extra": V_DQ}
    outputs = {"u_dq": V_DQ}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_pu, self.x_pu, self.w0 = cfg.r_v_pu, cfg.x_v_pu, unit.base.w0

    def initial_outputs(self):
        return {"u_dq": 0j}

    def update(self, t, inputs):
        i = inputs["i"] * _into(inputs["frame"])
        drop = (self.r_pu + 1j * inputs["omega"] / self.w0 * self.x_pu) * i
        return {"u_dq": complex(inputs["v_ref"], 0.0) - drop + inputs["extra"]}


@register_loop_type
class VirtualAdmittance(Loop):
    """Current reference from ``x_pu/w0 di/dt = v_ref - v - (r_pu + j omega/w0 x_pu) i``,
    optionally magnitude-limited. Named state: ``i_ref_pu`` (dq, pu)."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        x_v_pu: float
        r_v_pu: float = 0.0
        current_limit: CurrentLimitParams = field(default_factory=CurrentLimitParams)
        type: str = "virtual_admittance"

        _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance"}

        def __post_init__(self) -> None:
            if not self.x_v_pu > 0.0:
                raise ConfigError("x_v_pu must be positive")

    type = "virtual_admittance"
    role = "admittance"
    inputs = {"v_ref": VOLTAGE, "v": V_AB, "frame": ANGLE, "omega": FREQUENCY}
    outputs = {"id_ref": CURRENT, "iq_ref": CURRENT}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_pu, self.x_pu, self.w0, self.T = cfg.r_v_pu, cfg.x_v_pu, unit.base.w0, cfg.period
        self.i_limit = cfg.current_limit.limit_pu if cfg.current_limit.enable else math.inf
        self.i_ref = 0j

    def initial_outputs(self):
        return {"id_ref": 0.0, "iq_ref": 0.0}

    def update(self, t, inputs):
        v_ref_dq, v_dq, omega = complex(inputs["v_ref"], 0.0), inputs["v"] * _into(inputs["frame"]), inputs["omega"]
        i_ref = self.i_ref + self.T * self.w0 / self.x_pu * (
            v_ref_dq - v_dq - (self.r_pu + 1j * omega / self.w0 * self.x_pu) * self.i_ref)
        mag = abs(i_ref)
        if mag > self.i_limit:
            i_ref *= self.i_limit / mag
        self.i_ref = i_ref
        return {"id_ref": i_ref.real, "iq_ref": i_ref.imag}

    def get_state(self):
        return {"i_ref_pu": self.i_ref}

    def set_state(self, values):
        if set(values) - {"i_ref_pu"}:
            raise KeyError("virtual_admittance has the state i_ref_pu (current pu)")
        if "i_ref_pu" in values:
            self.i_ref = complex(values["i_ref_pu"])


@register_loop_type
class ActiveDamping(Loop):
    """Damping voltage ``-r_a`` times the current high-pass filtered at ``alpha_d`` (rad/s).

    Named state: ``hpf_pu`` (the low-passed current, pu; starts at zero).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        r_a_pu: float
        alpha_d: float
        type: str = "active_damping"

        _quantities = {"r_a_pu": "resistance"}

    type = "active_damping"
    role = "damping"
    inputs = {"i": I_AB, "frame": ANGLE}
    outputs = {"extra": V_DQ}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_a = cfg.r_a_pu
        self.hpf = HighPass1(cfg.alpha_d / (2.0 * math.pi), cfg.period, init_on_first=False, y0=0j)

    def initial_outputs(self):
        return {"extra": 0j}

    def update(self, t, inputs):
        return {"extra": -(self.r_a * self.hpf.update(inputs["i"] * _into(inputs["frame"])))}

    def get_state(self):
        return gather({"hpf_pu": self.hpf})

    def set_state(self, values):
        scatter({"hpf_pu": self.hpf}, values)


# ------------------------------------------------------------------ delay

_DELAY_SIGNALS = {"current_pu": CURRENT, "voltage_pu": VOLTAGE, "dc_voltage_pu": DC_VOLTAGE, "voltage_dq_pu": V_DQ,
                  "voltage_ab_pu": V_AB, "current_ab_pu": I_AB, "angle": ANGLE, "frequency": FREQUENCY,
                  "power_pu": POWER_PU}


@register_loop_type
class UnitDelay(Loop):
    """One-period delay of a signal of type ``signal``: breaks an instantaneous cycle of the wiring.

    Named state: ``value_pu`` (``value`` for an angle or frequency).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        initial: Optional[float] = None  # angle (rad) or frequency (rad/s)
        initial_pu: Optional[float] = None  # electrical signals
        signal: str = "current_pu"
        type: str = "unit_delay"

        _choices = {"signal": tuple(_DELAY_SIGNALS)}

        @staticmethod
        def _signal_quantities(values):
            scale = {"current_pu": "current", "current_ab_pu": "current",
                     "voltage_pu": "voltage", "voltage_dq_pu": "voltage", "voltage_ab_pu": "voltage",
                     "dc_voltage_pu": "dc_voltage", "power_pu": "power"}.get(values.get("signal", "current_pu"))
            return {"initial_pu": scale} if scale else {}

        def __post_init__(self) -> None:
            if self.initial_pu is not None and not self.signal.endswith("_pu"):
                raise ConfigError("initial_pu: angle/frequency delays use initial in rad or rad/s")

    type = "unit_delay"
    delayed = frozenset({"value"})

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.inputs = self.outputs = {"value": _DELAY_SIGNALS[cfg.signal]}
        electrical = cfg.signal.endswith("_pu")
        initial = cfg.initial_pu if electrical else cfg.initial
        self.value = 0.0 if initial is None else initial
        self.state_name = "value_pu" if electrical else "value"

    def initial_outputs(self):
        return {"value": self.value}

    def update(self, t, inputs):
        return {"value": self.value}

    def latch(self, inputs):
        self.value = inputs["value"]

    def get_state(self):
        return {self.state_name: self.value}

    def set_state(self, values):
        if set(values) - {self.state_name}:
            raise KeyError(f"unit_delay has the state {self.state_name}")
        self.value = values.get(self.state_name, self.value)
