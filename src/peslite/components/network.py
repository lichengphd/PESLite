"""AC network blocks and registered circuit-element types.

Quantities are SI; ac quantities are peak-scaled space vectors. Breakers expose
``open_breaker()`` and ``close_breaker()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, ClassVar, Optional

from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, ConfigError, Empty

__all__ = [
    "ThreePhaseSource", "RLBranch", "RCNode", "Element", "ELEMENT_TYPES",
    "register_element_type", "element_buses", "Load",
]


class _BranchState(Bag):
    __slots__ = ("i",)

class _BranchInp(Bag):
    __slots__ = ("u_from", "u_to")

class _BranchOut(Bag):
    __slots__ = ("i",)

class RLBranch:
    """Series R-L branch; an open breaker zeroes and holds its current."""

    state_names: ClassVar[tuple[str, ...]] = ("i",)
    outputs_need_inputs: ClassVar[bool] = False
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("inp.u_from", "out.i", 1.5, 1.0),
                                              PowerPort("inp.u_to", "out.i", 1.5, -1.0))

    def __init__(self, L: float, R: float, i0: complex = 0j) -> None:
        self.L, self.R = L, R
        self.storage = (StoragePort("i", L, 1.5, (("inp.u_from", 1.0), ("inp.u_to", -1.0)), "inductor"),)
        self.state = _BranchState(i=complex(i0))
        self.inp = _BranchInp(u_from=0j, u_to=0j)
        self.out = _BranchOut(i=complex(i0))
        self.breaker_open = False
        if type(self) is RLBranch:
            self.rhs = self._rhs_closed

    def open_breaker(self) -> None:
        self.breaker_open = True
        self.state.i = 0j
        if type(self) is RLBranch:
            self.rhs = self._rhs_open

    def close_breaker(self) -> None:
        """Close the breaker; current resumes from zero after an opening."""
        self.breaker_open = False
        if type(self) is RLBranch:
            self.rhs = self._rhs_closed

    def retune(self, L: float, R: float) -> None:
        """Change inductance and resistance without changing the current state."""
        self.L, self.R = L, R
        self.storage = (replace(self.storage[0], value=L),)

    def set_outputs(self, t: float) -> None:
        self.out.i = self.state.i

    def rhs(self, t: float):
        if self.breaker_open:
            return (0j,)
        return ((self.inp.u_from - self.inp.u_to - self.R * self.state.i) / self.L,)

    def _rhs_open(self, t: float):
        return (0j,)

    def _rhs_closed(self, t: float):
        return ((self.inp.u_from - self.inp.u_to - self.R * self.state.i) / self.L,)

    def dissipated_power(self) -> float:
        return 1.5 * self.R * abs(self.state.i) ** 2

    def supplied_power(self) -> float:
        return 0.0

class _NodeState(Bag):
    __slots__ = ("u_C",)

class _NodeInp(Bag):
    __slots__ = ("i_in",)

class _NodeOut(Bag):
    __slots__ = ("u",)

class RCNode:
    """Node with a shunt capacitor ``C`` (F) in series with ``R_d`` (ohm) to ground.

    Input ``i_in``: sum of currents into the node (fan-in connection). Output ``u = u_C + R_d i_in``.
    """

    state_names: ClassVar[tuple[str, ...]] = ("u_C",)
    outputs_need_inputs: ClassVar[bool] = True
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.u", "inp.i_in", 1.5),)

    def __init__(self, C: float, R_d: float, u0: complex = 0j) -> None:
        self.C, self.R_d = C, R_d
        self.storage = (StoragePort("u_C", C, 1.5, (("inp.i_in", 1.0),)),)
        self.state = _NodeState(u_C=complex(u0))
        self.inp = _NodeInp(i_in=0j)
        self.out = _NodeOut(u=complex(u0))

    def set_outputs(self, t: float) -> None:
        self.out.u = self.state.u_C + self.R_d * self.inp.i_in

    def rhs(self, t: float):
        return (self.inp.i_in / self.C,)

    def retune(self, C: float, R_d: float) -> None:
        """Change capacitance and damping resistance without changing voltage."""
        self.C, self.R_d = C, R_d
        self.storage = (replace(self.storage[0], value=C),)

    def dissipated_power(self) -> float:
        return 1.5 * self.R_d * abs(self.inp.i_in) ** 2

    def supplied_power(self) -> float:
        return 0.0


class _SourceOut(Bag):
    __slots__ = ("e_g", "phi", "theta")


class _SourceInp(Bag):
    __slots__ = ("i",)


class ThreePhaseSource:
    """Ideal three-phase voltage source ``e_g(t) = E exp(j (w0 t + phi(t)))``.

    ``w0`` is in rad/s, ``e_peak`` in V and ``phi(t)`` is the relative angle in rad.
    Event handling updates ``e_peak`` or ``phi`` at the event time.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.e_g", "inp.i", 1.5, -1.0),)
    has_source: ClassVar[bool] = True

    def __init__(self, w0: float, e_peak: float,
                 phi: Optional[Callable[[float], float]] = None) -> None:
        self.w0, self.e_peak = w0, e_peak
        self.phi = phi
        self.state = Empty()
        self.inp = _SourceInp(i=0j)
        self.out = _SourceOut()

    def set_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.out
        out.phi = phi
        out.theta = theta
        out.e_g = complex(self.e_peak * math.cos(theta), self.e_peak * math.sin(theta))

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        return 0.0

    def supplied_power(self) -> float:
        return 1.5 * (self.out.e_g * self.inp.i.conjugate()).real


# ------------------------------------------------------------------ circuit element types

class Element:
    """Base of a registered ``elements`` entry, built as ``cls(name, cfg, buses, p)``.

    A type defines ``type`` and a frozen ``Params`` dataclass with a matching ``type`` field.
    An instance supplies ``subsystems()``, ``connections()``, ``bus_name`` and ``injection``.
    Optional ``breakers`` or ``connect()`` enable switching; optional ``retune()`` enables
    runtime parameter changes.
    """

    type: ClassVar[str]
    Params: ClassVar[type]


ELEMENT_TYPES: dict[str, type] = {}


def register_element_type(cls: type) -> type:
    """Register an :class:`Element` class under ``cls.type`` and return it."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("an element type needs a name: the class attribute 'type'")
    if name in ELEMENT_TYPES:
        raise ValueError(f"element type {name!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or "type" not in params.__dataclass_fields__:
        raise TypeError(f"element type {name!r}: Params must be a dataclass with a type field")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"element type {name!r}: Params.type must default to {name!r}")
    ELEMENT_TYPES[name] = cls
    return cls


def element_buses(cfg: Any) -> dict[str, str]:
    """Return fields named ``bus`` or ending in ``_bus`` from an element's parameters."""
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)
            if f.init and (f.name == "bus" or f.name.endswith("_bus"))}


@register_element_type
class Load(Element):
    """Built-in series R-L load from ``bus`` to ground; a small impedance models a fault."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "load"
        bus: str
        l: float
        r: float = 0.0

        _quantities = {"l": "inductance", "r": "resistance"}
        _input_aliases = {"x_pu": "l_pu"}

        def __post_init__(self) -> None:
            if not self.l > 0.0:
                raise ConfigError(f"l must be > 0 (x_pu > 0), got {self.l}")
            if not self.r >= 0.0:
                raise ConfigError(f"r must be >= 0, got {self.r}")

    type = "load"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg, self.bus = name, cfg, buses[cfg.bus]
        self.branch = RLBranch(cfg.l, cfg.r)
        self.breakers = [self.branch]

    def subsystems(self) -> dict[str, Any]:
        return {f"{self.name}.branch": self.branch}

    def connections(self) -> dict:
        return {(self.branch, "u_from"): (self.bus, "u")}

    @property
    def bus_name(self) -> str:
        return self.cfg.bus

    @property
    def injection(self) -> tuple:
        return (self.branch, "i", -1.0)

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i": self.branch.out.i}

    def retune(self, cfg: Any) -> None:
        """Take new load parameters while preserving branch current."""
        self.cfg = cfg
        self.branch.retune(cfg.l, cfg.r)
