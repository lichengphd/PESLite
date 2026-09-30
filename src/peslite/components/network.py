"""AC network blocks: the three-phase source, series R-L branches and nodes with a damped shunt capacitor.

Quantities are SI; ac quantities are peak-scaled space vectors. Breakers expose ``open_breaker()``.
"""

from __future__ import annotations

import math
from typing import Callable, ClassVar, Optional

from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, Empty

__all__ = ["ThreePhaseSource", "RLBranch", "RCNode"]


class _BranchState(Bag):
    __slots__ = ("i",)

class _BranchInp(Bag):
    __slots__ = ("u_from", "u_to")

class _BranchOut(Bag):
    __slots__ = ("i",)

class RLBranch:
    """Series R-L branch (H, ohm); current positive from ``u_from`` to ``u_to``; ``open_breaker`` zeroes it."""

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

    def open_breaker(self) -> None:
        self.breaker_open = True
        self.state.i = 0j

    def set_outputs(self, t: float) -> None:
        self.out.i = self.state.i

    def rhs(self, t: float):
        if self.breaker_open:
            return (0j,)
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

    def dissipated_power(self) -> float:
        return 1.5 * self.R_d * abs(self.inp.i_in) ** 2

    def supplied_power(self) -> float:
        return 0.0


class _SourceOut(Bag):
    __slots__ = ("e_g", "phi", "theta")


class _SourceInp(Bag):
    __slots__ = ("i",)


class ThreePhaseSource:
    """Ideal three-phase voltage source ``e_g(t) = E(t) exp(j (w0 t + phi(t)))``.

    ``w0`` in rad/s; ``phi(t)`` angle deviation (rad) and ``magnitude(t)`` peak voltage (V),
    each ``None`` for zero / constant ``e_peak``. Input ``i``: delivered current (A), for energy accounting.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.e_g", "inp.i", 1.5, -1.0),)
    has_source: ClassVar[bool] = True

    def __init__(self, w0: float, e_peak: float, phi: Optional[Callable[[float], float]] = None,
                 magnitude: Optional[Callable[[float], float]] = None) -> None:
        self.w0, self.e_peak = w0, e_peak
        self.phi = phi
        self.magnitude = magnitude
        self.state = Empty()
        self.inp = _SourceInp(i=0j)
        self.out = _SourceOut()

    def set_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.out
        out.phi = phi
        out.theta = theta
        e = self.magnitude(t) if self.magnitude is not None else self.e_peak
        out.e_g = complex(e * math.cos(theta), e * math.sin(theta))

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        return 0.0

    def supplied_power(self) -> float:
        return 1.5 * (self.out.e_g * self.inp.i.conjugate()).real
