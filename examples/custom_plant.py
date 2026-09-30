"""Run a case with an extra bus, a synchronization law of a registered loop type and a user solver.

    python examples/custom_plant.py

The law's time constant is read from ``meta.custom.psc_lag_tau_s`` in the configuration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # without installing

import peslite  # noqa: E402
from peslite.control import SyncLaw, register_loop_type  # noqa: E402
from peslite.solver import SolverStep  # noqa: E402


def two_section_system(p: peslite.Params) -> peslite.Params:
    """Return ``p`` with the source impedance split by a new bus ``mid``.

    Topology: ``grid --[Z_s/2]-- mid --[Z_s/2]-- pcc --[Z_f]-- vsc``.
    """
    src, bus = p.sources["grid"], p.buses["pcc"]
    return p.replace(**{
        "buses": {"pcc": {"c": bus.c, "r_d": bus.r_d},
                  "mid": {"c": bus.c, "r_d": bus.r_d}},
        "sources.grid.bus": "mid",
        "sources.grid.l": 0.5 * src.l,
        "sources.grid.r": 0.5 * src.r,
        "branches": {"line": {"from_bus": "mid", "to_bus": "pcc",
                              "l": 0.5 * src.l, "r": 0.5 * src.r}},
    })


@register_loop_type
class LaggedPSC(SyncLaw):
    """Loop type ``lagged_psc``: PSC with a first-order lag ``tau`` (s) on the power feedback.

    As a synchronization law (a :class:`~peslite.control.SyncLaw`) the default ``gfm`` wiring
    connects it like the built-in ones. Named states: ``theta`` (rad), ``p_f`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        k_p_pu: float  # rad/s per pu power
        tau: float  # s
        type: str = "lagged_psc"

        _quantities = {"k_p_pu": "1/power"}

    type = "lagged_psc"
    state_names = ("theta", "p_f")

    def __init__(self, cfg, unit, scenario) -> None:
        super().__init__(cfg, unit, scenario)
        self.p_f = 0.0

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq) -> None:
        self.p_f += T / self.cfg.tau * (p_pu - self.p_f)
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.theta += T * self.omega
        self.v_mag = v_ref_pu


class Midpoint:
    """User solver: explicit midpoint rule with maximum step ``dt`` (s)."""

    def __init__(self, dt: float) -> None:
        self.dt, self.n_rhs = dt, 0

    def __call__(self, f, t0, t1, y0) -> SolverStep:
        n = max(1, int(np.ceil((t1 - t0) / self.dt - 1e-9)))
        h, y, t = (t1 - t0) / n, y0, t0
        for _ in range(n):
            k1 = f(t, y)
            y = y + h * f(t + 0.5 * h, y + 0.5 * h * k1)
            t += h
        self.n_rhs += 2 * n
        return SolverStep(t1, y, self.n_rhs)


def main(argv=None) -> int:
    default = (Path(__file__).resolve().parents[1] / "src" / "peslite" / "configs"
               / "gfm-psc-example.yaml")
    config = argv[0] if argv else (sys.argv[1] if len(sys.argv) > 1 else default)
    p = two_section_system(peslite.load(config, **{"simulation.t_end": 1.0, "simulation.progress_every": 0.0}))
    sync = p.unit("vsc").control.loops["sync"]
    p = p.replace(**{
        "initial.states.plant.mid.u_C": "source",  # the extra bus, pre-charged
        "units.vsc.control.loops.sync": {"type": "lagged_psc", "period": sync.period, "k_p_pu": sync.k_p_pu,
                                         "tau": p.meta["custom"]["psc_lag_tau_s"]},
    })
    u = p.unit("vsc")
    sim = peslite.Simulation(p, solver=Midpoint(p.simulation.solver.dt))
    print("states:", ", ".join(sim.state_names()))
    r = sim.run()
    print(f"two-section system, lagged PSC, midpoint solver: wall {r.wall_time:.1f} s, "
          f"rhs {r.n_rhs}, final p = {r.control['vsc.p_pu'][-1]:.4f} pu, "
          f"|v_pcc| = {abs(r.plant['vsc.u_g'][-1]) / u.base.v_phase_peak:.4f} pu, tripped = {r.tripped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
