"""The refactored controller, its loop registry and dependency boundary."""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.control import LOOP_TYPES, Loop, Measurement, SyncLaw, register_loop_type
from peslite.solver.model import ConfigError

CONTROL = Path(peslite.control.__file__).parent
QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}
FAST = {"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.01}


@register_loop_type
class _FixedFrequency(SyncLaw):
    """A registered synchronization law at nominal frequency."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        type: str = "test_fixed_frequency"

    type = "test_fixed_frequency"

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
             p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0
        self.theta += T * self.omega
        self.v_mag = v_ref_pu


def test_registered_loop_owns_its_schema_and_uses_default_role_wiring():
    p = peslite.load(
        EXAMPLES / "gfm-psc-example.pes",
        **QUIET,
        **FAST,
        **{"simulation.t_end": 0.01},
    )
    p = p.replace(**{
        "units.vsc.ctrl.loops.sync": {
            "type": "test_fixed_frequency",
            "period": 5e-5,
        }
    })
    sim = peslite.Simulation(p)
    ctrl = sim.unit().ctrl
    assert type(p.unit().ctrl.loops["sync"]) is _FixedFrequency.Params
    assert ctrl.graph.connections["impedance.v_ref"] == "sync.v_ref"
    result = sim.run()
    assert result.summary["vsc.law"] == "test_fixed_frequency"
    assert result.states["vsc.ctrl.sync.theta"][-1] == pytest.approx(
        2 * np.pi * 50.0 * 0.01, rel=1e-9
    )


def test_loop_registration_and_loop_owned_validation():
    class NoParams(Loop):
        type = "test_no_params"

    with pytest.raises(TypeError):
        register_loop_type(NoParams)
    with pytest.raises(ValueError):
        register_loop_type(LOOP_TYPES["psc"])
    with pytest.raises(ConfigError, match="loops.va.x_v_pu must be positive"):
        peslite.load(
            EXAMPLES / "gfm-droop-example.pes",
            **{"units.vsc.ctrl.loops.va.x_v_pu": 0.0},
        )


def test_controller_interfaces_have_one_definition():
    from peslite.components.adc import Measurement as ADCMeasurement

    assert ADCMeasurement is Measurement


def test_control_imports_only_itself_and_the_solver_kernel():
    allowed = {"control", "solver"}
    for path in CONTROL.glob("*.py"):
        imported = set()
        rel = path.relative_to(CONTROL.parent).with_suffix("")
        here = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = here[:len(here) - (node.level - 1)]
                target = base + (node.module.split(".") if node.module else [])
                if target:
                    imported.add(target[0])
        assert imported <= allowed, f"{path.name} imports {sorted(imported - allowed)}"
