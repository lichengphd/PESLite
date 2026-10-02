"""The refactored controller, its loop registry and dependency boundary."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.control import LOOP_TYPES, Loop, Measurement, SyncLaw, register_loop_type
from peslite.control.loops import V_DQ
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


@register_loop_type
class _AffineFeedback(Loop):
    """A stateless continuous loop used to exercise automatic loop-level feedback solving."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        gain: float
        bias: complex
        period: float | None = None
        type: str = "test_affine_feedback"

    type = "test_affine_feedback"
    inputs = {"x": V_DQ}
    outputs = {"y": V_DQ}

    def initial_outputs(self):
        return {"y": 0j}

    def update(self, t, inputs):
        return {"y": self.cfg.bias + self.cfg.gain * inputs["x"]}

    def continuous(self, t, inputs):
        return self.update(t, inputs), {}


def test_registered_loop_owns_its_schema_and_uses_default_role_wiring(tmp_path):
    p = peslite.load(
        EXAMPLES / "gfm-psc-example.pes",
        **QUIET,
        **FAST,
        **{"simulation.t_end": 0.01},
    )
    p = p.replace(**{
        "units.vsc.bridge.model": "pwm_averaging",
        "units.vsc.ctrl.loops.sync": {
            "type": "test_fixed_frequency",
            "period": 5e-5,
        }
    })
    sim = peslite.Simulation(p)
    ctrl = sim.unit().ctrl
    assert type(p.unit().ctrl.loops["sync"]) is _FixedFrequency.Params
    assert ctrl.graph.connections["vi.v_ref"] == "sync.v_ref"
    result = sim.run(out_dir=tmp_path)
    assert result.summary["vsc.law"] == "test_fixed_frequency"
    assert result.states["vsc.ctrl.sync.theta"][-1] == pytest.approx(
        2 * np.pi * 50.0 * 0.01, rel=1e-9
    )


def test_sampled_custom_loop_must_supply_its_own_continuous_equations_for_averaging():
    p = peslite.load(
        EXAMPLES / "gfm-psc-example.pes", **QUIET,
        **{"units.vsc.bridge.model": "averaging"},
    )
    p = p.replace(**{"units.vsc.ctrl.loops.sync": {
        "type": "test_fixed_frequency", "period": 5e-5,
    }})
    with pytest.raises(ConfigError, match="no continuous-time flow implementation"):
        peslite.Simulation(p)


def test_continuous_control_graph_automatically_solves_feedback_between_loops():
    p = peslite.load(
        EXAMPLES / "gfl-example.pes", **QUIET,
        **{"units.vsc.bridge.model": "averaging"},
    )
    unit = peslite.Simulation(p).unit().cfg
    loops = {
        **unit.ctrl.loops,
        "feedback_a": _AffineFeedback.Params(gain=0.25, bias=1.0),
        "feedback_b": _AffineFeedback.Params(gain=0.5, bias=2.0),
    }
    connections = {
        **unit.ctrl.connections,
        "feedback_a.x": "feedback_b.y",
        "feedback_b.x": "feedback_a.y",
    }
    configured = replace(
        unit, ctrl=replace(unit.ctrl, loops=loops, connections=connections)
    )
    object.__setattr__(configured, "base", unit.base)
    controller = peslite.UniteType(configured)

    controller.set_outputs(0.0)

    assert controller.graph.values["feedback_a.y"] == pytest.approx(12.0 / 7.0)
    assert controller.graph.values["feedback_b.y"] == pytest.approx(20.0 / 7.0)


def test_loop_registration_and_loop_owned_validation():
    class NoParams(Loop):
        type = "test_no_params"

    with pytest.raises(TypeError):
        register_loop_type(NoParams)
    with pytest.raises(ValueError):
        register_loop_type(LOOP_TYPES["psc"])
    params = peslite.load(EXAMPLES / "gfm-psc-example.pes")
    with pytest.raises(ConfigError, match="loops.vi.x_v_pu must be positive"):
        params.replace(**{"units.vsc.ctrl.loops.vi": {
            "type": "virtual_admittance", "x_v_pu": 0.0,
        }})


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
