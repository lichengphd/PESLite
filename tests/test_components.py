"""The refactored power-circuit and converter-hardware components."""

import ast
from pathlib import Path

import numpy as np

import peslite
from conftest import EXAMPLES
from peslite.components import (
    ADC,
    ComputationDelay,
    MeasurementPorts,
    ZOH,
)
from peslite.components.pwm import PWM
from peslite.control.blocks import abc2complex

COMPONENTS = Path(peslite.components.__file__).parent
PACKAGE = COMPONENTS.parent


def test_unit_delegates_sampling_and_modulation_state_to_components():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    unit = peslite.Simulation(p).unit()

    assert isinstance(unit.adc, ADC)
    assert isinstance(unit.pwm, PWM)
    assert isinstance(unit.pwm.computation_delay, ComputationDelay)
    assert not hasattr(unit, "sampler")
    assert not hasattr(unit, "window")
    assert not hasattr(unit, "modulator")
    assert not hasattr(unit, "computation_delay")


def test_adc_owns_oversampling_and_averaging_window_state():
    values = {"v": 1 + 0j, "i": 2 + 0j, "dc": 10.0}
    ports = MeasurementPorts(
        u_g=lambda: values["v"],
        i_c=lambda: values["i"],
        i_c_state=lambda: values["i"],
        u_dc=lambda: values["dc"],
    )
    adc = ADC(ports, period=1.0, samples=2, length=1.0,
              channels={"v": 0j, "i": 0j})
    adc.seed()
    values.update(v=3 + 0j, i=4 + 0j)
    adc.accumulate(0.5)
    adc.peek(0.5)
    values.update(v=5 + 0j, i=6 + 0j)
    adc.accumulate(0.5)
    sample = adc.sample(1.0)

    assert sample.u_g == 3 + 0j
    assert sample.i_c == 4 + 0j
    assert sample.u_dc == 10.0
    assert [s.t for s in sample.samples] == [0.5, 1.0]
    assert set(adc.get_state()) == {"x_v", "x_i", "x_v_open", "x_i_open"}


def test_pwm_owns_delay_publications_and_switching_schedule():
    delay = ComputationDelay(1)
    initial = np.array([0.2, 0.4, 0.6])
    delay.reset(initial)
    pwm = PWM(period=1e-3, modulator=ZOH(), computation_delay=delay)
    assert pwm.get_state() == {
        "d_a": 0.0, "d_b": 0.0, "d_c": 0.0,
        "computation_delay.0.d_a": 0.2,
        "computation_delay.0.d_b": 0.4,
        "computation_delay.0.d_c": 0.6,
    }

    pwm.publish(np.array([0.8, 0.7, 0.6]))
    assert np.array_equal(pwm.d, initial)
    q = pwm.modulate(0.0)
    assert q == abc2complex(initial)
    assert pwm.k == 1
    assert pwm.t_next == 1e-3
    assert pwm.next_switch == float("inf")
    assert pwm.get_state()["computation_delay.0.d_a"] == 0.8


def test_computation_delay_state_is_owned_by_pwm():
    params = peslite.load(EXAMPLES / "gfl-example.pes", **{"units.vsc.pwm.computation_delay.steps": 2})
    names = set(peslite.Simulation(params).state_names())
    assert {
        "vsc.pwm.computation_delay.0.d_a", "vsc.pwm.computation_delay.0.d_b",
        "vsc.pwm.computation_delay.0.d_c", "vsc.pwm.computation_delay.1.d_a",
        "vsc.pwm.computation_delay.1.d_b", "vsc.pwm.computation_delay.1.d_c",
    } <= names
    assert not any(name.startswith(("delay.", "pwm.")) for name in names)


def test_components_import_only_control_solver_and_themselves():
    allowed = {"components", "control", "solver"}
    for path in COMPONENTS.glob("*.py"):
        imported = set()
        rel = path.relative_to(COMPONENTS.parent).with_suffix("")
        here = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = here[:len(here) - (node.level - 1)]
                target = base + (node.module.split(".") if node.module else [])
                if target:
                    imported.add(target[0])
        assert imported <= allowed, f"{path.name} imports {sorted(imported - allowed)}"


def test_absorbed_hardware_packages_are_removed():
    for name in ("power", "sensing", "firmware", "modulation", "protection"):
        assert not (PACKAGE / name).exists()
