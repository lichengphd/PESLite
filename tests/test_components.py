"""The refactored power-circuit and converter-hardware components."""

import ast
from pathlib import Path

import numpy as np

import peslite
from conftest import EXAMPLES
from peslite.components import (
    ADC,
    MeasurementPorts,
    PWM,
    ZOH,
)
from peslite.control.blocks import abc2complex

COMPONENTS = Path(peslite.components.__file__).parent
PACKAGE = COMPONENTS.parent


def test_unit_delegates_sampling_and_modulation_state_to_components():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    unit = peslite.Simulation(p).unit()

    assert isinstance(unit.adc, ADC)
    assert isinstance(unit.pwm, PWM)
    assert peslite.ADC is ADC and peslite.PWM is PWM
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
    assert set(adc.get_state()) == {"x_v", "x_i"}


def test_adc_reconstructs_its_sampling_position_from_the_start_time():
    ports = MeasurementPorts(
        u_g=lambda: 1 + 0j,
        i_c=lambda: 2 + 0j,
        i_c_state=lambda: 2 + 0j,
        u_dc=lambda: 3.0,
    )
    adc = ADC(ports, period=1.0, samples=4)
    adc.start(0.6, 0.0, 1.0, held=True)
    assert adc.n_samp == 3
    assert [sample.t for sample in adc.peeks] == [0.25, 0.5]
    assert adc.t_sample(0.0) == 0.75


def test_pwm_owns_timer_registers_and_switching_schedule():
    initial = np.array([0.2, 0.4, 0.6])
    pwm = PWM(period=1e-3, load_period=1e-3, offset=0.0, computation=2e-4,
              carrier_period=1e-3, modulator=ZOH())
    pwm.reset(initial)
    assert pwm.get_state() == {
        "d_a": 0.2, "d_b": 0.4, "d_c": 0.6,
        "shadow.d_a": 0.2, "shadow.d_b": 0.4, "shadow.d_c": 0.6,
    }

    pwm.write(0.0, np.array([0.8, 0.7, 0.6]))
    q = pwm.load(0.0)
    assert q == abc2complex(initial)
    assert pwm.k == 1
    assert pwm.t_next == 1e-3
    assert pwm.next_switch == float("inf")
    pwm.load(1e-3)
    assert pwm.active[0] == 0.8


def test_duty_register_states_are_owned_by_pwm():
    params = peslite.load(EXAMPLES / "gfl-example.pes")
    names = set(peslite.Simulation(params).state_names())
    for part in ("", "shadow."):
        assert {f"vsc.pwm.{part}d_{phase}" for phase in "abc"} <= names
    assert not any(".clock." in name or "pending." in name or "computation_delay" in name
                   for name in names)


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
