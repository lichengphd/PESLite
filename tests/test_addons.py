"""Add-on discovery plus a custom PLL assembled and run with built-in controller loops."""

import importlib

import numpy as np
import peslite
import pytest
from peslite.addons import components as addon_components
from peslite.addons import controllers as addon_controllers
from peslite.addons.controllers.voltage_adaptive_pll import VoltageAdaptivePLL
from peslite.components import ELEMENT_TYPES
from peslite.control import LOOP_TYPES, Loop


def test_controller_addon_path_registers_loops(tmp_path, monkeypatch):
    path = tmp_path / "controllers"
    path.mkdir()
    (path / "test_controller_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.addons.controllers import Loop, register_loop_type

@register_loop_type
class AddonLoop(Loop):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float | None = None
        type: str = "addon_test_loop"

    type = "addon_test_loop"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(addon_controllers, "__path__", [str(path)])
    importlib.invalidate_caches()

    imported = addon_controllers.discover()

    assert [module.__name__ for module in imported] == [
        "peslite.addons.controllers.test_controller_addon"
    ]
    assert LOOP_TYPES["addon_test_loop"].__module__ == imported[0].__name__


def test_component_addon_path_registers_elements(tmp_path, monkeypatch):
    path = tmp_path / "components"
    path.mkdir()
    (path / "test_component_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.addons.components import Element, register_element_type

@register_element_type
class AddonElement(Element):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "addon_test_element"
        bus: str

    type = "addon_test_element"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(addon_components, "__path__", [str(path)])
    importlib.invalidate_caches()

    imported = addon_components.discover()

    assert [module.__name__ for module in imported] == [
        "peslite.addons.components.test_component_addon"
    ]
    assert ELEMENT_TYPES["addon_test_element"].__module__ == imported[0].__name__


def test_functions_hold_the_plotting_addon():
    assert peslite.addons.functions.plot_csv is peslite.addons.plot_csv
    assert peslite.addons.functions.plot_result is peslite.addons.plot_result


def test_addon_entry_points_match_builtin_public_apis():
    assert set(peslite.control.__all__) < set(addon_controllers.__all__)
    assert set(peslite.components.__all__) < set(addon_components.__all__)
    assert addon_controllers.LOOP_TYPES is LOOP_TYPES
    assert addon_components.ELEMENT_TYPES is ELEMENT_TYPES


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_custom_pll_assembles_with_builtin_loops_and_runs(examples, tmp_path, model):
    params = peslite.load(
        examples / "custom-pll-example.pes",
        **{
            "simulation.t_end": 0.004,
            "simulation.solver.linearisations": 0,
            "units.vsc.bridge.model": model,
        },
    )
    simulation = peslite.Simulation(params)
    nodes = simulation.unit().ctrl.graph.nodes

    assert set(nodes) == {"pll", "cc", "dvc"}
    assert isinstance(nodes["pll"], VoltageAdaptivePLL)
    assert type(nodes["pll"]).sample is Loop.sample
    assert type(nodes["pll"]).flow_path is Loop.flow_path
    assert set(nodes["pll"]._state_blocks) == {"theta", "omega_g", "u_g_pu"}
    assert nodes["cc"].type == "dq_current_pi"
    assert nodes["dvc"].type == "dc_voltage_pi"

    result = simulation.run(out_dir=tmp_path / model)
    theta = result.states["vsc.ctrl.pll.theta"]
    assert np.isfinite(theta).all()
    assert theta[-1] > 1.0
    assert not result.tripped
