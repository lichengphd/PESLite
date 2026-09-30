"""Issue #2 parameter layout, switches, derived defaults and resolved files."""

import math
import re

import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, dumps, from_dict, read_tree, to_dict


def _tree():
    return read_tree(EXAMPLES / "gfl-example.pes")


def test_initial_and_output_belong_to_simulation():
    p = from_dict(_tree())
    assert p.simulation.initial.t == 0.0
    assert p.simulation.output.period == 5e-4
    assert not hasattr(p, "initial") and not hasattr(p, "output")

    tree = _tree()
    tree["initial"] = tree["simulation"].pop("initial")
    with pytest.raises(ConfigError, match=r"unknown key.*initial"):
        from_dict(tree)


def test_log_and_old_parameter_names_are_rejected():
    tree = _tree()
    tree["simulation"]["log"] = {"plant_period": 5e-4}
    with pytest.raises(ConfigError, match=r"simulation: unknown key.*log"):
        from_dict(tree)

    tree = _tree()
    current = tree["units"]["vsc"]["control"]["loops"]["cc"]
    current["bw_hz"] = current.pop("bandwidth")
    with pytest.raises(ConfigError, match=r"unknown key.*bw_hz"):
        from_dict(tree)


def test_nested_electrical_values_follow_si_and_pu_spelling():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    amperes = 1.15 * p.unit("vsc").base.i_phase_peak
    q = p.replace(**{"units.vsc.protection.overcurrent.limit": amperes})
    assert q.unit("vsc").protection.overcurrent.limit_pu == pytest.approx(1.15)
    with pytest.raises(ConfigError, match="both actual and pu values"):
        p.replace(**{"units.vsc.protection.overcurrent.limit": amperes,
                     "units.vsc.protection.overcurrent.limit_pu": 1.15})


def test_switched_sections_require_settings_and_write_zero_or_one():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    written = yaml.safe_load(dumps(p))
    protection = written["units"]["vsc"]["protection"]
    assert protection["overcurrent"] == {"enable": 1, "limit_pu": 1.15}
    assert protection["frequency"] == {"enable": 0, "limit": None}
    assert written["simulation"]["stop_on_trip"] == 1
    assert written["simulation"]["output"]["signals"] == 0
    assert not re.search(r":\s*(true|false)\b", dumps(p))

    with pytest.raises(ConfigError, match=r"frequency\.limit: required when enable is 1"):
        p.replace(**{"units.vsc.protection.frequency.enable": 1})
    with pytest.raises(ConfigError, match=r"expected 1 \(on\) or 0 \(off\)"):
        p.replace(**{"simulation.output.signals": 2})


def test_dependent_defaults_follow_replace_and_stay_fixed_when_explicit():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    u = p.replace(**{"units.vsc.pwm.f_sw": 10_000.0, "base.f0": 60.0}).unit("vsc")
    assert u.pwm.update_period == pytest.approx(1e-4)
    assert u.control.references.omega == pytest.approx(120 * math.pi)
    assert u.control.sampling_period is None and u.measurement.window is None

    fixed = p.replace(**{"units.vsc.pwm.update_period": 5e-5})
    fixed = fixed.replace(**{"units.vsc.pwm.f_sw": 10_000.0})
    assert fixed.unit("vsc").pwm.update_period == 5e-5

    rebased = p.replace(**{"base.s_base": 1e6})
    assert rebased.unit("vsc").base.s_base == 1e6
    averaged = p.replace(**{"units.vsc.measurement.average": "window",
                            "units.vsc.pwm.f_sw": 10_000.0})
    written = yaml.safe_load(dumps(averaged))
    assert written["units"]["vsc"]["control"]["sampling_period"] == pytest.approx(1e-4)
    assert written["units"]["vsc"]["measurement"]["window"] == pytest.approx(1e-4)


def test_to_dict_preserves_following_defaults_but_dumps_resolves_them():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    compact = to_dict(p)
    complete = yaml.safe_load(dumps(p))
    assert compact["units"]["vsc"]["pwm"]["update_period"] is None
    assert compact["units"]["vsc"]["control"]["references"]["omega"] is None
    assert complete["units"]["vsc"]["pwm"]["update_period"] == pytest.approx(5e-5)
    assert complete["units"]["vsc"]["control"]["references"]["omega"] == pytest.approx(100 * math.pi)
    assert complete["units"]["vsc"]["control"]["sampling_period"] == pytest.approx(5e-5)
    assert complete["units"]["vsc"]["s_base"] == 2e6
    assert dumps(from_dict(complete)) == dumps(p)


def test_resolved_cli_and_meta_contract(capsys):
    assert peslite.main(["gfl-example", "--resolved", "--set", "units.vsc.pwm.f_sw=10000"]) == 0
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["units"]["vsc"]["pwm"]["update_period"] == pytest.approx(1e-4)
    from_dict(resolved)

    tree = _tree()
    tree["meta"] = {"title": "A case", "description": "Text only."}
    assert from_dict(tree).meta.title == "A case"
    tree["meta"] = {"custom": 3}
    with pytest.raises(ConfigError, match="meta holds only a title and a description"):
        from_dict(tree)


def test_dump_and_initial_file_use_the_nested_simulation_section(tmp_path):
    p = peslite.load(EXAMPLES / "gfl-example.pes").replace(
        **{"simulation.initial.t": 0.001,
           "simulation.initial.states.plant.pcc.u_C": [500.0, 0.0]})
    path = tmp_path / "restart.pes"
    peslite.dump(p, path)
    assert path.read_text(encoding="utf-8").startswith("# PESLite 0.1.1")

    restarted = peslite.load(EXAMPLES / "gfl-example.pes", initial=path)
    assert restarted.simulation.initial.t == 0.001
    assert restarted.simulation.initial.states["plant.pcc.u_C"] == [500.0, 0.0]
