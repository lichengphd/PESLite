"""Issue #3: each unit selects switching or one of the two averaging bridge models."""

import io
import re
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError


SHORT = ["--set", "simulation.t_end=0.004",
         "--set", "simulation.solver.linearisations=0"]


def _models(*args):
    """Return ``{unit: (enable, over)}`` after resolving command-line arguments."""
    output = io.StringIO()
    with redirect_stdout(output):
        assert peslite.main([*args, "--resolved"]) == 0
    units = yaml.safe_load(output.getvalue())["units"]
    return {name: (unit["averaging"]["enable"], unit["averaging"]["over"])
            for name, unit in units.items()}


def test_switching_is_default_and_averaging_is_a_switched_unit_section():
    assert _models("gfl-example") == {"vsc": (0, "pwm_period")}
    assert _models("gfm-droop-example") == {"vsc": (1, "pwm_period")}


@pytest.mark.parametrize("name", ["gfl-example", "gfm-droop-example", "two-converters-example"])
def test_averaging_flag_turns_averaging_on_for_every_unit(name):
    assert all(enable == 1 for enable, _over in _models(name, "--averaging").values())


def test_there_is_no_switching_flag(capsys):
    with pytest.raises(SystemExit):
        peslite.main(["gfm-droop-example", "--switching", "--resolved"])
    assert "unrecognized arguments: --switching" in capsys.readouterr().err


def _with_model(params, enable, over):
    changes = {f"units.{name}.averaging.{key}": value
               for name in params.units
               for key, value in (("enable", enable), ("over", over))}
    return params.replace(**changes)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*-example.pes")), ids=lambda path: path.stem)
@pytest.mark.parametrize("enable, over", [(0, "pwm_period"),
                                           (1, "pwm_period"),
                                           (1, "time_step")])
def test_every_bundled_file_runs_with_every_model(path, enable, over):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = peslite.load(
            path,
            **{"simulation.t_end": 0.004, "simulation.solver.linearisations": 0},
        )
        params = _with_model(params, enable, over)
    result = peslite.Simulation(params).run()
    assert np.isfinite(result.states["t"]).all()
    assert result.states["t"][-1] == pytest.approx(0.004)


def test_units_can_have_different_models():
    params = peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **{"simulation.t_end": 0.004,
           "simulation.solver.linearisations": 0,
           "units.vsc_m.averaging.enable": 1},
    )
    simulation = peslite.Simulation(params)
    models = tuple(type(simulation.units[name].pwm.modulator).__name__
                   for name in ("vsc_m", "vsc_l"))
    assert models == ("ZOH", "CarrierComparison")
    simulation.run()


def test_averaging_flag_is_the_same_as_setting_the_file_value(tmp_path):
    out_flag, out_set = tmp_path / "flag", tmp_path / "set"
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", "--averaging", *SHORT,
                             "--out", str(out_flag)]) == 0
        assert peslite.main(["gfl-example", "--set", "units.vsc.averaging.enable=1",
                             *SHORT, "--out", str(out_set)]) == 0
    assert (out_flag / "states.csv").read_text() == (out_set / "states.csv").read_text()
    assert peslite.load(out_flag / "simulation.pes").unit("vsc").averaging.enable


def test_averaging_flag_wins_over_set():
    assert _models("gfl-example", "--averaging",
                   "--set", "units.vsc.averaging.enable=0") == {
                       "vsc": (1, "pwm_period")}


def test_time_step_averaging_is_a_kind_of_averaging():
    assert _models("gfl-example", "--averaging",
                   "--set", "units.vsc.averaging.over=time_step") == {
                       "vsc": (1, "time_step")}
    assert _models("gfl-example", "--averaging") == {"vsc": (1, "pwm_period")}


def test_time_step_averaging_follows_carrier_within_each_period(gfl):
    runs = {}
    for over in ("pwm_period", "time_step"):
        params = gfl(**{"units.vsc.averaging.over": over,
                        "simulation.output.period": 12.5e-6})
        runs[over] = peslite.Simulation(params).run().states["plant.vsc.branch_f.i.re"]
    ripple = {name: np.ptp(np.diff(values[-40:], 2)) for name, values in runs.items()}
    assert ripple["time_step"] > 10 * ripple["pwm_period"]


def test_over_has_no_effect_on_switching_bridge(gfl):
    base = {"units.vsc.averaging.enable": 0, "simulation.t_end": 0.002}
    first = peslite.Simulation(gfl(**base)).run()
    second = peslite.Simulation(
        gfl(**base, **{"units.vsc.averaging.over": "time_step"})
    ).run()
    assert all(np.array_equal(first.states[name], second.states[name], equal_nan=True)
               for name in first.states)


def test_averaging_flag_results_use_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(peslite.solver.simulation, "_RESULTS", tmp_path)
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", *SHORT]) == 0
        assert peslite.main(["gfl-example", "--averaging", *SHORT]) == 0
        assert peslite.main(["gfm-droop-example", "--averaging", *SHORT]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gfl-example", "gfl-example-averaging", "gfm-droop-example"]
    assert not peslite.load(tmp_path / "gfl-example" / "simulation.pes").unit("vsc").averaging.enable
    assert peslite.load(tmp_path / "gfl-example-averaging" / "simulation.pes").unit("vsc").averaging.enable


def test_carrier_settings_are_ignored_for_pwm_period_averaging(gfl):
    duration = {"simulation.t_end": 0.004}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        synchronous = gfl(**duration,
                          **{"units.vsc.pwm.sync": "synchronous",
                             "units.vsc.pwm.carrier_phase": 0.3})
    asynchronous = gfl(**duration)
    first = peslite.Simulation(synchronous).run()
    second = peslite.Simulation(asynchronous).run()
    for name in second.states:
        assert np.array_equal(first.states[name], second.states[name], equal_nan=True), name


def test_time_step_averaging_has_only_an_asynchronous_carrier(gfl):
    message = ("units.vsc.pwm.sync = 'synchronous' is not available with time-step "
               "averaging (units.vsc.averaging.over = 'time_step')")
    with pytest.raises(ConfigError, match=re.escape(message)):
        gfl(**{"units.vsc.averaging.over": "time_step",
               "units.vsc.pwm.sync": "synchronous"})
    gfl(**{"units.vsc.averaging.enable": 0,
           "units.vsc.pwm.sync": "synchronous"})


def test_time_step_averaging_needs_fixed_step_solver(gfl):
    with pytest.raises(
        ConfigError,
        match=re.escape("units.vsc.averaging.over = 'time_step' requires the fixed-step solver"),
    ):
        gfl(**{"units.vsc.averaging.over": "time_step",
               "simulation.solver.type": "adaptive",
               "simulation.solver.method": "DP45"})
    with pytest.raises(ConfigError, match=re.escape("units.vsc.averaging.over: 'carrier' not in")):
        gfl(**{"units.vsc.averaging.over": "carrier"})
