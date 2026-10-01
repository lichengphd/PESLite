"""Per-unit exact-switching and PWM-period-averaged bridge models."""

import io
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
    """Return ``{unit: bridge model}`` after resolving command-line arguments."""
    output = io.StringIO()
    with redirect_stdout(output):
        assert peslite.main([*args, "--resolved"]) == 0
    units = yaml.safe_load(output.getvalue())["units"]
    return {name: unit["bridge"]["model"] for name, unit in units.items()}


def test_switching_is_default_and_pwm_averaging_is_selected_per_unit():
    assert _models("gfl-example") == {"vsc": "switching"}
    assert _models("gfm-droop-example") == {"vsc": "pwm_averaging"}


@pytest.mark.parametrize("name", ["gfl-example", "gfm-droop-example", "two-converters-example"])
def test_pwm_averaging_flag_selects_every_unit(name):
    assert set(_models(name, "--pwm-averaging").values()) == {"pwm_averaging"}


def test_there_is_no_global_switching_flag(capsys):
    with pytest.raises(SystemExit):
        peslite.main(["gfm-droop-example", "--switching", "--resolved"])
    assert "unrecognized arguments: --switching" in capsys.readouterr().err


def _with_model(params, model):
    return params.replace(**{f"units.{name}.bridge.model": model for name in params.units})


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*-example.pes")), ids=lambda path: path.stem)
@pytest.mark.parametrize("model", ["switching", "pwm_averaging"])
def test_every_bundled_file_runs_with_every_completed_model(path, model, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = peslite.load(
            path,
            **{"simulation.t_end": 0.004, "simulation.solver.linearisations": 0},
        )
        params = _with_model(params, model)
    result = peslite.Simulation(params).run(out_dir=tmp_path / f"{path.stem}-{model}")
    assert np.isfinite(result.states["t"]).all()
    assert result.states["t"][-1] == pytest.approx(0.004)


def test_units_can_have_different_models(tmp_path):
    params = peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **{"simulation.t_end": 0.004,
           "simulation.solver.linearisations": 0,
           "units.vsc_1.bridge.model": "pwm_averaging"},
    )
    simulation = peslite.Simulation(params)
    models = tuple(type(simulation.units[name].pwm.modulator).__name__
                   for name in ("vsc_1", "vsc_2"))
    assert models == ("ZOH", "CarrierComparison")
    simulation.run(out_dir=tmp_path)


def test_pwm_averaging_flag_is_the_same_as_setting_the_file_value(tmp_path):
    out_flag, out_set = tmp_path / "flag", tmp_path / "set"
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", "--pwm-averaging", *SHORT,
                             "--out", str(out_flag)]) == 0
        assert peslite.main(["gfl-example", "--set", "units.vsc.bridge.model=pwm_averaging",
                             *SHORT, "--out", str(out_set)]) == 0
    assert (out_flag / "states.csv").read_text() == (out_set / "states.csv").read_text()
    assert peslite.load(out_flag / "simulation.pes").unit("vsc").bridge.model == "pwm_averaging"


def test_pwm_averaging_flag_wins_over_set():
    assert _models("gfl-example", "--pwm-averaging",
                   "--set", "units.vsc.bridge.model=switching") == {
                       "vsc": "pwm_averaging"}


@pytest.mark.parametrize("path", ["units.vsc.averaging.enable", "units.vsc.bridge.over"])
def test_removed_averaging_paths_are_rejected(gfl, path):
    with pytest.raises(ConfigError, match="unknown"):
        gfl(**{path: 1})


def test_pwm_averaging_flag_results_use_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(peslite.solver.simulation, "_RESULTS", tmp_path)
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", *SHORT]) == 0
        assert peslite.main(["gfl-example", "--pwm-averaging", *SHORT]) == 0
        assert peslite.main(["gfm-droop-example", "--pwm-averaging", *SHORT]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gfl-example", "gfl-example-pwm-averaging", "gfm-droop-example-pwm-averaging"]
    assert peslite.load(tmp_path / "gfl-example" / "simulation.pes").unit("vsc").bridge.model == "switching"
    assert (peslite.load(tmp_path / "gfl-example-pwm-averaging" / "simulation.pes")
            .unit("vsc").bridge.model == "pwm_averaging")


@pytest.mark.parametrize("solver", [
    {},
    {"simulation.solver.type": "adaptive", "simulation.solver.method": "DP45"},
], ids=["fixed", "adaptive"])
def test_pwm_averaging_supports_fixed_and_adaptive_solvers(gfl, solver, tmp_path):
    params = gfl(**solver)
    result = peslite.Simulation(params).run(out_dir=tmp_path / ("adaptive" if solver else "fixed"))
    assert result.t[-1] == pytest.approx(params.simulation.t_end)


def test_pwm_averaging_accepts_synchronous_pwm(gfl, tmp_path):
    params = gfl(**{"units.vsc.pwm.sync": "synchronous",
                    "units.vsc.pwm.carrier_phase": 0.3})
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    assert result.t[-1] == pytest.approx(params.simulation.t_end)
