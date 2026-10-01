"""Output names and values follow the simulation-file SI/pu and null rules."""

import json
import re

import numpy as np
import pytest

import peslite


UNIT_SUFFIX = re.compile(r"_(s|hz|hz_s|J|rad)$")


def _run(params):
    return peslite.Simulation(params).run()


def test_run_without_trip_uses_none_and_lists_for_absent_results(gfl, tmp_path):
    result = _run(gfl())
    summary = result.summary
    assert (summary["tripped"], summary["vsc.tripped"]) == (0, 0)
    assert summary["vsc.trip_time"] is None and summary["vsc.trip_cause"] is None
    assert summary["vsc.alarms"] == []
    assert summary["vsc.modulation_saturation_first_t"] is None
    for criterion in ("overcurrent", "undervoltage", "overvoltage", "frequency",
                      "dc_voltage", "rocof"):
        assert summary[f"vsc.{criterion}_first_t"] is None
    assert summary["t_start"] == 0.0 and summary["t_stop"] == pytest.approx(0.004)

    result.save(tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["vsc.trip_time"] is None
    assert isinstance(saved["vsc.alarms"], list)
    assert isinstance(saved["energy_problems"], list)
    assert "wall_time" in saved and "wall_time_s" not in saved


def test_trip_reports_time_cause_alarm_and_first_crossing(gfl):
    result = _run(gfl(**{
        "units.vsc.protection.overcurrent.limit_pu": 1e-3,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.002,
    }))
    summary = result.summary
    assert (summary["tripped"], summary["vsc.tripped"], summary["vsc.trip_cause"]) == (
        1, 1, "overcurrent"
    )
    assert summary["vsc.trip_time"] == pytest.approx(summary["vsc.overcurrent_first_t"])
    assert "OVERCURRENT" in summary["vsc.alarms"]


def test_state_control_and_summary_names_have_no_unit_suffix(gfl):
    result = _run(gfl(**{"simulation.output.signals": 1}))
    names = [*result.summary, *result.states, *result.control, *result.columns()]
    assert [name for name in names if UNIT_SUFFIX.search(name)] == []


def test_controller_states_in_pu_are_named_as_pu(examples):
    gfl_names = peslite.Simulation(peslite.load(examples / "gfl-example.pes")).state_names()
    psc_names = peslite.Simulation(peslite.load(examples / "gfm-psc-example.pes")).state_names()
    assert {
        "vsc.ctrl.pll.integral_pu",
        "vsc.ctrl.dvc.integral_pu",
        "vsc.ctrl.cc.integral_pu.re",
        "vsc.tripped",
    } <= set(gfl_names)
    assert {
        "vsc.ctrl.power.p_pu",
        "vsc.ctrl.power.q_pu",
        "vsc.ctrl.sync.v_int_pu",
    } <= set(psc_names)
    assert not {
        "vsc.ctrl.pll.integral",
        "vsc.ctrl.power.p",
        "vsc.breaker_open",
    } & set(gfl_names + psc_names)
    assert not any(name.startswith(("plant.", "ctrl.", "pwm.", "meas."))
                   for name in gfl_names + psc_names)


def test_grid_following_and_grid_forming_logs_share_names(gfl, examples):
    result_gfl = _run(gfl())
    result_gfm = _run(peslite.load(
        examples / "gfm-psc-example.pes",
        **{"simulation.t_end": 0.004, "events.connect_vsc.t": 0.0,
           "simulation.solver.linearisations": 0},
    ))
    for result in (result_gfl, result_gfm):
        assert {"vsc.freq_dev", "vsc.angle_rel", "vsc.vac_pu"} <= set(result.control)
    assert "vsc.id_ref_pu" in result_gfl.control
    assert "vsc.v_mag_pu" not in result_gfm.control


def test_continuation_uses_the_renamed_states(gfl, tmp_path):
    result = _run(gfl(**{"simulation.t_end": 0.004}))
    result.save(tmp_path)
    restarted = peslite.load(
        tmp_path / "simulation.pes",
        initial=tmp_path / "states.csv",
        initial_time=0.002,
    )
    continued = _run(restarted)
    index = int(np.argmin(abs(result.states["t"] - 0.003)))
    continued_index = int(np.argmin(abs(continued.states["t"] - 0.003)))
    assert continued.states["vsc.ctrl.pll.integral_pu"][continued_index] == pytest.approx(
        result.states["vsc.ctrl.pll.integral_pu"][index], rel=1e-9
    )
