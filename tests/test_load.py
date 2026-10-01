"""Built-in R-L load and switching events in a running network."""

import numpy as np

import peslite
from peslite.assembly.params import from_dict


def _current(result, name):
    return np.abs(result.states[f"{name}.re"] + 1j * result.states[f"{name}.im"])


def _two_buses(case, events):
    tree = case()
    tree["buses"]["b2"] = {"c_pu": 0.02, "r_d_pu": 0.5}
    tree["branches"] = {
        "line": {"from_bus": "pcc", "to_bus": "b2", "x_pu": 0.05}
    }
    tree["elements"] = {
        "load": {"type": "load", "bus": "b2", "r_pu": 1.0, "x_pu": 0.5}
    }
    tree["events"] = events
    tree["simulation"]["output"] = {"period": 2.5e-5}
    return from_dict(tree)


def test_branch_source_and_load_switch_during_a_run(case):
    params = _two_buses(case, {
        "line_off": {"type": "disconnect", "target": "line", "t": 0.001},
        "line_on": {"type": "connect", "target": "line", "t": 0.002},
        "load_on": {"type": "connect", "target": "load", "t": 0.0015},
        "grid_off": {"type": "disconnect", "target": "grid", "t": 0.003},
    })
    result = peslite.Simulation(params).run()
    t = result.states["t"]
    line = _current(result, "line.i")
    load = _current(result, "load.branch.i")
    grid = _current(result, "grid.branch.i")
    assert np.all(line[(t > 0.001 + 1e-9) & (t < 0.002 - 1e-9)] == 0.0)
    assert np.max(line[t > 0.0022]) > 0.0
    assert np.all(load[t < 0.0015 - 1e-9] == 0.0)
    assert np.max(load[t > 0.0022]) > 0.0
    assert np.all(grid[t > 0.003 + 1e-9] == 0.0)
    assert np.max(grid[t < 0.003]) > 0.0


def test_small_load_can_model_a_switched_fault(case):
    tree = case()
    tree["elements"] = {
        "fault": {"type": "load", "bus": "pcc", "r_pu": 0.01, "x_pu": 0.01}
    }
    tree["events"] = {
        "fault_on": {"type": "connect", "target": "fault", "t": 0.002}
    }
    tree["simulation"].update(t_end=0.004, output={"period": 2.5e-5})
    result = peslite.Simulation(from_dict(tree)).run()
    t = result.states["t"]
    voltage = _current(result, "pcc.u_C")
    assert np.max(voltage[t > 0.0035]) < 0.2 * np.min(voltage[t < 0.002])
