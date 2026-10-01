"""State paths and state-table columns are fixed when a simulation is assembled."""

import pytest

from peslite.solver import ConfigError, StateRegistry


class _States:
    def __init__(self, **values):
        self.values = values
        self.loads = []

    def get_state(self):
        return dict(self.values)

    def set_state(self, values):
        self.loads.append(dict(values))
        self.values.update(values)


def test_registry_fixes_names_columns_and_owners_at_construction():
    plant = _States(i=1 + 2j)
    ctrl = _States(enabled=True, integral=3.0)
    states = StateRegistry({"vsc": plant, "vsc.ctrl": ctrl})

    assert states.names == ("vsc.i", "vsc.ctrl.enabled", "vsc.ctrl.integral")
    assert states.columns == (
        "vsc.i.re", "vsc.i.im", "vsc.ctrl.enabled", "vsc.ctrl.integral"
    )
    assert states.boolean_names == {"vsc.ctrl.enabled"}

    plant.values["i"] = 4 + 5j
    assert states.read_flat() == {
        "vsc.i.re": 4.0,
        "vsc.i.im": 5.0,
        "vsc.ctrl.enabled": 1.0,
        "vsc.ctrl.integral": 3.0,
    }
    states.load({"vsc.ctrl.integral": 7.0}, ["vsc.ctrl"])
    assert ctrl.loads == [{"integral": 7.0}]
    assert plant.loads == []


def test_registry_rejects_duplicate_paths_and_flat_columns_during_construction():
    with pytest.raises(ConfigError, match="state name 'vsc.x'.*both"):
        StateRegistry({"vsc": _States(x=1.0), "": _States(**{"vsc.x": 2.0})})

    with pytest.raises(ConfigError, match="state-table column 'x.re'.*both"):
        StateRegistry({"": _States(x=1 + 0j, **{"x.re": 2.0})})

    with pytest.raises(ConfigError, match="state prefix 'vsc'.*both"):
        StateRegistry([("vsc", _States(a=1.0)), ("vsc", _States(b=2.0))])


def test_registry_rejects_a_runtime_state_schema_change():
    part = _States(x=1.0)
    states = StateRegistry({"part": part})
    part.values["extra"] = 2.0

    with pytest.raises(ConfigError, match=r"changed after Simulation construction.*added: \['extra'\]"):
        states.read()
