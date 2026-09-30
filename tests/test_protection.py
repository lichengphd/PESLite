"""Protection runtime behavior and output semantics from the issue #2 reference tests."""

import peslite


def _run(params):
    return peslite.Simulation(params).run()


def test_disabled_overcurrent_never_trips(gfl):
    settings = {
        "units.vsc.protection.overcurrent.limit_pu": 1e-3,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.002,
    }
    assert _run(gfl(**settings)).tripped
    assert not _run(gfl(**settings, **{"units.vsc.protection.overcurrent.enable": 0})).tripped


def test_hold_zero_trips_only_while_the_criterion_is_met(gfl):
    settings = {
        "units.vsc.protection.dc_voltage.enable": 1,
        "units.vsc.protection.dc_voltage.limit_pu": 0.05,
        "units.vsc.protection.hold": 0.0,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.003,
        **{f"units.vsc.protection.{criterion}.enable": 0
           for criterion in ("overcurrent", "undervoltage", "overvoltage", "frequency")},
    }
    assert not _run(gfl(**settings)).tripped

    with_event = gfl(**settings, **{
        "events.vdc_reference": {
            "type": "set",
            "t": 0.001,
            "set": {"units.vsc.control.references.vdc_ref_pu": 1.1},
        }
    })
    result = _run(with_event)
    assert result.summary["vsc.trip_cause"] == "vdc"
    assert result.summary["vsc.trip_time"] >= 0.001
