"""Standalone C++ export: API, CLI layout and numerical smoke tests."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

import peslite
from conftest import EXAMPLES


def _compiler() -> str | None:
    return next((path for name in ("c++", "g++", "clang++")
                 if (path := shutil.which(name)) is not None), None)


def _quick(mode: str):
    solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
    return peslite.load(EXAMPLES / "gfl-example.pes").replace(**{
        "units.vsc.bridge.model": mode,
        "simulation.solver.type": solver[0],
        "simulation.solver.method": solver[1],
        "simulation.solver.subsystems": {},
        "simulation.t_end": 0.202,
        "simulation.output.period": 1e-3,
        "simulation.energy_check": "off",
    })


def test_python_export_api_uses_export_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = peslite.export(_quick("pwm_averaging"), "cpp")
    assert result.out_dir == Path("export/run")
    assert {path.name for path in result.files} == {
        "peslite.cpp", "simulation.pes", "CMakeLists.txt", "README.md",
    }
    assert "Standalone C++17" in (result.out_dir / "peslite.cpp").read_text()


def test_cli_export_preserves_configuration_and_mode_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert peslite.main([str(EXAMPLES / "gfl-example.pes"), "--averaging",
                         "--export", "cpp"]) == 0
    project = tmp_path / "export/gfl-example-averaging"
    assert (project / "peslite.cpp").is_file()
    resolved = peslite.load(project / "simulation.pes")
    assert resolved.unit("vsc").bridge.model == "averaging"
    assert (resolved.simulation.solver.type, resolved.simulation.solver.method) == (
        "adaptive", "DP45")


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_runs_all_bridge_modes(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick(mode)
    project = tmp_path / mode
    peslite.export(params, "cpp", project, name=mode)
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)

    def final(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}

    expected, actual = final(python_out / "states.csv"), final(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    tolerance = 2e-5 if mode == "averaging" else 2e-8
    assert max(abs(actual[key] - expected[key]) for key in expected) <= tolerance * scale


def test_cpp_export_rejects_an_already_run_simulation(tmp_path):
    simulation = peslite.Simulation(_quick("pwm_averaging"))
    simulation.run(out_dir=tmp_path / "output")
    with pytest.raises(RuntimeError, match="already-run"):
        simulation.export("cpp", tmp_path / "export")
