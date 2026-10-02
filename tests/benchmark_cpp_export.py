"""Manual Python/C++ timing comparison; prints measurements and has no performance assertion."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import peslite


def benchmark_cpp_export(config: str | Path, work: str | Path) -> list[dict[str, float | str]]:
    """Build and time all bridge presets for one simulation file."""
    compiler = next((path for name in ("c++", "g++", "clang++")
                     if (path := shutil.which(name)) is not None), None)
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required")
    root, rows = Path(work), []
    for mode in ("switching", "pwm_averaging", "averaging"):
        solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
        loaded = peslite.load(config)
        params = loaded.replace(**{
            **{f"units.{name}.bridge.model": mode for name in loaded.units},
            "simulation.solver.type": solver[0], "simulation.solver.method": solver[1],
            "simulation.solver.subsystems": {}, "simulation.progress.enable": 0,
        })
        project = root / mode
        peslite.export(params, "cpp", project, name=mode)
        executable = project / "peslite"
        start = time.perf_counter()
        subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                        str(project / "peslite.cpp"), "-o", str(executable)], check=True)
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        peslite.Simulation(params).run(out_dir=project / "python-output")
        python_seconds = time.perf_counter() - start
        start = time.perf_counter()
        subprocess.run([str(executable), str(project / "cpp-output")], check=True)
        cpp_seconds = time.perf_counter() - start
        rows.append({"mode": mode, "compile_s": compile_seconds,
                     "python_s": python_seconds, "cpp_s": cpp_seconds,
                     "speedup": python_seconds / cpp_seconds})
    return rows
