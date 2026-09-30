# PESLite

Time-domain simulation of power-electronic converters in Python.

- Networks of buses, lines, grid sources and any number of converters, defined in YAML.
- Grid-following (PLL, current loop, dc-voltage loop) and grid-forming control
  (PSC, droop, VSG, dVOC, matching), each loop on its own clock.
- Switching (ideal switches, exact switching instants), averaged and step-averaged bridges.
- Fixed-step, adaptive (SciPy or built-in DP45) and multirate integration.
- ADC sampling (instantaneous or window average), computation delay, PWM, protection.
- Energy accounting of the power circuit and restart from any saved state.

The power circuit works in SI units (V, A, H, F, ohm); controllers work in pu of each
converter's own base. Time is in s, angles in rad.

## Installation

```bash
python -m pip install -e .            # numpy, scipy, PyYAML
```

Python 3.10 or newer.

## Quick start

```bash
# run the first bundled example (sorted by name)
peslite

# run a bundled example by name; results go to output/<name>/
peslite gfl-example
peslite gfm-psc-example --out output/psc

# run your own configuration
peslite case.yaml
peslite ./cases/my-converter.yaml

# A suffix may be omitted
peslite case

# bundled examples use the -example suffix
peslite gfl-example
peslite gfm-psc-example
peslite gfm-droop-example

# override any parameter by its dotted path
peslite gfl-example --set simulation.t_end=1 --set units.vsc.delay.steps=1
peslite gfl-example --set simulation.solver.type=adaptive --set simulation.solver.method=DP45
peslite gfm-droop-example --set simulation.bridge=switching

# continue a run from its last saved state (or from time T with --initial-time T)
peslite gfl-example --out output/a
peslite gfl-example --initial output/a/states.csv --out output/b

# inspect a configuration
peslite gfl-example --list-states     # state names (the states.csv columns)
peslite gfl-example --ph-report       # port-Hamiltonian structure of the circuit
peslite --help
```

From Python (in this folder, or anywhere after `pip install -e .`):

```python
import peslite

p = peslite.load("case.yaml", **{"simulation.t_end": 1.0})
r = peslite.Simulation(p).run()

r.states["plant.vsc.dclink.u_C"]   # a state over time
r.plant["vsc.i_c"]                 # plant signals (complex space vectors)
r.control["vsc.id_pu"]             # controller log of unit "vsc"
r.final_states()                   # last row, usable as an initial state
r.summary                          # trips, alarms, peaks
r.save("output/run")               # states.csv, summary.json, params.yaml
```

## Example configurations

| File | Content |
|---|---|
| `gfl-example` | Grid-following converter; every key is annotated |
| `gfm-psc-example` | Grid-forming, power-synchronization control |
| `gfm-droop-example` | Grid-forming, droop with virtual admittance and current loop |
| `gfm-vsg-example` | Grid-forming, virtual synchronous generator |
| `gfm-dvoc-example` | Grid-forming, dispatchable virtual oscillator control |
| `gfm-matching-example` | Grid-forming, matching control |
| `two-converters-example` | A grid-forming and a grid-following unit on one grid |

## Frequently used settings

| Path | Values |
|---|---|
| `simulation.t_end` | end time, s |
| `simulation.bridge` | `switching` \| `averaged` \| `step_averaged` |
| `simulation.solver.type` / `.method` | `fixed`: `euler` \| `heun` \| `rk4`; `adaptive`: `RK45` \| `DOP853` \| `Radau` \| `BDF` \| `LSODA` \| `DP45` |
| `simulation.solver.dt` | maximum fixed step, s |
| `simulation.solver.subsystems` | own steps per subsystem, e.g. `{vsc.dclink: 10, pcc: 0.1}` |
| `simulation.log.plant_period` | snapshot interval, s |
| `simulation.energy_check` | `warn` \| `strict` \| `off` |
| `units.<u>.control.type` | `gfl` \| `gfm` \| `custom` |
| `units.<u>.control.loops.<loop>.period` | loop period, s |
| `units.<u>.measurement.average` | `instantaneous` \| `window` (with `window_s`) |
| `units.<u>.pwm.method` / `.sync` | `spwm` \| `svpwm`; `asynchronous` \| `synchronous` |
| `units.<u>.delay.steps` | computation delay in PWM updates |
| `output.states` / `.signals` / `.energy` | which files are written |

## Output

| File | Content |
|---|---|
| `states.csv` | every state at each snapshot; any row can start a new run |
| `plant.csv`, `control.<unit>.csv` | plant signals and controller logs (`output.signals: true`) |
| `energy.csv` | stored energy and power balance (`output.energy: true`) |
| `summary.json`, `params.yaml` | run summary and the full parameter set |

## Custom parts

A control loop type is one class: it owns its parameter dataclass, typed ports,
role in the default wiring, update and named state. Register the class before
loading a configuration that names its type:

```python
from dataclasses import dataclass
from peslite.control import SyncLaw, register_loop_type

@register_loop_type
class MyLaw(SyncLaw):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        k_p_pu: float
        type: str = "my_law"

    type = "my_law"

    def step(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
             p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        self.v_mag = v_ref_pu
```

The registered name can then be used at
`units.<u>.control.loops.<loop>.type`. A custom controller output stage can
also be built with `UniteType(cfg, scenario, pwm_method=..., limiter=...)`.
Other replaceable parts are `<unit>.modulator`, `<unit>.delay`, `solver`,
and extra circuit elements through `System(p, elements=[...])`.

## Layout

```
pyproject.toml
src/peslite/            the package: __init__.py and four code parts
  components/           what the system is made of
    network.py            three-phase source, R-L branch, bus (R-C node)
    converter.py          bridge, dc link (capacitor, current or voltage source)
    adc.py                sampling of a converter's measurements, averaging window, oversampling
    pwm.py                PWM peripheral: publications, computation delay, carrier, modulators
  control/              the converter's controller
    loops.py              what each loop type computes: its parameters, ports and update
    controller.py         controller interface, loop network, GFL/GFM wiring, UniteType
    protection.py         trip and alarm criteria
    modulation.py         output stage: voltage command to duty ratios, limiter, anti-windup
    blocks.py             transforms, filters and timers
  assembly/             a system built from a simulation file
    params.py             parameter classes, pu bases, construction, file reading and writing
    validate.py           checks across the file's sections
    events.py             time functions of events
    unit.py               a converter unit: power stage, ADC, controller, PWM
    system.py             the network and its units as one model
  solver/               the numerical kernel and the run
    model.py              subsystems, their connections and named states
    energy.py             energy declarations, power balance, port-Hamiltonian report
    integrators.py        solver interface and single-rate integrators
    multirate.py          multirate integration and make_solver
    splitbound.py         error estimate of a multirate split
    simulation.py         run, records, result files and command line
tests/                  development test suite (python -m pytest); not included in the wheel
examples/               bundled *-example YAML files; no Python files
```

The controller depends only on the solver kernel; components depend on the controller interface;
assembly builds a system from both; `solver.simulation` runs the assembled system. The YAML files
under the root `examples/` directory are wheel data, so the named `*-example` cases remain available
after installing only the wheel.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See `LICENSE`.
