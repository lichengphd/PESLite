# PESLite

Time-domain simulation of power-electronic converters in Python.

- Networks of buses, lines, grid sources and any number of converters, defined in YAML
  simulation files, with run-time changes described as events.
- Grid-following (PLL, current loop, dc-voltage loop) and grid-forming control
  (PSC, droop, VSG, dVOC, matching), each loop on its own clock.
- Switching bridges (ideal switches at exact instants) and bridges averaged over the PWM period
  or solver step, selected independently for each converter.
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

# run your own simulation file
peslite case.pes
peslite ./cases/my-converter.pes

# A suffix may be omitted
peslite case

# bundled examples use the -example suffix
peslite gfl-example
peslite gfm-psc-example
peslite gfm-droop-example

# override any parameter by its dotted path
peslite gfl-example --set simulation.t_end=1 --set units.vsc.pwm.computation_delay.steps=1
peslite gfl-example --set simulation.solver.type=adaptive --set simulation.solver.method=DP45
peslite gfm-droop-example --set units.vsc.averaging.enable=0

# average every converter for this run; print progress with selected quantities
peslite gfl-example --averaging
peslite gfl-example --progress 0.1 --watch vsc.vdc_pu --watch vsc.i_c

# continue a run from its last saved state (or from time T with --initial-time T)
peslite gfl-example --out output/a
peslite gfl-example --initial output/a/states.csv --out output/b

# inspect a configuration
peslite gfl-example --list-states     # state names (the states.csv columns)
peslite gfl-example --ph-report       # port-Hamiltonian structure of the circuit
peslite gfl-example --resolved        # complete resolved simulation file
peslite --help
```

From Python (in this folder, or anywhere after `pip install -e .`):

```python
import peslite

p = peslite.load("case.pes", **{"simulation.t_end": 1.0})
r = peslite.Simulation(p).run()

r.states["vsc.dclink.u_C"]         # a state over time
r.plant["vsc.i_c"]                 # plant signals (complex space vectors)
r.control["vsc.id_pu"]             # controller log of unit "vsc"
r.final_states()                   # last row, usable as an initial state
r.summary                          # trips, alarms, peaks
r.save("output/run")               # states.csv, summary.json, simulation.pes
```

## Simulation files and events

A simulation file contains the model (`base`, `buses`, `branches`, `sources`, `units` and
`elements`), what happens during the run (`events`), simulation settings (`simulation`) and
optional text notes (`meta`). `simulation.initial` and `simulation.output` keep the initial-state
and result settings together with the solver settings. Names without a suffix are SI quantities,
names ending in `_pu` are per-unit quantities, and switches are written as 0 or 1.

`--resolved` prints the complete parameter set after defaults and command-line overrides are
applied. The same complete configuration is saved as `simulation.pes`; it can be loaded directly
to repeat the run.

Events use one common top-level mapping:

```yaml
events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 1.0}
  p_step: {type: set, t: 2.0, set: {units.vsc.ctrl.references.p_ref_pu: 0.8}}
  load_on: {type: connect, target: load, t: 3.0}
  line_trip: {type: disconnect, target: line, t: 4.0}
```

`connect` and `disconnect` operate on units, sources, branches or elements. `set` changes a
declared run-time parameter path and validates the resulting parameter set before it is used.
Every event time is an exact integration boundary: the interval ending there uses the old model,
and integration after it uses the updated model. A disconnected converter pauses its controller;
a converter that trips remains disconnected.

The built-in `load` element is a series R-L load from a bus to ground. It can be connected,
disconnected or retuned by events; a small impedance can be used to model a fault.

## Bridge models

Each converter uses a switching bridge unless its `averaging` section is enabled. This permits a
system to mix switching and averaging converters:

```yaml
units:
  vsc:
    averaging: {enable: 1, over: pwm_period}  # pwm_period | time_step
```

| Setting | Model |
|---|---|
| `enable: 0` | ideal switches at the exact carrier-comparison instants |
| `enable: 1, over: pwm_period` | duty ratios held as continuous bridge values over each PWM period; no carrier ripple |
| `enable: 1, over: time_step` | carrier on-fraction averaged over each fixed solver step |

`--averaging` enables averaging for every converter for one run without editing the file, while
preserving each converter's configured `over` value. It takes precedence over an `enable: 0`
command-line override. If the option changes at least one model, the default result directory is
`output/<name>-averaging`.

Time-step averaging requires a fixed-step solver and an asynchronous carrier. With PWM-period
averaging there is no carrier, so carrier phase and synchronisation settings have no effect.

## Example configurations

| File | Content |
|---|---|
| `gfl-example.pes` | Grid-following converter; every key is annotated |
| `gfm-psc-example.pes` | Grid-forming, power-synchronization control |
| `gfm-droop-example.pes` | Grid-forming, droop with virtual admittance and current loop |
| `gfm-vsg-example.pes` | Grid-forming, virtual synchronous generator |
| `gfm-dvoc-example.pes` | Grid-forming, dispatchable virtual oscillator control |
| `gfm-matching-example.pes` | Grid-forming, matching control |
| `two-converters-example.pes` | A grid-forming and a grid-following unit on one grid |

## Frequently used settings

| Path | Values |
|---|---|
| `simulation.t_end` | end time, s |
| `simulation.solver.type` / `.method` | `fixed`: `euler` \| `heun` \| `rk4`; `adaptive`: `RK45` \| `DOP853` \| `Radau` \| `BDF` \| `LSODA` \| `DP45` |
| `simulation.solver.dt` | maximum fixed step, s |
| `simulation.solver.subsystems` | own steps per subsystem, e.g. `{vsc.dclink: 10, pcc: 0.1}` |
| `simulation.output.period` | snapshot interval, s |
| `simulation.energy_check` | `warn` \| `strict` \| `off` |
| `simulation.progress` | `{enable: 1, period: 0.1, watch: [...]}`; CLI: `--progress`, `--watch` |
| `units.<u>.averaging` | `{enable: 1, over: pwm_period \| time_step}`; CLI: `--averaging` |
| `units.<u>.ctrl.type` | `gfl` \| `gfm` \| `custom` |
| `units.<u>.ctrl.loops.<loop>.period` | loop period, s |
| `units.<u>.meas.average` | `instantaneous` \| `window` (with `window`) |
| `units.<u>.pwm.method` / `.sync` | `spwm` \| `svpwm`; `asynchronous` \| `synchronous` |
| `units.<u>.pwm.computation_delay.steps` | duty-ratio computation delay, in PWM updates |
| `simulation.output.states` / `.signals` / `.energy` | which files are written |

## Output

| File | Content |
|---|---|
| `states.csv` | every state at each snapshot; any row can start a new run |
| `plant.csv`, `control.<unit>.csv` | plant signals and controller logs (`simulation.output.signals: 1`) |
| `energy.csv` | stored energy and power balance (`simulation.output.energy: 1`) |
| `summary.json` | run summary |
| `simulation.pes` | complete resolved simulation file; loading it repeats the run |

Output names use the same unit convention as input parameters: an SI value has no unit suffix,
while a per-unit value ends in `_pu`. Runtime state paths are entity-first: physical states are
`<entity>.*`, while converter internals are `<unit>.ctrl.*`, `<unit>.pwm.*` and `<unit>.meas.*`.
The global solver keeps `solver.*`. In the summary, switches such as `tripped` are 0 or 1, events
that did not occur are `null`, and alarms, port-Hamiltonian defaults and energy problems are lists.

`--progress SECONDS` prints simulated time and wall time at the configured interval.
`--watch NAME` appends a value to each line and may be repeated or given comma-separated names.
A watched name may be a `states.csv` column, a `plant.csv`/controller column, a state alias, or a
complex state without `.re`/`.im` to print its magnitude. Converters with the same signal use their
unit prefix, for example `vsc_1.vdc_pu` and `vsc_2.vdc_pu`. State names use the same entity-first
paths as `states.csv`; there is no separate `plant.*` domain. Existing public result columns take
precedence when a signal and a state have the same name. For example:

```bash
peslite gfl-example --progress 0.1 \
  --watch vsc.vdc_pu \
  --watch vsc.i_c,vsc.u_dc
```

An unknown name is reported at the first progress line together with the available names.

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

The registered name can then be used at `units.<u>.ctrl.loops.<loop>.type`. Circuit element
types can likewise be registered with `register_element_type`, and event types with
`register_event_type`; each type owns a frozen parameter dataclass, so its file parameters use the
same parsing and validation as built-in types. A user solver may implement the normal solver call
alone; event-aware solvers may additionally provide `settle()` and `parameters_changed()` hooks.

A custom controller output stage can also be built with
`UniteType(cfg, scenario, pwm_method=..., limiter=...)`. Other replaceable parts are
`<unit>.modulator`, `<unit>.pwm.computation_delay` and `solver`. Executable examples of a custom loop, element,
event and user solver live in `tests/test_custom_parts.py`; the root `examples/` directory remains
simulation-data-only.

## Layout

```
pyproject.toml
src/peslite/            the package: __init__.py and four code parts
  components/           what the system is made of
    network.py            three-phase source, R-L branch, bus (R-C node), element types
    converter.py          bridge, dc link (capacitor, current or voltage source)
    adc.py                sampling of a converter's measurements, averaging window, oversampling
    pwm.py                PWM peripheral: publications, shadow-register computation delay, carrier, modulators
  control/              the converter's controller
    loops.py              what each loop type computes: its parameters, ports and update
    controller.py         controller interface, loop network, GFL/GFM wiring, UniteType
    protection.py         trip and alarm criteria
    modulation.py         output stage: voltage command to duty ratios, limiter, anti-windup
    blocks.py             transforms, filters and timers
  assembly/             a system built from a simulation file
    params.py             parameter classes, pu bases, runtime changes, file reading and writing
    validate.py           checks across the file's sections
    events.py             event types, connection ramps, unit setpoints and source scenarios
    unit.py               a converter unit: power stage, ADC, controller, PWM
    system.py             the network, units and elements as one model; applies events
  solver/               the numerical kernel and the run
    model.py              subsystems, their connections and named states
    energy.py             energy declarations, power balance, port-Hamiltonian report
    integrators.py        solver interface and single-rate integrators
    multirate.py          multirate integration and make_solver
    splitbound.py         error estimate of a multirate split
    simulation.py         run, records, result files and command line
tests/                  development test suite (python -m pytest); not included in the wheel
examples/               bundled *-example.pes files (YAML); no Python files
```

The controller depends only on the solver kernel; components depend on the controller interface;
assembly builds a system from both; `solver.simulation` runs the assembled system. The YAML files
under the root `examples/` directory use the `.pes` extension and are wheel data, so the named
`*-example` cases remain available
after installing only the wheel.

## License

GNU Affero General Public License v3.0 only (`AGPL-3.0-only`). See `LICENSE`.
