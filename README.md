# PESLite

Time-domain simulation of power-electronic converters in Python.

- Networks of buses, lines, grid sources and any number of converters, defined in YAML
  simulation files, with run-time changes described as events.
- Grid-following (PLL, current loop, dc-voltage loop) and grid-forming control
  (PSC, droop, VSG, dVOC, matching), in sampled or continuous form.
- Switching, PWM-period-averaged and ideal continuously averaged bridges, selected independently
  for each converter.
- Fixed-step, adaptive (SciPy or built-in DP45) and multirate integration.
- Closed sampled controllers with ADC/PWM timing, or controller states integrated directly with
  the plant for ideal averaging, plus unit-owned protection.
- Energy accounting of the power circuit and restart from any saved state.

The power circuit works in SI units (V, A, H, F, ohm); controllers work in pu of each
converter's own base. Time is in s, angles in rad.

## Installation

```bash
pip install peslite
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

# override any parameter by its dotted path
peslite gfl-example --set simulation.t_end=1 --set units.vsc.ctrl.computation=2e-6
peslite gfl-example --set simulation.solver.type=adaptive --set simulation.solver.method=DP45
peslite gfm-psc-example --set units.vsc.bridge.model=switching

# override the bridge model for every converter in this run
peslite gfl-example --switching
peslite gfl-example --pwm-averaging
peslite gfl-example --averaging

# print progress with selected quantities
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

From Python after `pip install peslite`:

```python
import peslite

p = peslite.load("case.pes", **{"simulation.t_end": 1.0})
r = peslite.Simulation(p).run()     # streams directly to output/run

r.states["vsc.dclink.u_C"]         # a state over time
r.plant["vsc.i_conv_a"]            # a plant signal in the final CSV schema
r.ctrl["vsc.id_pu"]                # controller log of unit "vsc"
r.final_states()                   # last row, usable as an initial state
r.summary                          # trips, alarms, peaks
r.files                            # files already written by run()
```

## Export as C++

Export a resolved configuration as a standalone, parameter-specialised C++17 simulator:

```bash
peslite gfl-example --export cpp
c++ -O3 -DNDEBUG -std=c++17 export/gfl-example/peslite.cpp \
  -o export/gfl-example/peslite
export/gfl-example/peslite
```

Paths after the format remain variable in the compiled program. `peslite-convert` is the shorter
C++-specific spelling, and `all` retains every parameter supported by that export:

```bash
peslite gfl-example --export cpp simulation.t_end units.vsc.ctrl.references.p_ref_pu
peslite-convert gfl-example simulation.t_end units.vsc.ctrl.references.p_ref_pu
peslite-convert gfl-example all
```

The executable uses the same override form. A `.pes` file may provide the retained values; other
values in that file are ignored with a warning, and command-line `--set` has priority:

```bash
export/gfl-example/peslite --list-params
export/gfl-example/peslite --config run.pes \
  --set simulation.t_end=5.0 --out output/test
```

The export directory initially contains only the self-contained `peslite.cpp` translation unit.
The generated executable uses only the C++17 standard library. It performs the simulation itself
and writes the same `states.csv`, optional signal CSVs, `summary.json` and `simulation.pes` file
layout as the Python runner. Its default result directory is `output/<configuration-name>`; an
explicit directory may be passed as its first argument.

The same export is available from Python. With no directory argument it mirrors the normal
`output/run` convention under `export/run`:

```python
project = peslite.export(p, "cpp", variables=["simulation.t_end"])
# equivalently: peslite.Simulation(p).export("cpp", variables=["simulation.t_end"])
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
and integration after it uses the updated model. A unit connection is a host run command to its
controller; while stopped, integrating loops hold and the PWM is blocked through its normal
register path. A converter that trips remains disconnected.

The built-in `load` element is a series R-L load from a bus to ground. It can be connected,
disconnected or retuned by events; a small impedance can be used to model a fault.

## Controller and converter hardware

A converter controller is a closed discrete-time block. At each control interrupt it receives one
SI `Measurement` and returns `ControlOutput` with duty ratios, PWM enable, start-up completion and
synchronization data. It does not own or decide protection. Its only host input affecting operation
is `command(run, ramp)`.

The controller's `Startup` state counts its own interrupts. A connect command releases PWM with the
first computed duty word, ramps the active-power setpoint, and reports when the ramp ends; the unit
uses that status to arm its sampled protection. A disconnect command blocks PWM and resets that
ramp. Grid-forming synchronization laws track a usable terminal voltage before starting, avoiding
an artificial phase jump at connection.

All protection belongs to `Unit`. One protection subsystem owns both execution paths and their sole
trip latch: fast over-current checks run at switching and register-load instants, while voltage,
frequency, DC-voltage and ROCOF checks run from ADC/controller-rate samples. The unit arms the
latter when the controller reports that its start-up ramp is complete; the controller does not make
the trip decision. Any trip blocks the gates, opens the AC terminal and disconnects the DC source
for the rest of the run.

## Bridge models

Each converter selects its own bridge model, so all three models can share a system:

```yaml
units:
  vsc:
    bridge: {model: pwm_averaging}  # default; or averaging | switching
```

| `model` | Behaviour |
|---|---|
| `switching` | ideal switches at the exact carrier-comparison instants |
| `pwm_averaging` | the active duty ratios are continuous bridge values until the next PWM register load; no carrier ripple |
| `averaging` | an ideal controlled voltage source with continuous measurements and controller equations; no ADC or PWM timing |

When `bridge.model` is omitted, a converter uses `pwm_averaging`. `--switching`, `--pwm-averaging`
and `--averaging` are run presets: respectively `switching + fixed/rk4`,
`pwm_averaging + fixed/rk4` and `averaging + adaptive/DP45`. The priority is
`--set` > run preset > simulation file.

PWM averaging supports fixed and adaptive solvers and asynchronous or synchronous PWM. It keeps
the same timer, active/shadow registers, computation eligibility and single/double register-load
timing as switching. With single update a duty is held for one carrier period; with double update
it may load at each carrier valley and peak. An asynchronous carrier phase still shifts that
unit's control interrupts and loads, although the carrier waveform itself is not evaluated.

Ideal averaging keeps the same electrical bridge boundary, but connects the controller output
directly to the ideal controlled voltage source. Controller states join the plant ODE and terminal
measurements are evaluated at every solver stage. It constructs no ADC, control-interrupt grid,
computation wait, carrier, active/shadow register or PWM delay state. Consequently `ctrl.period`,
`ctrl.computation`, loop periods, ADC sampling/window settings and PWM timing settings are retained
in a loaded configuration but silently ignored in this mode; the same file can be reused unchanged
with `--switching` or `--pwm-averaging`.

Instantaneous feedback through component ports can form algebraic loops—for example terminal DC
voltage, controller command and bridge DC current. Model assembly detects these loops from declared
port dependencies, reduces each strongly connected group to the smallest connected feedback
variables it can tear, and solves it internally at each derivative evaluation. No loop-specific
ordering or manual break is configured, and the controller continues to measure the actual terminal
`u_dc` rather than a substituted storage voltage. The continuous control graph applies the same
strongly-connected-component rule to direct feedback between custom control loops; acyclic loop
networks retain their single-pass evaluation path.

## Example configurations

| File | Content |
|---|---|
| `gfl-example.pes` | Grid-following converter; every key is annotated |
| `gfm-psc-example.pes` | Grid-forming, power-synchronization control |
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
| `units.<u>.bridge.model` | `pwm_averaging` (default) \| `averaging` \| `switching`; CLI: `--pwm-averaging`, `--averaging`, `--switching` |
| `units.<u>.ctrl.type` | `gfl` \| `gfm` \| `custom` |
| `units.<u>.ctrl.period` / `.computation` | sampled-mode control-interrupt period and computation time, s; ignored by `averaging` |
| `units.<u>.ctrl.loops.<loop>.period` | sampled-mode loop period, an integer multiple of `ctrl.period`; ignored by `averaging` |
| `units.<u>.meas.period` / `.average` | sampled-mode ADC period and window-averaged channels; ignored by `averaging` |
| `units.<u>.pwm.update` | `single` (valleys) \| `double` (valleys and peaks) |
| `units.<u>.pwm.method` / `.sync` | `spwm` \| `svpwm`; `asynchronous` \| `synchronous` |
| `simulation.output.states` / `.signals` / `.energy` | which files are written |

## Output

| File | Content |
|---|---|
| `states.csv` | every state at each snapshot; any row can start a new run |
| `plant.csv`, `ctrl.<unit>.csv` | plant signals and controller logs (`simulation.output.signals: 1`) |
| `energy.csv` | stored energy and power balance (`simulation.output.energy: 1`) |
| `summary.json` | run summary |
| `simulation.pes` | complete resolved simulation file; loading it repeats the run |

`simulation.initial.t` and `simulation.t_end` are used exactly; neither is aligned or rounded to
a converter's timer grid. A saved row restores either the active/shadow PWM registers and ADC
averaging-window accumulators, or the continuous controller states of an ideal averaged unit, and
derives any next timer points from that row's time. An oversampled ADC's intermediate samples are not states, so its saved
run must be continued from a control interrupt.
With `simulation.output.states: 0`, no `states.csv` is written and only the terminal state is
collected internally, so `final_states()` remains available without serialising every snapshot.
All enabled histories are streamed directly to their final CSV files in bounded batches; a run
never accumulates the complete history in RAM. Python result columns are read from those final
files only when requested and are not part of the write path.
With `simulation.output.energy: 0`, energy checks still update the summary but their full time
history is not retained.

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

    def flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu,
             p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - p_pu)
        self.v_mag = v_ref_pu
        return {"theta": self.omega}
```

The registered name can then be used at `units.<u>.ctrl.loops.<loop>.type`. Circuit element
types can likewise be registered with `register_element_type`, and event types with
`register_event_type`; each type owns a frozen parameter dataclass, so its file parameters use the
same parsing and validation as built-in types. A custom loop used by ideal `averaging` implements
its continuous outputs and state derivatives (`flow()` for a `SyncLaw`); a sampled-only loop still
works with `switching` and `pwm_averaging`. A user solver may implement the normal solver call
alone; event-aware solvers may additionally provide `settle()` and `parameters_changed()` hooks.

A custom controller output stage can also be built with
`UniteType(cfg, pwm_method=..., limiter=...)`. Other replaceable parts are
`<unit>.modulator` and `solver`. Executable examples of a custom loop, element, event and user
solver live in `tests/test_custom_parts.py`; the root `examples/` directory remains
simulation-data-only.

## Layout

```
pyproject.toml
src/peslite/            the package: __init__.py and four code parts
  components/           what the system is made of
    network.py            three-phase source, R-L branch, bus (R-C node), element types
    converter.py          switching/PWM-averaged bridges, ideal continuous bridge and dc link
    adc.py                sampling of a converter's measurements, averaging window, oversampling
    pwm.py                PWM timer, duty registers, carrier and modulators used inside a bridge
  control/              the converter's controller
    loops.py              what each loop type computes: its parameters, ports and update
    controller.py         controller interface, Startup, loop network, GFL/GFM wiring, UniteType
    modulation.py         output stage: voltage command to duty ratios, limiter, anti-windup
    blocks.py             transforms, filters and timers
  assembly/             a system built from a simulation file
    params.py             parameter classes, pu bases, runtime changes, file reading and writing
    validate.py           checks across the file's sections
    events.py             event types, connection state/ramp metadata and source scenarios
    protection.py         fast and sampled criteria, timers, alarms and the unit's trip latch
    unit.py               power stage, ADC/PWM peripherals, controller, protection and trip actions
    system.py             the network, units and elements as one model; applies events
    exporter.py           format-neutral export API and backend registry
    cpp.py                parameter-specialised standalone C++17 simulator generator
  solver/               the numerical kernel and the run
    model.py              subsystems, connections, automatic algebraic-loop solving and named states
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
