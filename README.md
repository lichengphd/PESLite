# PESLite

**[Documentation](https://github.com/lonaparte/PESLite/wiki)**
· **[Examples](https://github.com/lonaparte/PESLite/tree/main/examples)**
· **[PyPI](https://pypi.org/project/peslite/)**

PESLite is an open-source, lightweight, multirate power electronics simulator in Python. It is designed for time-domain simulation of power electronic converters and converter-based systems, supporting switching and averaged models, digital control, and flexible numerical integration for efficient simulations across multiple time scales. It provides:

- grid-following and grid-forming control in sampled or continuous form;
- switching, PWM-period-averaged and ideal averaged bridge models, selectable per converter;
- fixed-step, adaptive and multirate integration;
- ADC/PWM timing, converter-owned protection, energy accounting and restart from saved states;
- YAML simulation files with buses, lines, sources, loads, converters and timed events;
- export of a configured simulation as a standalone C++17 simulator.

The power circuit uses SI units (V, A, H, F and ohm), controller quantities ending in `_pu` are
per-unit values, time is in seconds and angles are in radians.

## Installation

```bash
pip install peslite
```

Python 3.10 or newer is required.

## Running simulations

Run a simulation file, a bundled `*-example`, or the first bundled example when no name is given:

```bash
peslite case.pes
peslite gfl-example
peslite
```

Simulation files are YAML text; the `.pes` extension is conventional but not required. Results are
written to `output/<name>/` by default. Dotted paths override individual parameters:

```bash
peslite case.pes --set simulation.t_end=1.0 --out output/test
```

`--switching`, `--pwm-averaging` and `--averaging` select a bridge/solver preset for every unit.
An explicit `--set` has higher priority than a preset, which has higher priority than the file.
Progress can be printed with `--progress SECONDS` and extended with repeatable `--watch NAME`.
Use `peslite --help` for the complete command-line interface; `--resolved`, `--list-states` and
`--ph-report` provide configuration and model diagnostics.

The same workflow is available from Python:

```python
import peslite

p = peslite.load("case.pes", **{"simulation.t_end": 1.0})
r = peslite.Simulation(p).run()       # output/run

r.states["vsc.dclink.u_C"]
r.plant["vsc.i_conv_a"]
r.ctrl["vsc.id_pu"]
r.final_states()
r.summary
```

Histories are read from the final CSV files only when requested; they are not retained in memory
during the simulation.

## Exporting a C++ simulator

`peslite-convert` exports one self-contained `peslite.cpp`. Parameters following the simulation
file remain variable in the compiled simulator; `all` selects every parameter supported by that
configuration and backend:

```bash
peslite-convert case.pes simulation.t_end units.vsc.ctrl.references.p_ref_pu
# or: peslite-convert case.pes all
```

The default destination is `export/<name>/peslite.cpp`. Compile it with any C++17 compiler:

```bash
c++ -O3 -DNDEBUG -std=c++17 export/case/peslite.cpp -o export/case/peslite
```

The executable follows the normal PESLite command conventions for configuration overrides and
output selection:

```bash
export/case/peslite --list-params
export/case/peslite --config another.pes \
  --set simulation.t_end=5.0 --out output/test
```

`--list-params` reports the parameters retained by this particular export. At run time, compiled
defaults have the lowest priority, `--config` overrides them, and `--set` has the highest priority.
Values in a supplied configuration that were not exported as variables are warned about and
ignored. Attempting to set such a path reports that the parameter is not variable.

The generated source uses only the C++17 standard library; it does not need Python, NumPy, SciPy or
a YAML library. The compiled simulator writes the same `states.csv`, signal files, `summary.json`
and resolved `simulation.pes` as the Python command. Exporting is also available through
`peslite.export(..., "cpp", variables=[...])` or `Simulation.export()`.

## Simulation files and events

A simulation file has model sections (`base`, `buses`, `branches`, `sources`, `units` and optional
`elements`), an `events` mapping, `simulation` settings and optional `meta` text. Initial values,
solver settings and output settings live under `simulation`. Switches are written as `0` or `1`.

`--resolved` prints the complete parameter tree after defaults and overrides. Every run saves that
same resolved tree as `simulation.pes`, which can be loaded to repeat the run.

Events share one mapping and may connect or disconnect an entity, or change validated run-time
parameters:

```yaml
events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 1.0}
  p_step: {type: set, t: 2.0, set: {units.vsc.ctrl.references.p_ref_pu: 0.8}}
  line_trip: {type: disconnect, target: line, t: 4.0}
```

Every event time is an exact integration boundary: the interval ending there uses the old model
and the following interval uses the updated model. Units, sources, branches and registered elements
use their own connect, disconnect and retune operations. A tripped converter remains disconnected.

## Bridge, control and solver models

Each converter selects its bridge independently with `units.<name>.bridge.model`:

| Model | Behaviour |
|---|---|
| `switching` | Ideal switches evaluated at exact carrier-comparison instants. |
| `pwm_averaging` | Active duty ratios drive a continuous bridge between PWM register loads. |
| `averaging` | Ideal controlled voltage source with controller states integrated continuously. |

`pwm_averaging` is the file default. It preserves the sampled controller, ADC window, PWM timer,
active/shadow registers and computation timing of switching while omitting carrier ripple. Ideal
`averaging` removes the ADC/PWM schedule and integrates controller equations with the plant;
sampled-only settings may remain in a shared configuration and are silently ignored in this mode.

Fixed and adaptive single-rate solvers are available, along with multirate subsystem steps. Ideal
averaging normally uses adaptive DP45, while the two sampled bridge presets use fixed-step RK4.
Instantaneous feedback loops are detected from port dependencies and solved internally, including
feedback between custom continuous control loops; users do not configure manual loop breaks.

Controllers receive SI measurements and return duty ratios or continuous voltage commands.
Startup controls gate release and reference ramping, while all fast and sampled protection belongs
to the converter `Unit`. A trip blocks the gates, opens the AC terminal and disconnects the DC
source for the rest of the run.

## Output, progress and restart

| File | Content |
|---|---|
| `states.csv` | State snapshots; any saved row can initialize another run. |
| `plant.csv`, `ctrl.<unit>.csv` | Optional plant and controller signals. |
| `energy.csv` | Optional stored-energy and power-balance history. |
| `summary.json` | Stop condition, solver counts, protection and energy summary. |
| `simulation.pes` | Complete configuration actually used for the run. |

Enabled histories stream directly to their final files in bounded batches. Disabling state output
still retains the terminal state for `final_states()`, and disabling energy output suppresses only
the history, not the summary checks.

`simulation.initial.t` and `simulation.t_end` are used exactly rather than rounded to a converter
timer grid. Restart restores saved physical and controller state, PWM registers, ADC accumulators
and timers as applicable. Intermediate oversampled ADC samples are not states, so such a run should
be continued from a control interrupt.

Runtime paths are entity-first: physical states use `<entity>.*`, converter internals use
`<unit>.ctrl.*`, `<unit>.pwm.*` and `<unit>.meas.*`, and global solver state uses `solver.*`.
Complex states may be watched by name to display their magnitude. Unit prefixes distinguish the
same quantity on different converters.

## Extending PESLite

Control loops, circuit elements and events can be registered with `register_loop_type`,
`register_element_type` and `register_event_type`. Each registered type owns a frozen parameter
dataclass, so custom file parameters use the normal parser and validation. A custom loop may provide
sampled updates, continuous state derivatives, or both. User solvers may additionally implement
`settle()` and `parameters_changed()` for event-aware operation.

Executable custom-type examples live in the tests. The root `examples/` directory intentionally
contains only YAML simulation files, which are included in the wheel and remain available by their
`*-example` names after installation.

## Development

The package uses a `src/` layout:

- `components/` contains network, converter, ADC and PWM models;
- `control/` contains loops, controller graphs and modulation;
- `assembly/` validates configuration and builds systems, events and exporters;
- `solver/` contains model evaluation, integration, energy checks and simulation I/O.

Run the test suite with:

```bash
python -m pytest
```

## License

GNU Affero General Public License v3.0 only (`AGPL-3.0-only`). See `LICENSE`.
