# Changelog

## 0.1.4

### The controller as a block (#1)

- A converter controller is now one closed sampled block. Each interrupt gives it an ADC
  `Measurement` including the gate driver's fault; it returns duty ratios, PWM enable and any
  trip. Connect/disconnect events reach it only as `command(run, ramp)`, so it no longer reads the
  plant or absolute event schedule.
- `Sequencer` owns start/stop, an interrupt-counted power-setpoint ramp and the arming of sampled
  protection. Its named states are `<unit>.ctrl.sequence.run`, `.ramp` and `.steps`. Integrating
  loops hold while stopped, and grid-forming laws track the terminal voltage before starting.
- PWM starts blocked. Its enable travels with the duty ratios in the active and shadow registers
  (`<unit>.pwm.on`, `<unit>.pwm.shadow.on`); there is no plant-derived start-up duty and no separate
  pending register.
- The AC breaker, gate enable and instantaneous `OvercurrentComparator` belong to `Unit`. The
  comparator checks plant phase current at switching/load instants, trips immediately even during
  start-up, and supplies a latched digital fault to the controller. A trip blocks gates, opens the
  AC terminal and disconnects the DC source permanently.
- A unit acts on the continuous model only through the run's `Plant` interface. At a coincident
  instant all units protect first, all sample next, and all actuate last, so sampling is independent
  of unit registration order.
- Summary trip/current fields now come from the unit while sampled protection statistics come from
  the controller. The split-bound linearisation keeps sequencer counters and PWM modes fixed.
- Removed `UnitScenario`, controller access to scenarios, `initial_duty`, `align_startup`,
  `fast_check`, `Protection.check_current` and the compatibility `ctrl.update()` entry point.

### Digital control timing and continuation (#4)

- Each converter owns its control-interrupt, PWM-load and ADC timer grids. A control result is
  written to its sole PWM shadow register at its interrupt and becomes eligible for an active-register
  load after `ctrl.computation`; `pwm.update` selects valley-only or valley-and-peak loads.
- PWM state paths are `<unit>.pwm.d_*` for the active compare registers and
  `<unit>.pwm.shadow.d_*` for the shadow registers. Computation progress is derived from the saved
  time and timer grid; there is no separate queue, `pending`, `ready` or configurable delay step.
- Runs start at `simulation.initial.t` and end at `simulation.t_end` exactly. Continuing from a
  state-table row restores PWM registers, computation progress, timer positions and ADC averaging
  windows. An oversampled ADC continues from an interrupt because its intermediate samples are not
  states.
- Each averaged ADC channel keeps one window accumulator (`<unit>.meas.x_*`); the redundant
  absolute integral and window-opening copy (`x_*_open`) are removed.
- All enabled result histories are streamed directly to their final CSV files in batches of 1000
  rows instead of accumulating in RAM. `simulation.solver.write_length` changes that batch size;
  Python result columns read those final files on demand and are not part of the write path.
- Repeated port-Hamiltonian energy audits use a precompiled topology and run at the first, final
  and every `simulation.solver.phs_check_step`-th snapshot (1000 by default).
- With `simulation.output.states: 0`, only the terminal state row is retained for
  `final_states()`. Disabled energy output keeps the energy summary checks but no energy history.
- Controller logs consistently use the abbreviation: `r.ctrl`, `ctrl.<unit>.csv` and
  `simulation.output.ctrl_every`; the old full-word interfaces are removed.

### Per-unit bridge models (#3)

- The bridge model is selected independently for each converter. A unit uses exact switching by
  default; `units.<u>.averaging: {enable: 1, over: pwm_period}` averages over each PWM period, and
  `over: time_step` averages the carrier comparison over each fixed solver step.
- `--averaging` enables averaging for every unit for one run and takes precedence over `--set` of
  its `enable` switch. When it changes a model, the default result directory gains the
  `-averaging` suffix. Different units in the same system may use different models.
- Time-step averaging requires a fixed-step solver and an asynchronous carrier. Carrier phase and
  synchronisation settings are ignored by PWM-period averaging. `StepAveragedCarrier` is renamed
  to `TimeStepAveragedCarrier`.

### Progress lines and watched quantities (#3)

- `simulation.progress: {enable, period, watch}` replaces `simulation.progress_every`.
  `--progress SECONDS` enables the lines; repeatable `--watch NAME` arguments, including
  comma-separated names, replace the file's watch list.
- Watched names may be state-table columns, plant/controller result columns, state aliases, or a
  complex state without `.re`/`.im` to report its magnitude. State paths are entity-first, with no
  `plant.*` domain. Unknown names report the known set at the first progress line, and observing
  values does not change the simulation trajectory.

### Packaging and verification

- Runtime states use entity-first paths: `<entity>.*`, `<unit>.ctrl.*`, `<unit>.pwm.*` and
  `<unit>.meas.*`; global solver bookkeeping remains `solver.*`. Unit configuration uses the same
  abbreviations under `units.<unit>.ctrl` and `units.<unit>.meas`; old functional-first state names
  and the `control`/`measurement` configuration keys are not aliases.
- PWM register states use `<unit>.pwm.d_*` and `<unit>.pwm.shadow.d_*`; timing is derived rather
  than exposed as clock, queue or generic `delay.*` states.
- Project version is 0.1.4. The root `examples/` directory remains data-only, keeps the
  `*-example.pes` names and is included in the wheel. The PEP 639 `AGPL-3.0-only` metadata is
  unchanged.
- Added Issue #3 acceptance coverage for every bundled file under all three bridge models,
  mixed-model systems, CLI precedence and output paths, solver/PWM compatibility, progress values,
  aliases, complex magnitudes and run invariance.

## 0.1.2

### Issue #2 simulation-file and event model

- Simulation files now keep `initial` and `output` under `simulation`; output-period settings are
  part of `simulation.output`. Parameter names follow the SI/no-suffix and per-unit/`_pu`
  convention, switches resolve to 0 or 1, and derived defaults continue to follow `--set` and
  `Params.replace` changes. `peslite.dumps()` returns and `peslite.dump()` writes complete resolved
  files.
- One top-level `events` mapping provides `connect`, `disconnect` and `set`. Target names are
  unique across buses, branches, sources, units and elements. Runtime paths are validated after
  every `set`, and custom event types can be registered with `register_event_type`.
- Components support connection changes and parameter retuning. Controllers retain their named
  state when their parameters change. The built-in `load` element is an event-capable series R-L
  load, and custom element types can be registered with `register_element_type`.
- Every event time is an exact integration boundary. User solvers may implement `settle()` and
  `parameters_changed()`; multirate runs preserve their window grid after an early event boundary.
  Disconnected converters pause control, and tripped converters remain disconnected.

### Output, examples and packaging

- State, control-signal and summary names use the same SI/per-unit convention as inputs.
  Controller states use `<unit>.ctrl.<state>`. Missing events are `None`/JSON `null`, switches
  are 0 or 1, and alarms, port-Hamiltonian defaults and energy problems are lists.
- Results save the fully resolved configuration as `simulation.pes`; `--resolved` prints the same
  representation, and both resolved and result-saved files can be run again.
- The single root `examples/` directory contains only YAML-format `*-example.pes` files. They are
  included in the wheel and remain discoverable by the installed CLI. Custom Python examples are
  executable tests instead of files under `examples/`.
- Project version is 0.1.2. The existing PEP 639 `AGPL-3.0-only` metadata is retained.

### Verification

- Added migration and acceptance coverage for parameter resolution, output semantics, events,
  protection, R-L loads, registered custom types, fixed/adaptive/multirate/user solvers, exact
  event boundaries, continuation, resolved-file reruns and installed-wheel example discovery.

## 0.1.1

### Package layout

- The repository is laid out as a PyPI package: `pyproject.toml`, the package in `src/peslite/`,
  the tests in `tests/` (`python -m pytest`; `pip install -e ".[test]"` installs pytest), and
  bundled simulation files in the root `examples/` directory. That directory contains only
  YAML-format `*-example.pes` files. The command is
  `peslite` after installation; names such as `peslite gfl-example` resolve from the installed
  wheel, while other files are given by path. `peslite.py` and the former Python example scripts
  are removed. The release workflow runs the tests before building and checks the packaged examples.
- The package is `__init__.py` and four parts, each module with one role:

  | Part | Module | Role | Was |
  |---|---|---|---|
  | `components` | `network` | three-phase source, R-L branch, bus | `power.source`, `power.network` |
  | | `converter` | bridge, dc link | `power.converter`, `power.dclink` |
  | | `adc` | sampling, averaging window, oversampling, and their timing | `sensing`, part of `simulation` |
  | | `pwm` | PWM peripheral: publications, computation delay, carrier, modulators | `modulation.modulators`, `firmware.delay`, part of `simulation` |
  | `control` | `loops` | what each loop type computes | `control.*` (8 modules), `params.loops`, part of `params.schema` |
  | | `protection` | trip and alarm criteria | `protection.relay` |
  | | `modulation` | output stage: voltage command to duty ratios, limiter, anti-windup | `modulation.methods`, `firmware.limiter`, part of `assembly.unit` |
  | | `controller` | the controller as one block: interface, the loop network (wiring by typed ports, order, clocks, held values), gfl/gfm wiring, `UniteType` | `control.graph`, `assembly.protocols`, part of `assembly.unit` |
  | | `blocks` | transforms, filters, timers | `firmware.blocks`, `firmware.transforms` |
  | `assembly` | `params` | the file's parameters: classes, pu bases, construction from a mapping, reading and writing the files | `params.schema`, `params.base`, `params.io` |
  | | `validate` | checks across sections | `params.validate` |
  | | `events`, `unit`, `system` | event time functions, a converter unit, the network as one model | `assembly.*` |
  | `solver` | `model` | subsystems connected into one model; named states | `phs.model`, `phs.states`, `phs.containers`, part of `phs.protocols` |
  | | `energy` | energy declarations and accounting | `phs.energy`, part of `phs.protocols` |
  | | `integrators`, `multirate` | the solver interface and the integrators | `phs.solvers`, part of `phs.protocols` |
  | | `splitbound` | error estimate of a multirate split | `phs.splitbound` |
  | | `simulation` | the run, what it produces (records, result files) and the command line | `simulation`, `results` |

  The controller imports only the solver's kernel; the components import the controller's interface
  (`Measurement`, `ControlOutput`) and transforms; the assembly builds a system of both; the run
  (`solver.simulation`) runs what the assembly built. A test checks this.
- A loop type is one class (`peslite.control.Loop`): its parameters (`Params`), typed ports, role
  in the default wiring, update and named states. It replaces the pairs of an algorithm class and
  its adapter (`SRFPLL` and `PLLLoop`, `CurrentController` and `CCLoop`, ...) and the second registry
  of parameter classes (`params.loops.LOOP_SCHEMAS` beside `control.loops.LOOP_TYPES`); a parameter
  class is no longer split between a base in the schema and a loop class. The grid-forming laws
  share `SyncLaw`, which does their frame, setpoints and outputs.
  - `register_loop_type(cls)` registers a class (a class decorator); the default `gfl`/`gfm`
    wiring places a loop by its class's `role`, so a registered type takes the place of the built-in
    one of its role. `loop_overrides` is removed. The loop network (`ControlGraph`, now in
    `control.controller`) knows no particular loop.
  - A loop type's own checks are in its `Params` (`virtual_admittance`: `x_v_pu > 0`; `unit_delay`:
    no `initial_pu` for an angle or frequency).
- The controller is `UniteType` with its loop network, its protection and its output stage
  (`OutputStage`, was `ConverterFirmware`, which also held the protection and the scaling to pu);
  `UniteType(cfg, scenario, ...)` and `make_controller(cfg, scenario)` take the unit's scenario.
- A unit's ADC and PWM keep their own timing (samples, averaging window, publications, switching
  instants) and the PWM its duty ratios; the event loop no longer copies them into a bookkeeping
  object of its own. `Unit.pwm` holds the modulator and the computation delay (were `Unit.modulator`,
  `Unit.delay`), `Unit.adc` the sampler (were `Unit.measure`, `open_window`, `seed_window`,
  `accumulate`, `phase_currents`, `Unit.window`, `Unit.T_avg`, `T_s`, `T_samp`).
- A simulation file is read in one pass: each section's SI/pu inputs are converted where it is
  built, on the bases in force there (was a second walk over the file before the construction).
- Code that was written twice is written once: the RK stage arithmetic of the fixed-step and
  multirate solvers (`TABLEAUS`, `combine`, `stage_state`), the construction of a single-rate
  integrator (`integrator`, used by `make_solver` and the multirate windows), the check of a
  multirate step (`parse_step`, used by the solver and the validation), the multirate split's groups
  (`MultirateSolver.groups`, used by the energy check), `Re(e conj(f))` (`energy.re_product`), the
  walk along the carrier (duty fraction and carrier comparison), the four timed protection criteria,
  the reading of YAML and JSON files, the high-pass filter (a `LowPass1`), the check of the
  controller's input (once, where it is scaled to pu).
- Removed, having no use: `GFMParams`, `PSCParams`, `PLLParams`, `CurrentLoopParams`,
  `DCVoltageLoopParams`, `DroopParams`, `VSGParams`, `DVOCParams`, `MatchingParams` (the loop
  types' `Params`), `control.make_law`, `CurrentController.from_bandwidth`, `SyncOutput`,
  `SynchronizationLaw`, `UnitLike`, `SystemLike`, `SampledCarrier`, `Model.verify_energy`,
  `.defaulted`, `.undeclared`, `energy.verify`, `.declared`, `.default_spec`, `states.is_unset`,
  `transforms.rotate`, `.power`, `System.trip`, `.breaker_open`, `.breakers`, `Unit.presets`,
  `UnitParams.L`/`R`/`C`, `Params.L`/`R`/`C`, `DCBase.L`/`R`, `MultirateSolver.inner_names`,
  `.outer_names` and the `theta_seed` input of the synchronization laws.
- The results of a simulation file are unchanged: bit for bit on the bundled files and on the
  other settings (bridge models, averaging windows, oversampling, delays, synchronous PWM,
  multirate splits, adaptive solvers, trips, restarts).

### Simulation file parameters (#2)

- `initial` and `output` are inside `simulation`; the former `simulation.log` fields are now
  `simulation.output.period` and `simulation.output.ctrl_every`.
- Parameter names follow the SI/pu convention (no suffix for SI and `_pu` for per unit).
  `measurement.window_s`, current-loop `bw_hz`, VSG `h_s` and dVOC `kappa_rad` are now
  `window`, `bandwidth`, `h` and `kappa`.
- Protection criteria, the power measurement filter and the virtual-admittance current limit use
  nested `enable` switches. Switches are accepted and written as 1 (on) or 0 (off).
- Defaults derived from the system base, PWM frequency or nominal frequency continue to follow
  those values through `--set` and `Params.replace`; explicitly supplied values remain fixed.
- `meta` contains optional text-only `title` and `description` fields.
- `peslite.dumps` returns and `peslite.dump` writes a complete resolved simulation file.
  Run results contain this file as `simulation.pes`, and `peslite FILE --resolved` prints it.

### Unified event model (#2)

- Source and unit-specific event blocks are replaced by one top-level `events` mapping. Built-in
  event types are `connect`, `disconnect` and `set`; an event which acts on one object names it
  with `target`.
- Event types can be registered with `peslite.register_event_type`. A custom event owns its frozen
  parameter dataclass and an `apply(event, system, t)` method, and is parsed and validated like a
  built-in event.
- Names are unique across buses, branches, sources and units. Event targets, times and each
  target's connect/disconnect sequence are checked while the simulation file is loaded.
- `Scenario`, `UnitScenario` and `SourceScenario` describe connection ramps and source parameters
  over time. Each unit carries its targeted events in time order.
- A `set` event may change only declared runtime paths. Events are applied in time and file order
  to the values left by preceding events; the complete parameter tree is rebuilt and validated
  after every event. `Params.changes` exposes the resulting parameters and canonical SI paths.

### Event-capable components (#2)

- Network branches, sources, buses, converter units and DC sources expose connection or retuning
  operations. `System.switch()` and `System.apply()` route checked events to the affected parts and
  refresh their energy declarations after parameter changes.
- Control-reference changes take effect at the next controller update. A changed loop is rebuilt
  from its new parameters while retaining its named state and runtime counters; protection settings
  can be retuned without replacing the controller.
- The new top-level `elements` mapping contains typed circuit elements. `load` is the built-in
  series R-L load to ground; custom types register with `peslite.register_element_type` and declare
  their own parameter dataclass, buses, subsystems and optional event operations.
