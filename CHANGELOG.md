# Changelog

## 0.1.1

### Package layout

- The repository is laid out as a PyPI package: `pyproject.toml`, the package in `src/peslite/`,
  the tests in `tests/` (`python -m pytest`; `pip install -e ".[test]"` installs pytest), and
  bundled simulation files in the root `examples/` directory. That directory contains YAML only. The command is
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
