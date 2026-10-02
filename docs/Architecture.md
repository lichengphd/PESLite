# Architecture

PESLite separates physical components, controller logic, system assembly and numerical execution.
The separation keeps model ownership explicit while allowing the same configuration to run with
different bridge and solver implementations.

## Package layers

```text
src/peslite/
├── components/   network, converter, ADC and PWM hardware
├── control/      control loops, graphs, startup and modulation
├── assembly/     parameters, events, protection, units, systems and exporters
└── solver/       connected model kernel, energy model, integrators and simulation run
```

Dependencies point inward through small interfaces: components use the model kernel, controller
logic uses typed signals, assembly wires both into a system, and `solver.simulation` schedules the
assembled system.

## Components

`components/network.py` contains grid sources, R-L branches, bus R-C nodes and registered element
types. `components/converter.py` provides the common lossless bridge boundary, three bridge
implementations and DC-link models. `components/adc.py` and `components/pwm.py` implement sampled
converter peripherals.

Physical component state and parameters are SI quantities. Components declare ports, state,
derivatives and energy/storage relationships without knowing the global system order.

## Control

Each loop owns its parameter schema, typed ports, state and sampled/continuous update. `ControlGraph`
checks wiring and creates an execution plan. Sampled graphs require an explicit delay to break a
cycle; continuous graphs identify strongly connected loop groups and solve only those cyclic
groups iteratively.

`Startup` owns run/stop and reference-ramp progress. The modulation output stage converts controller
voltage commands into duty ratios. Protection does not belong to the controller.

## Assembly

The assembly layer:

1. parses strict parameter dataclasses and converts per-unit inputs to SI;
2. validates topology, timing and cross-section constraints;
3. creates buses, branches, sources, elements and converter units;
4. registers globally unique entity-first state and signal names;
5. compiles model connection and output-evaluation plans;
6. builds event scenarios and validated parameter changes.

`Unit` owns one converter's power stage, controller, sensing, startup interface, protection latch,
breakers and gates. Both startup and protection act through the same hardware operations.

## Model kernel

`solver/model.py` packs subsystem state into one flat real vector, using two entries for a complex
state. The model compiles direct connection and output plans during construction; ordinary RHS
evaluation does not repeatedly merge arbitrary state dictionaries.

Port dependency analysis identifies algebraic groups. Acyclic outputs use one pass, while only
connected feedback variables are solved iteratively. The same mechanism supports ideal averaging
without manual case-specific loop breaks.

## Port-Hamiltonian information

Components can declare storage and power ports. The energy layer uses these declarations to:

- account for stored, supplied and dissipated energy;
- check Tellegen/power-balance residuals;
- produce the `--ph-report` structure view;
- evaluate multirate interface and hold diagnostics.

The port-Hamiltonian kernel is a structural and diagnostic layer; it does not force every solver to
use one integration method.

## Simulation execution flow

`Simulation.run()` repeatedly selects the next exact boundary among:

- requested end and output times;
- file events;
- converter ADC, control, PWM load and switching instants;
- multirate window boundaries.

The solver advances only to that boundary. The simulation then settles deferred work, applies file
events, evaluates fast protection, samples every due unit, actuates every unit and records output.
This ordering gives all units the same pre-actuation view at a shared instant.

## State and continuation

State ownership is fixed during construction. The public registry combines physical, controller,
ADC/PWM, protection and solver state into unique entity-first paths. Output rows and restart use the
same schema, while reconstructible timer indices and combinational outputs are derived from saved
time/state rather than stored redundantly.

## C++ generation

The C++ backend consumes the already assembled `Simulation`, lowers its fixed model/output plans to
straight-line C++ and specializes fixed configuration values. The result is one translation unit;
runtime-variable parameters become typed fields, not a generic dictionary in the RHS path.

See [C++ Export](Cpp-Export.md) for supported solvers and customization limits.
