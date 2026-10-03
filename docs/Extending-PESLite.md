# Extending PESLite

PESLite uses registries and small typed protocols for custom control loops, circuit elements and
events. Modules placed in `peslite/addons/controllers/` or `peslite/addons/components/` are imported
automatically when PESLite starts and register into the same type tables as built-in modules.
Types kept elsewhere can still be imported and registered explicitly before loading a simulation
file that names them.

Executable examples live in `tests/test_custom_parts.py`; Python implementation files are not
placed in the root `examples/` directory.

The bundled `custom-pll-example` demonstrates this layout: its `.pes` file remains in `examples/`,
while `peslite.addons.controllers.voltage_adaptive_pll` contains the executable custom loop. It
uses a filtered measured-voltage magnitude and grid-frequency state, then joins the same GFL graph
as the built-in current and DC-voltage loops.

## Custom control loops

Register a loop class with `register_loop_type`:

```python
from dataclasses import dataclass
from peslite.addons.controllers import SyncLaw, register_loop_type

@register_loop_type
class LaggedSync(SyncLaw):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        k_p_pu: float
        tau: float
        type: str = "lagged_sync"

    type = "lagged_sync"
    state_names = {"theta": "theta", "p_f_pu": "p_f"}

    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.p_f = 0.0

    def _sample(self, period, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.p_f += period / self.cfg.tau * (p_pu - self.p_f)
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.theta += period * self.omega
        self.v_mag = v_ref_pu

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu,
              p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.v_mag = v_ref_pu
        return self.omega, (p_pu - self.p_f) / self.cfg.tau
```

The nested frozen `Params` dataclass defines the file schema and defaults. A module stored under
`peslite/addons/controllers/` is discovered automatically; the registered name is then available at
`units.<u>.ctrl.loops.<loop>.type` exactly like a built-in loop.

A specialized loop may expose separate sampled and continuous-flow laws. When its dynamics can be
built from the standard blocks, define one `equation()` instead: the graph supplies its inputs,
collects the registered states and derivatives, and selects sampled or continuous block
implementations during construction. Typed ports still validate all connections before running.

The essential pattern is:

```python
from peslite.addons.controllers import Filter, Integrator, PI

def __init__(self, cfg, unit, startup):
    super().__init__(cfg, unit, startup)
    self.angle = self.state_block(
        "theta", Integrator(self.block_period, initial=0.0)
    )
    self.regulator = self.state_block(
        "integral_pu", PI(kp, ki, self.block_period)
    )
    self.feedback = self.state_block(
        "feedback_pu",
        Filter(num=(1.0,), den=(tau, 1.0), period=self.block_period),
    )

def equation(self, signal, omega, reference, feedforward):
    measured = self.feedback(signal)
    command = self.regulator(reference, measured, feedforward)
    theta = self.angle(omega)
    return command, theta
```

`Loop.block_period` is the configured loop period for sampled bridge models and `None` for ideal
averaging. Construction therefore selects a sampled or continuous implementation once; the call
itself has no mode branch. `Filter` follows the Transfer Fcn coefficient convention: numerator and
denominator coefficients are in descending powers of `s`, and the transfer function must be
proper. Its sampled implementation, and the sampled integral branch of `Integrator` and `PI`, use
the trapezoidal (Tustin) method. The complete bundled example is
`peslite.addons.controllers.voltage_adaptive_pll`.

## Custom circuit elements

An element class registered by `register_element_type` owns:

- a frozen `Params` dataclass including its `type` and `bus` or `busN` fields;
- `subsystems()` for its physical model objects;
- `connections()` for its internal and auxiliary signal wiring;
- `terminals()` binding each configured bus to an electrical `Terminal`;
- optional `connect()`, `disconnect()` or `retune()` behavior.

```python
from peslite.addons.components import Element, Terminal, register_element_type

@register_element_type
class MyElement(Element):
    type = "my_element"
    # Params and assembly methods...
```

The file can then use:

```yaml
elements:
  device: {type: my_element, bus: pcc}
```

Modules under `peslite/addons/components/` are discovered automatically. Built-in and add-on
elements follow the same registry mechanism. Public names are namespaced by the element instance,
so two instances do not share state paths.

An electrical `Terminal` names a subsystem voltage input and current output once. Its direction is
`+1` when positive current enters the subsystem and `-1` when it leaves. A one-terminal element
uses `bus` and returns one binding; a multi-terminal element uses `bus1`, `bus2`, and so on and
returns one binding per terminal. System assembly converts these declarations to direct model
connections, so terminal metadata adds no work to the integration loop.

## Custom events

Register an `Event` subclass with `register_event_type`. Its `Params` dataclass contains at least
`type` and `t`, and `apply(event, system, t)` performs the action:

```python
from dataclasses import dataclass
from peslite.assembly import Event, register_event_type

@register_event_type
class Marker(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        t: float
        type: str = "marker"

    type = "marker"

    @staticmethod
    def apply(event, system, t):
        pass
```

Custom events receive the same exact-boundary scheduling as built-in events.

## Custom solvers

A user solver is a callable receiving `(rhs, t0, t1, y0)` and returning `SolverStep`. It owns its
RHS count and must finish exactly at `t1`. Optional hooks are:

- `settle(t, y)`: finish deferred work before a model-changing event;
- `parameters_changed()`: refresh cached data after a `set` event;
- named state methods when the solver has continuation state.

Pass the object to `Simulation(params, solver=solver)`. A simple single-rate solver does not need
the optional hooks.

## Custom controllers and modulators

`UniteType` builds a controller from a parameter tree, PWM method and output stage. A replacement
controller satisfies the `Controller` protocol; a replacement modulator satisfies `Modulator` and
returns a `SwitchingSequence`. The unit checks these protocols at assembly time.

Keep protection decisions in the unit hardware layer. Controller output contains commands and
startup status; it should not directly open breakers or mutate the system.

## Validation and state rules

- Validate custom scalar relationships in `Params.__post_init__` by raising `ConfigError`.
- Declare stable public state names so restart and CSV output remain deterministic.
- Preserve named state when rebuilding a retuned object.
- Keep structural changes out of `set` events; expose only parameters that can be applied safely.
- Use SI inside physical components and explicit per-unit conversion at controller boundaries.

## C++ export

Python registration alone does not automatically generate C++. A custom subsystem, loop, event or
modulator needs corresponding C++ lowering support. The exporter rejects unsupported custom types
instead of silently changing their behavior.
