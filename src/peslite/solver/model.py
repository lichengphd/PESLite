"""Continuous-time model: what a subsystem is, subsystems connected by signals into one model, and
named states.

A subsystem has ``state``, ``inp`` and ``out`` records (:class:`Bag`) and state derivatives; the
:class:`Model` packs the states of its subsystems into a flat real vector (two entries per complex
state) named ``<subsystem>.<state>`` and evaluates their outputs and connections in signal order.
Every part with a state (a model, a controller, a delay line, a solver) is :class:`Stateful`: it
gives its state by name, parts are composed by dotted names (``ctrl.vsc.pll``), flattened to the
real columns of ``states.csv`` and resolved from ``initial.states`` values.
"""

from __future__ import annotations

import keyword
import linecache
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import energy as _energy

__all__ = ["ConfigError", "Bag", "Empty", "OutputStage", "Subsystem", "Model", "GroupPlan",
           "Stateful", "join", "gather", "scatter", "flatten", "resolve", "expand_aliases", "assign"]


# ------------------------------------------------------------------ subsystems

class ConfigError(ValueError):
    """Invalid model or solver configuration."""


class Bag:
    """Record whose ``__slots__`` fields default to ``0.0`` unless given as keywords (``state``, ``inp``, ``out``)."""

    __slots__ = ()

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0.0))


class Empty(Bag):
    """Record with no fields."""

    __slots__ = ()


@dataclass(frozen=True)
class OutputStage:
    """One output method of a subsystem and its direct input dependencies.

    ``method``: name of a method taking ``t``; ``inputs``: ``inp`` fields it reads;
    ``outputs``: ``out`` fields it writes. Every output field needs exactly one producer.
    """

    method: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@runtime_checkable
class Subsystem(Protocol):
    """Continuous block with ``state``, ``inp`` and ``out`` records and state derivatives.

    ``state_names``: packing order (a complex state takes two real entries);
    ``outputs_need_inputs``: whether ``set_outputs`` reads ``inp``; optional ``output_stages``
    (tuple of :class:`OutputStage`) replaces ``set_outputs``.
    ``out`` and connected ``inp`` fields are current when ``rhs`` runs.
    """

    state: Any
    inp: Any
    out: Any
    state_names: ClassVar[tuple[str, ...]]
    outputs_need_inputs: ClassVar[bool]

    def set_outputs(self, t: float) -> None:
        """Update ``out`` from ``state`` (and ``inp`` if ``outputs_need_inputs``)."""

    def rhs(self, t: float) -> Sequence[complex | float]:
        """Time derivatives of the states, in ``state_names`` order."""


# ------------------------------------------------------------------ named states

@runtime_checkable
class Stateful(Protocol):
    """Named state access for initialization and continuation.

    ``get_state()`` returns ``{name: float | complex | bool}`` (``""`` names the part itself);
    ``set_state(values)`` accepts any subset of those names.
    """

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, values: Mapping[str, Any]) -> None: ...


def join(prefix: str, name: str) -> str:
    """Join two name parts with a dot; an empty part is dropped."""
    if not prefix:
        return name
    return f"{prefix}.{name}" if name else prefix


def gather(parts: Mapping[str, Any]) -> dict[str, Any]:
    """Return the states of the named parts with keys prefixed by the part names.

    Parts without ``get_state`` are skipped; an empty part name adds no prefix.
    """
    out: dict[str, Any] = {}
    for pname, part in parts.items():
        get = getattr(part, "get_state", None)
        if get is None:
            continue
        if pname:
            for key, value in get().items():
                out[f"{pname}.{key}" if key else pname] = value
        else:
            out.update(get())
    return out


def scatter(parts: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Pass each part the entries of ``values`` prefixed with its name."""
    if not values:
        return
    for pname, part in parts.items():
        if not isinstance(part, Stateful):
            continue
        own = {key: values[join(pname, key)] for key in part.get_state() if join(pname, key) in values}
        if own:
            part.set_state(own)


def flatten(states: Mapping[str, Any]) -> dict[str, float]:
    """Convert states to real columns: complex ``x`` -> ``x.re``, ``x.im``; flags -> 0.0 / 1.0."""
    out: dict[str, float] = {}
    for key, value in states.items():
        if isinstance(value, complex):
            out[key + ".re"] = value.real
            out[key + ".im"] = value.imag
        elif isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        else:
            out[key] = float(value)
    return out


def expand_aliases(given: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Rename alias keys (and their ``.re`` / ``.im`` parts) to canonical names."""
    out: dict[str, Any] = {}
    for key, value in given.items():
        for alias, canonical in aliases.items():
            if key == alias or key.startswith(alias + "."):
                key = canonical + key[len(alias):]
                break
        if key in out:
            raise ValueError(f"state {key!r} is given twice (directly and through an alias)")
        out[key] = value
    return out


def _convert(where: str, target: Any, value: Any, presets: Any, name: str = "") -> Any:
    if isinstance(value, str):
        word = value.strip()
        if callable(presets):  # presets(state name, keyword)
            try:
                value = presets(name or where, word)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc.args[0] if exc.args else exc}") from None
        elif word in presets:
            value = presets[word]
        else:
            known = ", ".join(sorted(presets)) or "none"
            raise ValueError(f"{where}: unknown keyword {value!r} (keywords here: {known})")
    if isinstance(value, (list, tuple)):
        if not isinstance(target, complex) or len(value) != 2:
            raise ValueError(f"{where}: a [re, im] pair only fits a complex state")
        return complex(float(value[0]), float(value[1]))
    if isinstance(target, complex):
        return complex(value)
    if isinstance(value, complex):
        raise ValueError(f"{where}: complex value {value} for a real state")
    if isinstance(target, bool):
        return bool(round(float(value)))
    return float(value)


def resolve(template: Mapping[str, Any], given: Mapping[str, Any],
            presets: Any = None, where: str = "simulation.initial.states") -> dict[str, Any]:
    """Convert the entries of ``given`` to the types of the matching ``template`` entries.

    ``template``: state name -> current value. ``given``: state names or ``x.re``/``x.im`` parts;
    values are a number, ``[re, im]``, 0/1 or true/false for a flag, or a keyword of ``presets``.
    ``presets``: mapping keyword -> value, or callable ``(state name, keyword) -> value``.
    Raises KeyError for unknown names.
    """
    presets = presets or {}
    out: dict[str, Any] = {}
    parts: dict[str, dict[str, float]] = {}
    for name, value in given.items():
        if name in template:
            out[name] = _convert(f"{where}.{name}", template[name], value, presets, name)
            continue
        base, _, part = name.rpartition(".")
        if part in ("re", "im") and isinstance(template.get(base), complex):
            parts.setdefault(base, {})[part] = _convert(f"{where}.{name}", 0.0, value, presets, base)
            continue
        known = ", ".join(flatten(template))
        raise KeyError(f"{where}: unknown state {name!r}. Known states: {known}")
    for base, pr in parts.items():
        if base in out:
            raise ValueError(f"{where}: {base!r} is given both whole and by its .re/.im parts")
        t = template[base]
        out[base] = complex(pr.get("re", t.real), pr.get("im", t.imag))
    return out


def assign(obj: Any, values: Mapping[str, Any], names: tuple[str, ...] | Mapping[str, str]) -> None:
    """Set attributes from named states.

    ``names`` is either a tuple whose state and attribute names agree, or a
    mapping from public state names to internal attribute names.
    """
    attrs = names if isinstance(names, Mapping) else {name: name for name in names}
    for key, value in values.items():
        if key not in attrs:
            raise KeyError(f"{type(obj).__name__} has no state {key!r} (states: {', '.join(attrs)})")
        setattr(obj, attrs[key], value)


# ------------------------------------------------------------------ the model

Connection = tuple[Any, str]


class GroupPlan:
    """Subsystem group and its evaluation plans (see :meth:`Model.groups`).

    ``mask``: the group's entries of the state vector. ``plan``: updates all member inputs.
    ``own``: member outputs and intra-group copies only. ``feed``: copies from outside the group.
    """

    __slots__ = ("members", "mask", "plan", "own", "feed", "inputs", "rhs")

    def __init__(self, members: tuple, mask: NDArray[np.bool_], plan: tuple, own: tuple, feed: tuple,
                 inputs: tuple, rhs: tuple) -> None:
        self.members, self.mask, self.plan, self.own, self.feed, self.inputs, self.rhs = (
            members, mask, plan, own, feed, inputs, rhs)


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def _auto_names(subsystems: Sequence[Any]) -> list[str]:
    """Snake-case names from class names; repeated classes are numbered from 1."""
    base = [_snake(type(s).__name__) for s in subsystems]
    total, seen = Counter(base), Counter()
    names = []
    for b in base:
        seen[b] += 1
        names.append(f"{b}_{seen[b]}" if total[b] > 1 else b)
    return names


class Model:
    """Connected subsystems evaluated as ``rhs(t, y)`` on a flat real state vector.

    ``subsystems``: sequence (named from class names) or mapping name -> subsystem;
    dots in names form namespaces.
    ``connections``: ``(dst, input) -> (src, output[, gain])`` or a list of these (summed).
    ``zoh_connections``: ``(dst, input) -> label`` for held inputs set by :meth:`set_zoh_input`.
    """

    def __init__(
        self,
        subsystems: Sequence[Subsystem] | Mapping[str, Subsystem],
        connections: Mapping[Connection, Connection | Sequence[Connection]],
        zoh_connections: Mapping[Connection, str] | None = None,
    ) -> None:
        if isinstance(subsystems, Mapping):
            names = [str(n) for n in subsystems]
            subsystems = list(subsystems.values())
        else:
            subsystems = list(subsystems)
            names = _auto_names(subsystems)
        for name in names:
            if not name or any(not part for part in name.split(".")):
                raise ValueError(f"subsystem name {name!r} must be non-empty and have no empty part "
                                 f"(a dot separates namespaces: 'vsc1.branch_f')")
        if len(set(names)) != len(names):
            raise ValueError(f"subsystem names must be unique, got {names}")
        if len(set(map(id, subsystems))) != len(subsystems):
            raise ValueError("a subsystem appears more than once")
        self.names: list[str] = names
        for sub in subsystems:
            if not isinstance(sub, Subsystem):
                raise TypeError(
                    f"{type(sub).__name__} does not satisfy the Subsystem protocol "
                    "(needs state/inp/out, state_names, outputs_need_inputs, set_outputs, rhs)")
        self.subsystems: list[Subsystem] = list(subsystems)
        self.connections = dict(connections)
        self.zoh_connections = dict(zoh_connections or {})
        self._zoh_targets: dict[str, list[tuple[Any, str]]] = {}
        for (dst, name), label in self.zoh_connections.items():
            self._zoh_targets.setdefault(label, []).append((dst.inp, name))
        self._check_connections()
        self._build_state_layout()
        self._build_evaluation_plan()
        self._compile()
        # energy declaration per subsystem (default if none declared)
        self.energy_specs: dict[str, _energy.EnergySpec] = {n: _energy.spec_of(s) for n, s in zip(names, subsystems)}

    # ---------------------------------------------------------------- setup
    def _check_connections(self) -> None:
        known = set(map(id, self.subsystems))
        for (dst, dst_name), src in self.connections.items():
            if id(dst) not in known:
                raise ValueError(f"connection target {type(dst).__name__} is not in the subsystem list")
            if not hasattr(dst.inp, dst_name):
                raise ValueError(f"{type(dst).__name__}.inp has no field {dst_name!r}")
            sources = src if isinstance(src, list) else [src]
            for entry in sources:
                s_obj, s_name = entry[0], entry[1]
                if id(s_obj) not in known:
                    raise ValueError(f"connection source {type(s_obj).__name__} is not in the subsystem list")
                if not hasattr(s_obj.out, s_name):
                    raise ValueError(f"{type(s_obj).__name__}.out has no field {s_name!r}")
        for (dst, dst_name) in self.zoh_connections:
            if not hasattr(dst.inp, dst_name):
                raise ValueError(f"{type(dst).__name__}.inp has no field {dst_name!r}")

    def _build_state_layout(self) -> None:
        """Map every state field to a slice of the flat real vector."""
        ops: list[tuple[Any, str, int, bool]] = []
        index = 0
        for sub in self.subsystems:
            for name in sub.state_names:
                value = getattr(sub.state, name)
                is_complex = isinstance(value, complex)
                ops.append((sub.state, name, index, is_complex))
                index += 2 if is_complex else 1
        self._state_ops = tuple(ops)
        self.n_states = index
        # per-subsystem packing of the rhs return values
        rhs_layout = []
        for sub in self.subsystems:
            entries = []
            for name in sub.state_names:
                for st, nm, idx, is_c in ops:
                    if st is sub.state and nm == name:
                        entries.append((idx, is_c))
                        break
            rhs_layout.append((sub, tuple(entries)))
        self._rhs_layout = tuple(rhs_layout)
        # stateful subsystems with complex flags, in vector order
        self._rhs_flags = tuple((sub, tuple(is_c for _idx, is_c in entries))
                                for sub, entries in rhs_layout if entries)
        self._rhs_plan = tuple((sub, flags, all(flags)) for sub, flags in self._rhs_flags)

    def _output_stages(self, sub: Subsystem, name: str):
        """Validate a subsystem's ``output_stages`` and return them, or None if absent."""
        stages = getattr(sub, "output_stages", None)
        if stages is None:
            return None
        if not isinstance(stages, (tuple, list)) or not stages:
            raise ValueError(f"{name}.output_stages must be a nonempty sequence of OutputStage declarations")
        produced, methods = set(), set()
        for stage in stages:
            if not isinstance(stage, OutputStage):
                raise TypeError(f"{name}.output_stages: expected OutputStage, got {type(stage).__name__}")
            where = f"{name}.output_stages[{stage.method!r}]"
            if not isinstance(stage.method, str) or not callable(getattr(sub, stage.method, None)):
                raise ValueError(f"{where}: method must name a callable on the subsystem")
            if stage.method in methods:
                raise ValueError(f"{where}: duplicate stage method")
            methods.add(stage.method)
            for label, fields, record in (("inputs", stage.inputs, sub.inp), ("outputs", stage.outputs, sub.out)):
                if not isinstance(fields, (tuple, list)) or any(not isinstance(f, str) for f in fields):
                    raise ValueError(f"{where}.{label}: expected a sequence of field names")
                if len(set(fields)) != len(fields):
                    raise ValueError(f"{where}.{label}: duplicate field")
                for field in fields:
                    if not hasattr(record, field):
                        raise ValueError(f"{where}.{label}: unknown field {field!r}")
            if not stage.outputs:
                raise ValueError(f"{where}: at least one output is required")
            if produced.intersection(stage.outputs):
                raise ValueError(f"{where}: an output has more than one producer")
            produced.update(stage.outputs)
        outputs = {f for f in dir(sub.out) if not f.startswith("_") and not callable(getattr(sub.out, f))}
        missing = outputs - produced
        if missing:
            raise ValueError(f"{name}.output_stages: outputs without a producer: {sorted(missing)}")
        return stages

    def _build_evaluation_plan(self) -> None:
        """Sort output-method calls and individual input copies by signal dependency."""
        names = {id(s): n for n, s in zip(self.names, self.subsystems)}
        connected = {id(s): [] for s in self.subsystems}
        for sub, field in self.connections:
            connected[id(sub)].append(field)

        operations, owners, dependencies, labels = [], [], [], []

        def add(operation, owner, sources, label):
            index = len(operations)
            operations.append(operation)
            owners.append((owner, sources))
            dependencies.append(set())
            labels.append(label)
            return index

        producers, legacy, stages = {}, {}, []
        for name, sub in zip(self.names, self.subsystems):
            declared = self._output_stages(sub, name)
            if declared is None:
                node = None
                if sub.outputs_need_inputs or sub.out is not sub.state:
                    node = add(("out", sub.set_outputs), sub, (), f"{name}.set_outputs")
                    stages.append((node, sub, connected[id(sub)] if sub.outputs_need_inputs else ()))
                legacy[id(sub)] = node  # state aliases need no output evaluation
            else:
                for stage in declared:
                    node = add(("out", getattr(sub, stage.method)), sub, (), f"{name}.{stage.method}")
                    stages.append((node, sub, stage.inputs))
                    for field in stage.outputs:
                        producers[id(sub), field] = node

        def producer(sub, field):
            key = (id(sub), field)
            if key in producers:
                return producers[key]
            if id(sub) in legacy:
                return legacy[id(sub)]
            raise ValueError(f"{names[id(sub)]}.out.{field}: no output stage produces this signal")

        copies = {}
        for (dst, field), spec in self.connections.items():
            entries = spec if isinstance(spec, list) else [spec]
            sources = tuple(entry[0] for entry in entries)
            if len(entries) == 1 and (len(entries[0]) < 3 or entries[0][2] == 1.0):
                src, src_field = entries[0][:2]
                operation = (dst.inp, field, src.out, src_field, None)
            else:
                terms = tuple((entry[0].out, entry[1], float(entry[2]) if len(entry) > 2 else 1.0)
                              for entry in entries)
                operation = (dst.inp, field, None, None, terms)
            node = add(("copy", operation), dst, sources, f"{names[id(dst)]}.inp.{field}")
            copies[id(dst), field] = node
            for entry in entries:
                source = producer(entry[0], entry[1])
                if source is not None:
                    dependencies[node].add(source)
        for node, sub, fields in stages:
            for field in fields:
                copy = copies.get((id(sub), field))
                if copy is not None:
                    dependencies[node].add(copy)

        successors = [[] for _ in operations]
        indegree = [len(deps) for deps in dependencies]
        for node, deps in enumerate(dependencies):
            for dep in sorted(deps):
                successors[dep].append(node)
        ready = deque(i for i, degree in enumerate(indegree) if degree == 0)
        order = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for nxt in successors[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(operations):
            blocked = [labels[i] for i, degree in enumerate(indegree) if degree]
            raise ValueError(f"algebraic loop between output stages and inputs: {blocked}")
        position = {node: i for i, node in enumerate(order)}
        self._plan = tuple(operations[i] for i in order)
        self._plan_owners = tuple(owners[i] for i in order)
        self._plan_dependencies = tuple(tuple(position[d] for d in sorted(dependencies[i])) for i in order)
        # staged subsystems appear once per output call
        aliases = [s for s in self.subsystems if id(s) in legacy and legacy[id(s)] is None]
        self.output_order = aliases + [owner for (kind, _), (owner, _) in zip(self._plan, self._plan_owners)
                                      if kind == "out"]

    def _compile(self) -> None:
        """Generate and compile ``_load``, ``_outputs``, ``rhs_list`` (source in ``compiled_source``)."""
        owners = {id(s): n for n, s in zip(self.names, self.subsystems)}
        env: dict[str, Any] = {}
        refs: dict[int, str] = {}

        def ref(obj: Any) -> str:
            key = refs.get(id(obj))
            if key is None:
                key = refs[id(obj)] = f"_{len(refs)}"
                env[key] = obj
            return key

        def plain(field: str) -> bool:
            return field.isidentifier() and not keyword.iskeyword(field)

        def get(obj: Any, field: str) -> str:
            return f"{ref(obj)}.{field}" if plain(field) else f"getattr({ref(obj)}, {field!r})"

        def put(obj: Any, field: str, value: str) -> str:
            return f"{ref(obj)}.{field} = {value}" if plain(field) else f"setattr({ref(obj)}, {field!r}, {value})"

        load = [put(st, name, f"complex(v[{idx}], v[{idx + 1}])" if is_c else f"v[{idx}]")
                for st, name, idx, is_c in self._state_ops]

        plan = []
        for (kind, item), (owner, _srcs) in zip(self._plan, self._plan_owners):
            if kind == "out":
                line = f"{ref(item)}(t)"
            else:
                dst, field, src, src_field, fanin = item
                if fanin is None:
                    value = get(src, src_field)
                else:
                    value = "0"
                    for out, name, gain in fanin:
                        value = f"({value} + {ref(gain)} * {get(out, name)})"
                line = put(dst, field, value)
            plan.append(f"{line}  # {owners[id(owner)]}")

        derive, packed = [], []
        for j, (sub, flags, _all_complex) in enumerate(self._rhs_plan):
            got = [f"d{j}_{q}" for q in range(len(flags))]
            derive.append(f"{', '.join(got)}, = {ref(sub)}.rhs(t)  # {owners[id(sub)]}")
            for d, is_c in zip(got, flags):
                packed += [f"{d}.real", f"{d}.imag"] if is_c else [d]

        def function(signature: str, body: list[str]) -> list[str]:
            return [f"def {signature}:"] + ["    " + line for line in body or ["pass"]]

        source = "\n".join(function("_load(v)", load) + function("_outputs(t)", plan)
                           + function("rhs_list(t, v)", load + plan + derive + [f"return [{', '.join(packed)}]"])) + "\n"
        filename = f"<peslite model {id(self):#x}>"
        exec(compile(source, filename, "exec"), env)
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        self.compiled_source = source
        self._load, self._outputs, self.rhs_list = env["_load"], env["_outputs"], env["rhs_list"]

    # -------------------------------------------------------------- groups
    def groups(self, assignment: Mapping[str, str], system: str = "system") -> dict[str, GroupPlan]:
        """Split subsystems into groups and return ``{label: GroupPlan}``.

        ``assignment`` maps subsystem names or namespaces (``"vsc1"`` for ``vsc1.*``) to
        group labels; unassigned subsystems go to ``system``.
        """
        resolved: dict[str, str] = {}
        for key, label in assignment.items():
            members = self.select(key)
            if not members:
                raise KeyError(f"unknown subsystem(s) or namespace(s) [{key!r}]; this model has {self.names}")
            for name in members:
                if resolved.get(name, label) != label:
                    raise KeyError(f"subsystem {name!r} is assigned to both {resolved[name]!r} and {label!r}")
                resolved[name] = label
        labels: dict[str, list] = {}
        for name, sub in zip(self.names, self.subsystems):
            labels.setdefault(resolved.get(name, system), []).append(sub)
        labels.setdefault(system, [])
        return {label: self._group_plan(members) for label, members in labels.items()}

    def _group_plan(self, members: Sequence[Any]) -> GroupPlan:
        member_ids = {id(m) for m in members}
        mask = np.zeros(self.n_states, dtype=bool)
        for st, name, idx, is_c in self._state_ops:
            if any(st is m.state for m in members):
                mask[idx:idx + (2 if is_c else 1)] = True

        def closure(seeds):
            needed = set(seeds)
            pending = list(needed)
            while pending:
                for dep in self._plan_dependencies[pending.pop()]:
                    if dep not in needed:
                        needed.add(dep)
                        pending.append(dep)
            return tuple(self._plan[i] for i in sorted(needed))

        member_ops = [i for i, (owner, _) in enumerate(self._plan_owners) if id(owner) in member_ids]
        plan = closure(member_ops)
        own = tuple(item for item, (owner, srcs) in zip(self._plan, self._plan_owners)
                    if id(owner) in member_ids and (item[0] == "out" or all(id(x) in member_ids for x in srcs)))
        boundary = [i for i, ((kind, _), (owner, srcs)) in enumerate(zip(self._plan, self._plan_owners))
                    if kind == "copy" and id(owner) in member_ids and any(id(x) not in member_ids for x in srcs)]
        inputs = tuple(self._plan[i] for i in boundary)
        # only the stages producing boundary signals
        feed = closure(boundary)
        rhs = tuple((sub, entries) for sub, entries in self._rhs_layout if entries and id(sub) in member_ids)
        return GroupPlan(tuple(members), mask, plan, own, feed, inputs, rhs)

    def run_plan(self, t: float, plan: tuple) -> None:
        """Run a plan (subset of the output plan) on the current subsystem records."""
        for kind, item in plan:
            if kind == "out":
                item(t)
            else:
                dst_inp, dst_name, src_out, src_name, fanin = item
                if fanin is None:
                    setattr(dst_inp, dst_name, getattr(src_out, src_name))
                else:
                    total = 0
                    for out, name, gain in fanin:
                        total += gain * getattr(out, name)
                    setattr(dst_inp, dst_name, total)

    def rhs_group(self, t: float, y: NDArray[np.float64], group: GroupPlan, own: bool = False) -> NDArray[np.float64]:
        """Return the derivatives of one group's states (zeros elsewhere) for the full state ``y``.

        ``own=True`` keeps inputs from outside the group unchanged.
        """
        self.set_states(y)
        self.run_plan(t, group.own if own else group.plan)
        dy = np.zeros(self.n_states)
        for sub, entries in group.rhs:
            for d, (idx, is_c) in zip(sub.rhs(t), entries):
                if is_c:
                    dy[idx] = d.real
                    dy[idx + 1] = d.imag
                else:
                    dy[idx] = d
        return dy

    # ------------------------------------------------------------- packing
    def get_initial_values(self) -> NDArray[np.float64]:
        y = np.empty(self.n_states)
        for st, name, idx, is_c in self._state_ops:
            v = getattr(st, name)
            if is_c:
                y[idx] = v.real
                y[idx + 1] = v.imag
            else:
                y[idx] = v
        return y

    def set_states(self, y: NDArray[np.float64]) -> None:
        self._load(y.tolist())

    def state_labels(self) -> list[str]:
        """One label per entry of the flat vector (``<subsystem>.<state>[.re|.im]``)."""
        labels = []
        for sname, (sub, entries) in zip(self.names, self._rhs_layout):
            for name, (_idx, is_c) in zip(sub.state_names, entries):
                base = f"{sname}.{name}"
                labels += [base + ".re", base + ".im"] if is_c else [base]
        return labels

    def select(self, prefix: str) -> list[str]:
        """Return the subsystem named ``prefix`` or all subsystems named ``prefix.*``."""
        head = prefix + "."
        return [n for n in self.names if n == prefix or n.startswith(head)]

    def name_of(self, sub: Subsystem) -> str:
        for name, s in zip(self.names, self.subsystems):
            if s is sub:
                return name
        raise KeyError(f"{type(sub).__name__} is not in this model")

    def get_state(self) -> dict[str, complex | float]:
        """Return all continuous states as ``{"<subsystem>.<state>": value}``.

        Reads the subsystem records; call :meth:`sync` or :meth:`set_states` first.
        """
        out: dict[str, complex | float] = {}
        for sname, sub in zip(self.names, self.subsystems):
            st = sub.state
            for name in sub.state_names:
                out[f"{sname}.{name}"] = getattr(st, name)
        return out

    def set_state(self, values: Mapping[str, Any]) -> None:
        """Write named states into the subsystem records (repack with :meth:`get_initial_values`)."""
        index = {f"{sname}.{name}": (sub.state, name)
                 for sname, sub in zip(self.names, self.subsystems) for name in sub.state_names}
        for key, value in values.items():
            if key not in index:
                raise KeyError(f"unknown model state {key!r}; known: {', '.join(index)}")
            st, name = index[key]
            setattr(st, name, complex(value) if isinstance(getattr(st, name), complex) else float(value))

    def state_slice(self, sub: Subsystem, name: str) -> slice:
        """Position of a subsystem's state field in the flat vector."""
        for st, nm, idx, is_c in self._state_ops:
            if st is sub.state and nm == name:
                return slice(idx, idx + (2 if is_c else 1))
        raise KeyError(f"{type(sub).__name__} has no state {name!r}")

    # --------------------------------------------------------- evaluation
    def set_zoh_input(self, label: str, value: Any) -> None:
        """Set a held external input (e.g. the switching state) by label."""
        for inp, name in self._zoh_targets[label]:
            setattr(inp, name, value)

    def evaluate_outputs(self, t: float) -> None:
        """Run the ordered output-method / input-copy plan for the current states."""
        self._outputs(t)

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``dy/dt`` at time ``t`` (s) for the flat real state vector ``y``."""
        return np.array(self.rhs_list(t, y.tolist()))

    def sync(self, t: float, y: NDArray[np.float64]) -> None:
        """Load ``y`` into the subsystems and refresh all outputs (for sampling/logging)."""
        self._load(y.tolist())
        self._outputs(t)

    # ---------------------------------------------------------------- energy
    def energy_balance(self, t: float, y: NDArray[np.float64]) -> "_energy.EnergyReport":
        """Energy accounting at ``(t, y)`` from the subsystems' declarations (see :mod:`peslite.solver.energy`)."""
        return _energy.balance(self, t, y)

    def ph_report(self, zoh: Mapping[str, Any] | None = None, groups: Mapping[str, Any] | None = None,
                  rtol: float = 1e-8, hold: Mapping[str, float] | None = None) -> "_energy.PHReport":
        """Return the port-Hamiltonian report of the model (:class:`peslite.solver.energy.PHReport`).

        ``groups``: result of :meth:`groups`; ``hold``: ``{"step": dt, "window": W}`` hold times in s.
        """
        return _energy.ph_report(self, zoh=dict(zoh) if zoh else None, rtol=rtol,
                                 groups=dict(groups) if groups else None, hold=dict(hold) if hold else None)
