"""The registered event types of a simulation file and their time-dependent scenarios.

An event is one entry of the top-level ``events`` mapping: ``{type, t, ...}``. Events which act
on one named object use ``target``. The built-in types are ``connect``, ``disconnect`` and ``set``;
custom types are classes registered with :func:`register_event_type`.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_right
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional

from ..control.blocks import smoothstep
from ..solver.model import ConfigError

__all__ = [
    "NAMED", "SWITCHABLE", "SWITCHING", "Event", "EVENT_TYPES", "register_event_type",
    "Connect", "Disconnect", "Set", "events_for", "switching_schedule", "connected_at",
    "Scenario", "UnitScenario", "SourceScenario",
]

NAMED = ("buses", "branches", "sources", "units")
SWITCHABLE = ("branches", "sources", "units")
SWITCHING = ("connect", "disconnect")


class Event:
    """Base class of a registered event type.

    A type supplies ``type``, a frozen ``Params`` dataclass containing at least ``type`` and ``t``,
    and ``apply(event, system, t)``. The run will call ``apply`` at the event time once event-time
    scheduling is connected to the solver.
    """

    type: ClassVar[str]
    Params: ClassVar[type]

    @staticmethod
    def apply(event: Any, system: Any, t: float) -> None:
        raise NotImplementedError


EVENT_TYPES: dict[str, type] = {}


def register_event_type(cls: type) -> type:
    """Register an :class:`Event` class under ``cls.type`` and return the class."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("an event type needs a name: the class attribute 'type'")
    if name in EVENT_TYPES:
        raise ValueError(f"event type {name!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or not {"type", "t"} <= set(params.__dataclass_fields__):
        raise TypeError(f"event type {name!r}: Params must be a dataclass with type and t fields")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"event type {name!r}: Params.type must default to {name!r}")
    if getattr(cls, "apply", Event.apply) is Event.apply:
        raise TypeError(f"event type {name!r}: it needs apply(event, system, t), which runs it")
    EVENT_TYPES[name] = cls
    return cls


@register_event_type
class Connect(Event):
    """Connect ``target`` at ``t``; ``ramp`` is the optional ramp duration in seconds."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "connect"
        target: str
        t: float
        ramp: float = 0.0

        def __post_init__(self) -> None:
            if not (math.isfinite(self.ramp) and self.ramp >= 0.0):
                raise ConfigError(f"ramp must be finite and >= 0, got {self.ramp}")

    type = "connect"

    @staticmethod
    def apply(event: Any, system: Any, t: float) -> None:
        unit = getattr(system, "units", {}).get(event.target)
        if unit is not None and getattr(unit, "tripped", getattr(unit, "breaker_open", False)):
            warnings.warn(f"t = {t:.6g} s: unit {event.target!r} has tripped, so its connect event "
                          f"leaves it disconnected", stacklevel=4)
        system.switch(event.target, True, t, event.ramp)


@register_event_type
class Disconnect(Event):
    """Disconnect ``target`` at ``t``."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "disconnect"
        target: str
        t: float

    type = "disconnect"

    @staticmethod
    def apply(event: Any, system: Any, t: float) -> None:
        system.switch(event.target, False, t)


class Set(Event):
    """Set runtime-changeable parameter paths to new values at ``t``."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "set"
        t: float
        set: dict = field(default_factory=dict)

    type = "set"


# A set event is applied through the checked Change object held by Params.changes.
EVENT_TYPES["set"] = Set


def events_for(events: Mapping[str, Any], target: str, type: Optional[str] = None) -> list:
    """Return events aimed at ``target``, optionally restricted to ``type``, in time order."""
    found = [event for event in events.values()
             if getattr(event, "target", None) == target and (type is None or event.type == type)]
    return sorted(found, key=lambda event: event.t)


def switching_schedule(events: Mapping[str, Any]) -> dict[str, list[tuple[float, bool]]]:
    """Return each target's time-ordered ``(time, connected)`` steps."""
    out: dict[str, list[tuple[float, bool]]] = {}
    for event in events.values():
        if event.type in SWITCHING:
            out.setdefault(event.target, []).append((event.t, event.type == "connect"))
    return {target: sorted(steps) for target, steps in out.items()}


def connected_at(steps: list[tuple[float, bool]], t: float) -> bool:
    """Return a target's state at ``t``; before its first step it is in the opposite state."""
    state = not steps[0][1] if steps else True
    for event_t, connected in steps:
        if event_t > t + 1e-10:
            break
        state = connected
    return state


class Scenario:
    """Connection state and ramp of one target over time."""

    def __init__(self, events: Iterable[Any] = ()) -> None:
        events = sorted(events, key=lambda event: event.t)
        self._steps = [(event.t, event.type == "connect") for event in events if event.type in SWITCHING]
        self._connections = [(event.t, event.ramp) for event in events if event.type == "connect"]

    def connected(self, t: float) -> bool:
        return connected_at(self._steps, t)

    def since(self, t: float) -> tuple[float, float]:
        """Return the time and ramp of the connection in force at ``t``."""
        return max(((start, ramp) for start, ramp in self._connections if start <= t + 1e-10),
                   default=(-math.inf, 0.0))

    def ramp_value(self, t: float) -> float:
        """Return 0 while disconnected, otherwise the current connection's smooth ramp in [0, 1]."""
        if not self.connected(t):
            return 0.0
        start, ramp = self.since(t)
        if ramp <= 0.0 or t >= start + ramp:
            return 1.0
        return smoothstep((t - start) / ramp)

    def armed(self, t: float) -> bool:
        """Whether protection is armed: connected and past the connection ramp."""
        start, ramp = self.since(t)
        return self.connected(t) and t >= start + ramp


class UnitScenario(Scenario):
    """A converter unit's connection scenario and ramped active-power reference."""

    def setpoints(self, t: float, p_ref: float, q_ref: float, v_ref: float) -> tuple[float, float, float]:
        return self.ramp_value(t) * p_ref, q_ref, v_ref


class SourceScenario:
    """A source's magnitude, frequency and angle between successive parameter changes."""

    def __init__(self, cfg: Any, f0: float, steps: Iterable[tuple[float, Any]] = ()) -> None:
        self.f0 = f0
        frequency = lambda value: value.f if value.f is not None else f0  # noqa: E731
        self._v_from, self._v = [-math.inf], [cfg.v]
        changes = [(-math.inf, frequency(cfg), cfg.angle)]
        for event_t, value in sorted(steps, key=lambda step: step[0]):
            if value.v != self._v[-1]:
                self._v_from.append(event_t)
                self._v.append(value.v)
            if (frequency(value), value.angle) != changes[-1][1:]:
                changes.append((event_t, frequency(value), value.angle))

        self._angle_from: list[float] = []
        self._angle: list[tuple[float, float, float]] = []
        phase = 0.0
        for index, (event_t, freq, angle) in enumerate(changes):
            if index:
                previous = changes[index - 1]
                phase += 2.0 * math.pi * (previous[1] - f0) * (event_t - max(previous[0], 0.0))
            self._angle_from.append(event_t)
            self._angle.append((freq, angle, phase))

    def law(self, t: float) -> tuple[float, Optional[Callable[[float], float]]]:
        """Return the magnitude and relative-angle law in force at ``t``."""
        voltage = self._v[bisect_right(self._v_from, t) - 1]
        index = bisect_right(self._angle_from, t) - 1
        frequency, angle, phase = self._angle[index]
        slope = 2.0 * math.pi * (frequency - self.f0)
        start = max(self._angle_from[index], 0.0)
        if slope == 0.0:
            return voltage, None if phase + angle == 0.0 else (lambda _t: phase + angle)
        return voltage, lambda at: phase + slope * (at - start) + angle

    def magnitude(self, t: float) -> float:
        """Return the magnitude in force at ``t`` for the current source component."""
        return self.law(t)[0]

    def angle(self, t: float) -> float:
        """Return the relative source angle at ``t``."""
        _voltage, law = self.law(t)
        return 0.0 if law is None else law(t)
