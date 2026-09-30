"""Construction of registered control-loop parameter dataclasses.

Each loop type owns its ``Params`` dataclass and is registered once in
``peslite.control.LOOP_TYPES``. Keeping construction here lets the existing
configuration package consume that single registry while the remaining
assembly modules are migrated in later refactor steps.
"""

from __future__ import annotations

from dataclasses import is_dataclass

from ..control.loops import LOOP_TYPES
from ..solver.model import ConfigError
from .schema import build, to_dict

__all__ = ["loop_params", "build_loop"]


def loop_params(kind: object, where: str) -> type:
    """Return the parameter dataclass of registered loop ``kind``.

    The lookup is deliberately dynamic: a custom loop registered before
    loading a file is immediately available to the configuration reader.
    """
    cls = LOOP_TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ConfigError(
            f"{where}.type: unknown loop type {kind!r}; known: {sorted(LOOP_TYPES)}"
        )
    return cls.Params


def build_loop(data, where: str):
    """Build and validate one registered loop's parameter object."""
    if is_dataclass(data):
        data = to_dict(data)
    kind = data.get("type") if isinstance(data, dict) else None
    params = loop_params(kind, where)
    try:
        return build(params, data)
    except ConfigError as exc:
        message = str(exc)
        prefix = f"{where}."
        raise ConfigError(message if message.startswith(prefix) else prefix + message) from exc
