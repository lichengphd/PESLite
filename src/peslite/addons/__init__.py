"""Optional functions and automatically discovered custom PESLite types."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import components, controllers

# These imports happen once when PESLite starts.  They populate the existing registries; no
# directory search or add-on dispatch remains in the simulation hot path.
controllers.discover()
components.discover()

__all__ = ["controllers", "components", "functions", "plot_csv", "plot_result"]


def __getattr__(name: str) -> Any:
    """Load optional function implementations only when their API is requested."""
    if name == "functions":
        module = import_module(f"{__name__}.functions")
        globals()[name] = module
        return module
    if name in {"plot_csv", "plot_result"}:
        module = import_module(f"{__name__}.functions")
        globals().update({public: getattr(module, public) for public in ("plot_csv", "plot_result")})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
