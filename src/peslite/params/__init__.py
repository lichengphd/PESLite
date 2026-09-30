"""Configuration schemas, SI/pu input conversion and file I/O.

Electrical inputs without suffix are SI values; names ending in _pu are per
unit. Control-loop parameter dataclasses belong to their registered loop
classes in peslite.control.
"""

from ..solver.integrators import ADAPTIVE_METHODS, FIXED_METHODS
from ..solver.model import ConfigError
from .base import BaseValues, DCBase
from .io import dump, from_dict, load, read_initial
from .schema import (
    ACFilterParams,
    BranchParams,
    BusParams,
    ControlParams,
    DCCapacitorParams,
    DCLinkParams,
    DCSourceParams,
    DelayParams,
    LogParams,
    MeasurementParams,
    OutputParams,
    PLLGainStepParams,
    PWMParams,
    Params,
    ProtectionParams,
    ReferenceParams,
    SetpointStepParams,
    SimulationParams,
    SolverParams,
    SourceEventParams,
    SourceParams,
    StartupParams,
    UnitEventParams,
    UnitParams,
    to_dict,
)

__all__ = [
    "Params", "BaseValues", "DCBase", "load", "dump", "read_initial",
    "from_dict", "to_dict", "ConfigError", "FIXED_METHODS", "ADAPTIVE_METHODS",
    "BusParams", "BranchParams", "SourceParams", "ACFilterParams",
    "DCLinkParams", "DCCapacitorParams", "DCSourceParams", "PWMParams",
    "MeasurementParams", "ControlParams", "ReferenceParams", "DelayParams",
    "ProtectionParams", "StartupParams", "SourceEventParams",
    "SetpointStepParams", "PLLGainStepParams", "UnitEventParams", "UnitParams",
    "SolverParams", "LogParams", "SimulationParams", "OutputParams",
]
