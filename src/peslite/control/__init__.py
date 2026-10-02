"""The controller of a converter unit.

``loops``: what each loop type computes; ``controller``: the controller as one block (its
interface, the loop network that connects and runs the loops, the gfl/gfm wiring,
:class:`UniteType`); ``modulation``: the output stage,
from the voltage command to duty ratios; ``blocks``: transforms, filters and timers they are built of.
Signals are pu; time s, angles rad, frequencies rad/s. The controller depends on nothing but the
solver's kernel: the hardware adapts to its interface (:class:`Measurement`, :class:`ControlOutput`).
"""

from ..solver import ConfigError
from .loops import (ANGLE, CURRENT, DC_VOLTAGE, FREQUENCY, I_AB, LOOP_TYPES, POWER_PU, V_AB,
                    V_DQ, VOLTAGE, Loop, SignalType, SyncLaw, register_loop_type)
from .modulation import ModulationLimiter, OutputStage
from .controller import (ControlGraph, ControlMeasurement, ControlOutput, Controller, Measurement,
                         Startup, UniteType, default_wiring, make_controller)

__all__ = ["Loop", "SyncLaw", "SignalType", "ConfigError", "V_AB", "I_AB", "V_DQ", "VOLTAGE",
           "DC_VOLTAGE", "CURRENT", "ANGLE", "FREQUENCY", "POWER_PU",
           "LOOP_TYPES", "register_loop_type", "ControlGraph", "default_wiring",
           "ModulationLimiter", "OutputStage", "Measurement", "ControlMeasurement", "ControlOutput",
           "Controller", "Startup", "UniteType", "make_controller"]
