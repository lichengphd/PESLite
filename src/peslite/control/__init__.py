"""The controller of a converter unit.

``loops``: what each loop type computes; ``controller``: the controller as one block (its
interface, the loop network that connects and runs the loops, the gfl/gfm wiring,
:class:`UniteType`); ``modulation``: the output stage,
from the voltage command to duty ratios; ``blocks``: transforms, filters and timers they are built of.
Signals are pu; time s, angles rad, frequencies rad/s. The controller depends on nothing but the
solver's kernel: the hardware adapts to its interface (:class:`Measurement`, :class:`ControlOutput`).
"""

from ..solver import ConfigError
from .loops import (ANGLE, FREQUENCY, I, I_AB, I_DQ, LOOP_TYPES, POWER, PQ, V, V_AB, V_DQ,
                    Loop, SignalType, SyncLaw, register_loop_type)
from .modulation import ModulationLimiter, OutputStage
from .controller import (CONTROL_INTERFACE, ControlGraph, ControlInterface, ControlMeasurement,
                         ControlOutput, Controller, Measurement, Startup, UniteType,
                         default_wiring, make_controller)

__all__ = ["Loop", "SyncLaw", "SignalType", "ConfigError",
           "I_AB", "V_AB", "I_DQ", "V_DQ", "I", "V",
           "ANGLE", "FREQUENCY", "POWER", "PQ",
           "LOOP_TYPES", "register_loop_type", "ControlInterface", "CONTROL_INTERFACE",
           "ControlGraph", "default_wiring",
           "ModulationLimiter", "OutputStage", "Measurement", "ControlMeasurement", "ControlOutput",
           "Controller", "Startup", "UniteType", "make_controller"]
