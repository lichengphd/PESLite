"""The controller of a converter unit.

``loops``: what each loop type computes; ``controller``: the controller as one block (its
interface, the loop network that connects and runs the loops, the gfl/gfm wiring,
:class:`UniteType`); ``protection``: the trip and alarm criteria; ``modulation``: the output stage,
from the voltage command to duty ratios; ``blocks``: transforms, filters and timers they are built of.
Signals are pu; time s, angles rad, frequencies rad/s. The controller depends on nothing but the
solver's kernel: the hardware adapts to its interface (:class:`Measurement`, :class:`ControlOutput`).
"""

from .loops import LOOP_TYPES, Loop, SignalType, SyncLaw, register_loop_type
from .protection import Protection
from .modulation import ModulationLimiter, OutputStage
from .controller import (ControlGraph, ControlMeasurement, ControlOutput, Controller, Measurement, UniteType,
                         default_wiring, make_controller)

__all__ = ["Loop", "SyncLaw", "SignalType", "LOOP_TYPES", "register_loop_type", "ControlGraph", "default_wiring",
           "Protection", "ModulationLimiter", "OutputStage", "Measurement", "ControlMeasurement", "ControlOutput",
           "Controller", "UniteType", "make_controller"]
