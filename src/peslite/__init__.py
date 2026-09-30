"""Power-electronics converter simulation built on the :mod:`peslite.solver` model and solver kernel.

Plant quantities are in SI; controllers work in each unit's pu bases.
"""

from .solver import AdaptiveSolver, DormandPrince45, FixedStepSolver, Model, MultirateSolver, make_solver

from . import (solver, assembly, control, firmware, modulation, params, power, protection, results,
               sensing)
from .control import OutputStage, UniteType, make_controller
from .assembly import System, Unit, SourceScenario, UnitScenario
from .firmware import ComputationDelay
from .modulation import ZOH, CarrierComparison, SampledCarrier, StepAveragedCarrier, make_modulator
from .params import Params, load
from .simulation import Simulation, main
from .results import SimulationResult
from .sensing import MeasurementPorts

__version__ = "0.1.1"

__all__ = [
    "assembly", "control", "firmware", "modulation", "params", "power", "protection",
    "results", "sensing", "simulation", "solver",
    "Params", "load", "System", "Unit", "MeasurementPorts", "Model", "SourceScenario", "UnitScenario",
    "Simulation", "SimulationResult", "UniteType", "main",
    "make_controller", "make_modulator", "make_solver", "OutputStage",
    "ComputationDelay", "CarrierComparison", "ZOH", "StepAveragedCarrier", "SampledCarrier",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver",
    "__version__",
]
