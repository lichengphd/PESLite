"""Power-electronics converter simulation built on the :mod:`peslite.solver` model and solver kernel.

Plant quantities are in SI; controllers work in each unit's pu bases.
"""

from .solver import AdaptiveSolver, DormandPrince45, FixedStepSolver, Model, MultirateSolver, make_solver

from . import solver, control, components, assembly, params, results
from .control import OutputStage, UniteType, make_controller
from .components import (ZOH, CarrierComparison, ComputationDelay, MeasurementPorts,
                         StepAveragedCarrier, make_modulator)
from .assembly import System, Unit, SourceScenario, UnitScenario
from .params import Params, load
from .simulation import Simulation, main
from .results import SimulationResult

__version__ = "0.1.1"

__all__ = [
    "solver", "control", "components", "assembly", "params", "results", "simulation",
    "Params", "load", "System", "Unit", "MeasurementPorts", "Model", "SourceScenario", "UnitScenario",
    "Simulation", "SimulationResult", "UniteType", "main",
    "make_controller", "make_modulator", "make_solver", "OutputStage",
    "ComputationDelay", "CarrierComparison", "ZOH", "StepAveragedCarrier",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver",
    "__version__",
]
