"""Power-electronics converter simulation on a port-Hamiltonian model kernel.

The package has four parts: :mod:`.components` (power circuit and converter hardware),
:mod:`.control` (the converter's controller), :mod:`.assembly` (a system built from a simulation
file) and :mod:`.solver` (the model kernel, the integrators and the run). Plant quantities are in
SI; controllers work in each unit's pu bases.
"""

from . import solver, control, components, assembly
from .solver import (AdaptiveSolver, DormandPrince45, FixedStepSolver, Model, MultirateSolver, make_solver)
from .control import OutputStage, UniteType, make_controller, register_loop_type
from .components import (ZOH, CarrierComparison, ComputationDelay, MeasurementPorts, TimeStepAveragedCarrier,
                         make_modulator, register_element_type)
from .assembly import (Params, SourceScenario, System, Unit, UnitScenario, dump, dumps, load,
                       register_event_type)
from .solver.simulation import Simulation, SimulationResult, main

__version__ = "0.1.2"

__all__ = [
    "solver", "control", "components", "assembly",
    "Params", "load", "dump", "dumps", "System", "Unit", "MeasurementPorts", "Model", "SourceScenario", "UnitScenario",
    "Simulation", "SimulationResult", "UniteType", "main", "make_controller", "make_modulator", "make_solver",
    "register_loop_type", "register_element_type", "register_event_type",
    "OutputStage", "ComputationDelay", "CarrierComparison", "ZOH", "TimeStepAveragedCarrier",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "__version__",
]
