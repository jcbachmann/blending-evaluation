import logging
from typing import List, Optional

from bmh.benchmark.material_deposition import Material
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import S
from jmetal.component.observer import Observer
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from .jmetal_ext.algorithm.multiobjective.hpsea import HPSEA
from .jmetal_ext.component.dask_evaluator import DaskEvaluator
from .jmetal_ext.component.evaluator_observer import EvaluatorObserver
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem
from .plot_server.bokeh_plot_server import BokehPlotServer


class OptimizationResult:
    def __init__(self, result_population, all_variables, all_objectives, variable_labels: List[str],
                 objective_labels: List[str]):
        self.result_population = result_population
        self.all_variables = all_variables
        self.all_objectives = all_objectives
        self.variable_labels = variable_labels
        self.objective_labels = objective_labels


class VerboseHoardingAlgorithmObserver(Observer):
    def __init__(self):
        self.population = []
        self.last_evaluations = None
        self.last_computing_time = None
        self.logger = logging.getLogger(__name__)

    def update(self, *args, **kwargs):
        evaluations = kwargs['evaluations']
        self.population = kwargs['population']
        computing_time = kwargs['computing time']
        e_diff = evaluations - self.last_evaluations if self.last_evaluations else evaluations
        t_diff = computing_time - self.last_computing_time if self.last_computing_time else computing_time
        cps = e_diff / t_diff if t_diff > 0 else '-'
        self.logger.info(
            f'{evaluations} evaluations / {computing_time:.1f}s @{cps:.2f}cps, '
            f'first: {str(self.population[0].objectives)}'
        )
        self.last_evaluations = evaluations
        self.last_computing_time = computing_time

    def get_population(self):
        return {
            'f1': [o.objectives[0] for o in self.population],
            'f2': [o.objectives[1] for o in self.population]
        }


class HoardingEvaluatorObserver(EvaluatorObserver):
    def __init__(self):
        self.evaluated_variables = []
        self.evaluated_objectives = []

    def notify(self, solution_list: List[S]):
        for solution in solution_list:
            self.evaluated_variables.append(solution.variables)
            self.evaluated_objectives.append(solution.objectives)

    def get_new_solutions(self, start: int):
        new_solutions = self.evaluated_objectives[start:]

        return {
            'f1': [o[0] for o in new_solutions],
            'f2': [o[1] for o in new_solutions]
        }

    def get_path(self, path_id: int):
        if 0 <= path_id < len(self.evaluated_variables):
            return self.evaluated_variables[path_id]

        return None


def optimize(bed_size_x: float, bed_size_z: float, variables: int, material: Material, population_size: int,
             max_evaluations: int, scheduler_address: Optional[str] = None) -> OptimizationResult:
    problem = HomogenizationProblem(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        material=material,
        number_of_variables=variables,
    )

    evaluator_observer = HoardingEvaluatorObserver()

    algorithm = HPSEA[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        mutation=Polynomial(3.3 / variables, distribution_index=20),
        crossover=SBX(0.9, distribution_index=15),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
        evaluator=DaskEvaluator(
            observer=evaluator_observer,
            scheduler_address=scheduler_address
        ),
        offspring_size=2 * int(0.5 * 0.15 * population_size)
    )

    algorithm_observer = VerboseHoardingAlgorithmObserver()
    algorithm.observable.register(algorithm_observer)

    plot_server = BokehPlotServer(
        all_callback=evaluator_observer.get_new_solutions,
        pop_callback=algorithm_observer.get_population,
        path_callback=evaluator_observer.get_path
    )
    plot_server.serve_background()

    algorithm.run()

    return OptimizationResult(
        result_population=algorithm.get_result(),
        all_variables=evaluator_observer.evaluated_variables,
        all_objectives=evaluator_observer.evaluated_objectives,
        variable_labels=problem.get_variable_labels(),
        objective_labels=problem.get_objective_labels()
    )
