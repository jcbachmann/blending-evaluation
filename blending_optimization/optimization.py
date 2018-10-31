import logging
from typing import List, Tuple

import pandas as pd
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import S
from jmetal.component.observer import Observer
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from blending_optimization.dash_plot_server import PlotServer
from blending_optimization.dask_evaluator import DaskEvaluator
from blending_optimization.evaluator_observer import EvaluatorObserver
from blending_optimization.homogenization_problem import HomogenizationProblem
from blending_optimization.hpsea import HPSEA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationResult:
    def __init__(self, result_population, all_variables, all_objectives):
        self.result_population = result_population
        self.all_variables = all_variables
        self.all_objectives = all_objectives


class MyAlgorithmObserver(Observer):
    def __init__(self):
        self.population = []
        self.last_evaluations = None
        self.last_computing_time = None

    def update(self, *args, **kwargs):
        evaluations = kwargs['evaluations']
        self.population = kwargs['population']
        computing_time = kwargs['computing time']
        e_diff = evaluations - self.last_evaluations if self.last_evaluations else evaluations
        t_diff = computing_time - self.last_computing_time if self.last_computing_time else computing_time
        cps = e_diff / t_diff if t_diff > 0 else '-'
        logger.info(
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


class MyEvaluatorObserver(EvaluatorObserver):
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


def optimize(length: float, depth: float, variables: int, material: pd.DataFrame, parameter_columns: List[str],
             population_size: int, max_evaluations: int) -> Tuple[OptimizationResult, HomogenizationProblem]:
    problem = HomogenizationProblem(
        length=length,
        depth=depth,
        number_of_variables=variables,
        material=material,
        parameter_columns=parameter_columns
    )

    evaluator_observer = MyEvaluatorObserver()

    algorithm = HPSEA[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        mutation=Polynomial(3.3 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(0.9, distribution_index=15),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
        evaluator=DaskEvaluator(
            observer=evaluator_observer,
            # scheduler_address='127.0.0.1:8786'
        ),
        offspring_size=20
    )

    algorithm_observer = MyAlgorithmObserver()
    algorithm.observable.register(algorithm_observer)

    plot_server = PlotServer(
        all_callback=evaluator_observer.get_new_solutions,
        pop_callback=algorithm_observer.get_population,
        path_callback=evaluator_observer.get_path
    )
    plot_server.serve_background()

    algorithm.run()

    return OptimizationResult(
        result_population=algorithm.get_result(),
        all_variables=evaluator_observer.evaluated_variables,
        all_objectives=evaluator_observer.evaluated_objectives
    ), problem
