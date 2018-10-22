import logging
from typing import List, Tuple

import pandas as pd
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import S
from jmetal.component.observer import Observer, VisualizerObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from blending_optimization.homogenization_problem import HomogenizationProblem
from blending_optimization.hpsea import HPSEA
from blending_optimization.multiprocess_evaluator import MultiprocessEvaluator, EvaluatorObserver
from blending_optimization.plot_server import PlotServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationResult:
    def __init__(self, result_population, all_variables, all_objectives):
        self.result_population = result_population
        self.all_variables = all_variables
        self.all_objectives = all_objectives


class MyAlgorithmObserver(Observer):
    def update(self, *args, **kwargs):
        evaluations = kwargs['evaluations']
        population = kwargs['population']
        computing_time = kwargs['computing time']
        cps = evaluations / computing_time if computing_time > 0 else '-'
        logger.info(
            f'{evaluations} evaluations / {computing_time:.1f}s @{cps:.2f}cps, first: {str(population[0].objectives)}')


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


def optimize(length: float, depth: float, variables: int, material: pd.DataFrame, population_size: int,
             max_evaluations: int) -> Tuple[OptimizationResult, HomogenizationProblem]:
    problem = HomogenizationProblem(
        length=length,
        depth=depth,
        number_of_variables=variables,
        material=material
    )

    evaluator_observer = MyEvaluatorObserver()

    algorithm = HPSEA[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        mutation=Polynomial(1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
        evaluator=MultiprocessEvaluator(observer=evaluator_observer),
        offspring_size=20
    )

    algorithm.observable.register(MyAlgorithmObserver())
    algorithm.observable.register(VisualizerObserver())

    plot_server = PlotServer(evaluator_observer.get_new_solutions)
    plot_server.serve_background()

    algorithm.run()

    return OptimizationResult(
        result_population=algorithm.get_result(),
        all_variables=evaluator_observer.evaluated_variables,
        all_objectives=evaluator_observer.evaluated_objectives
    ), problem
