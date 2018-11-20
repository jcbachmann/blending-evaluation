import logging
from typing import List, Optional

from jmetal.algorithm import NSGAII
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import S
from jmetal.component.observer import Observer
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from .jmetal_ext.algorithm.multiobjective.hpsea import HPSEA
from .jmetal_ext.algorithm.multiobjective.ssnsgaii import SSNSGAII
from .jmetal_ext.component.dask_evaluator import DaskEvaluator
from .jmetal_ext.component.evaluator_observer import EvaluatorObserver
from .jmetal_ext.component.multiprocess_evaluator import MultiprocessEvaluator
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem
from .plot_server.bokeh_plot_server import BokehPlotServer
from .plot_server.dash_plot_server import DashPlotServer
from .plot_server.mpl_plot_server import MplPlotServer
from ..benchmark.material_deposition import Material


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


def optimize_deposition(
        bed_size_x: float,
        bed_size_z: float,
        material: Material,
        variables: int = 31,
        population_size: int = 250,
        max_evaluations: int = 25000,
        evaluator_str: Optional[str] = 'dask',
        algorithm_str: Optional[str] = 'hpsea',
        plot_server_str: Optional[str] = 'bokeh',
        **kwargs
) -> OptimizationResult:
    logger = logging.getLogger(__name__)
    logger.debug('Initializing optimization problem')

    problem = HomogenizationProblem(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        material=material,
        number_of_variables=variables,
    )

    algorithm_dict = {
        'hpsea': (HPSEA, {'offspring_size': kwargs.get('offspring_size', None)}),
        'nsgaii': (NSGAII, {}),
        'ssnsgaii': (SSNSGAII, {}),
    }
    algorithm_type, algorithm_kwargs = algorithm_dict.get(algorithm_str, (HPSEA, {}))

    logger.debug('Creating evaluator observer')
    evaluator_observer = HoardingEvaluatorObserver()

    evaluator_dict = {
        'dask': (DaskEvaluator, {'scheduler_address': kwargs.get('scheduler_address', None)}),
        'multiprocess': (MultiprocessEvaluator, {}),
    }
    evaluator_type, evaluator_kwargs = evaluator_dict.get(evaluator_str, (None, {}))
    if evaluator_type:
        logger.debug(f'Creating evaluator {evaluator_type.__name__} with kwargs: {str(evaluator_kwargs)}')
        algorithm_kwargs['evaluator'] = evaluator_type(observer=evaluator_observer, **evaluator_kwargs)

    logger.debug(f'Creating algorithm {algorithm_type.__name__} with kwargs: {str(algorithm_kwargs)}')
    algorithm = algorithm_type[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        mutation=Polynomial(3.3 / variables, distribution_index=20),
        crossover=SBX(0.9, distribution_index=15),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
        **algorithm_kwargs
    )

    logger.debug('Creating algorithm observer')
    algorithm_observer = VerboseHoardingAlgorithmObserver()
    algorithm.observable.register(algorithm_observer)

    plot_server_dict = {
        'bokeh': BokehPlotServer,
        'dash': DashPlotServer,
        'mpl': MplPlotServer,
    }
    plot_server_type = plot_server_dict.get(plot_server_str, None)
    logger.debug(f'Creating plot_server {plot_server_type.__name__}')
    plot_server = plot_server_type(
        all_callback=evaluator_observer.get_new_solutions,
        pop_callback=algorithm_observer.get_population,
        path_callback=evaluator_observer.get_path
    ) if plot_server_type else None
    if plot_server:
        plot_server.serve_background()

    logger.debug('Running algorithm')
    algorithm.run()
    logger.debug('Algorithm finished')

    return OptimizationResult(
        result_population=algorithm.get_result(),
        all_variables=evaluator_observer.evaluated_variables,
        all_objectives=evaluator_observer.evaluated_objectives,
        variable_labels=problem.get_variable_labels(),
        objective_labels=problem.get_objective_labels()
    )
