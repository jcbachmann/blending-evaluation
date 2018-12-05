import logging
from typing import List, Optional, Dict, Any, Tuple, Type

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
from .jmetal_ext.component.distributed_evaluator import DistributedEvaluator
from .jmetal_ext.component.evaluator_observer import EvaluatorObserver
from .jmetal_ext.component.multiprocess_evaluator import MultiprocessEvaluator
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem
from .plot_server.bokeh_plot_server import BokehPlotServer
from .plot_server.dash_plot_server import DashPlotServer
from .plot_server.mpl_plot_server import MplPlotServer
from ..benchmark.material_deposition import Material, Deposition, DepositionMeta


class OptimizationResult:
    def __init__(self, deposition: Deposition, variables: List[float], objectives: List[float],
                 objective_labels: List[str]):
        self.deposition = deposition
        self.variables = variables
        self.objectives = objectives
        self.objective_labels = objective_labels


class VerboseHoardingAlgorithmObserver(Observer):
    def __init__(self):
        self.population = []
        self.last_evaluations: Optional[int] = None
        self.last_computing_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)

    def update(self, *args, **kwargs):
        evaluations = kwargs['evaluations']
        self.population = kwargs['population']
        computing_time = kwargs['computing time']
        e_diff = evaluations - self.last_evaluations if self.last_evaluations else evaluations
        t_diff = computing_time - self.last_computing_time if self.last_computing_time else computing_time
        cps = e_diff / t_diff if t_diff > 0 else '-'
        best = min(self.population, key=lambda s: s.objectives[0] * s.objectives[0] + s.objectives[1] * s.objectives[1])
        self.logger.info(
            f'{evaluations} evaluations / {computing_time:.1f}s @{cps:.2f}cps, '
            f'first: {str(best.objectives)}'
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


class DepositionOptimizer:
    def __init__(
            self,
            *,
            deposition_meta: DepositionMeta,
            x_min: float,
            x_max: float,
            material: Material,
            variables: int = 31,
            population_size: int = 250,
            max_evaluations: int = 25000,
            evaluator_str: str = 'dask',
            algorithm_str: str = 'hpsea',
            plot_server_str: str = 'bokeh',
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Initializing optimization problem')

        self.problem: HomogenizationProblem = HomogenizationProblem(
            deposition_meta=deposition_meta,
            x_min=x_min,
            x_max=x_max,
            material=material,
            number_of_variables=variables,
        )

        algorithm_dict: Dict[str, Tuple[Type, Dict[str, Any]]] = {
            'hpsea': (HPSEA, {'offspring_size': kwargs.get('offspring_size', None)}),
            'nsgaii': (NSGAII, {}),
            'ssnsgaii': (SSNSGAII, {}),
        }
        algorithm_type, algorithm_kwargs = algorithm_dict.get(algorithm_str, (HPSEA, {}))

        self.logger.debug('Creating evaluator observer')
        self.evaluator_observer = HoardingEvaluatorObserver()

        evaluator_dict = {
            'dask': (DaskEvaluator, {'scheduler': kwargs.get('scheduler', None)}),
            'distributed': (DistributedEvaluator, {'scheduler': kwargs.get('scheduler', None)}),
            'multiprocess': (MultiprocessEvaluator, {}),
        }
        evaluator_type, evaluator_kwargs = evaluator_dict.get(evaluator_str, (None, {}))
        if evaluator_type:
            self.logger.debug(f'Creating evaluator {evaluator_type.__name__} with kwargs: {str(evaluator_kwargs)}')
            algorithm_kwargs['evaluator'] = evaluator_type(observer=self.evaluator_observer, **evaluator_kwargs)

            self.logger.debug(f'Creating algorithm {algorithm_type.__name__} with kwargs: {str(algorithm_kwargs)}')
        self.algorithm = algorithm_type[FloatSolution, List[FloatSolution]](
            problem=self.problem,
            population_size=population_size,
            max_evaluations=max_evaluations,
            mutation=Polynomial(min(3.3 / variables, 1.0), distribution_index=20),
            crossover=SBX(0.9, distribution_index=15),
            selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
            **algorithm_kwargs
        )

        self.logger.debug('Creating algorithm observer')
        algorithm_observer = VerboseHoardingAlgorithmObserver()
        self.algorithm.observable.register(algorithm_observer)

        self.plot_server = None
        plot_server_dict = {
            'bokeh': BokehPlotServer,
            'dash': DashPlotServer,
            'mpl': MplPlotServer,
        }
        plot_server_type = plot_server_dict.get(plot_server_str, None)
        if plot_server_type is not None:
            self.logger.debug(f'Creating plot_server {plot_server_type.__name__}')
            self.plot_server = plot_server_type(
                all_callback=self.evaluator_observer.get_new_solutions,
                pop_callback=algorithm_observer.get_population,
                path_callback=self.evaluator_observer.get_path
            )

    def run(self) -> None:
        if self.plot_server:
            self.plot_server.serve_background()

        self.logger.debug('Running algorithm')
        self.algorithm.run()
        self.logger.debug('Algorithm finished')

        if self.plot_server:
            self.plot_server.stop_background()

    def get_all_results(self) -> List[OptimizationResult]:
        self.logger.debug('Collecting all results')
        objective_labels = self.problem.get_objective_labels()
        return [OptimizationResult(self.problem.variables_to_deposition(v), v, o, objective_labels) for v, o in
                zip(self.evaluator_observer.evaluated_variables, self.evaluator_observer.evaluated_objectives)]

    def get_final_results(self) -> List[OptimizationResult]:
        self.logger.debug('Collecting final results')
        result_population = self.algorithm.get_result()
        objective_labels = self.problem.get_objective_labels()
        return [OptimizationResult(self.problem.variables_to_deposition(s.variables), s.variables, s.objectives,
                                   objective_labels) for s in result_population]
