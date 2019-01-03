import logging
from typing import List, Optional, Dict, Any

from jmetal.algorithm import NSGAII
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import S, Evaluator
from jmetal.component.observer import Observer
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from .jmetal_ext.algorithm.multiobjective.hpsea import HPSEA
from .jmetal_ext.algorithm.multiobjective.ssnsgaii import SSNSGAII
from .jmetal_ext.component.evaluator_observer import EvaluatorObserver
from .jmetal_ext.component.multiprocess_evaluator import MultiprocessEvaluator
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem
from .plot_server.plot_server import PlotServer, PlotServerInterface
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
            f'first: {best.objectives}'
        )
        self.last_evaluations = evaluations
        self.last_computing_time = computing_time

    def get_population(self) -> Dict[str, List[float]]:
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

    def get_new_solutions(self, start: int) -> Dict[str, List[float]]:
        new_solutions = self.evaluated_objectives[start:]

        return {
            'f1': [o[0] for o in new_solutions],
            'f2': [o[1] for o in new_solutions]
        }

    def get_path(self, path_id: int) -> List[float]:
        if 0 <= path_id < len(self.evaluated_variables):
            return self.evaluated_variables[path_id]

        raise ValueError(f'Invalid path ID {path_id}')


def get_evaluator(
        evaluator_str: Optional[str], *, kwargs: Dict[str, Any], evaluator_observer: EvaluatorObserver
) -> Optional[Evaluator[S]]:
    logger = logging.getLogger(__name__)

    evaluator_kwargs = {'observer': evaluator_observer}

    def get_dask_evaluator():
        nonlocal evaluator_kwargs
        try:
            from .jmetal_ext.component.dask_evaluator import DaskEvaluator
            if 'scheduler' in kwargs and kwargs.get('scheduler'):
                evaluator_kwargs['scheduler'] = kwargs.get('scheduler')
            return DaskEvaluator
        except ImportError:
            logger.error(f'Please install DaskEvaluator requirements')
            raise

    def get_distributed_evaluator():
        nonlocal evaluator_kwargs
        try:
            from .jmetal_ext.component.distributed_evaluator import DistributedEvaluator
            if 'scheduler' in kwargs:
                if 'scheduler' in kwargs and kwargs.get('scheduler'):
                    evaluator_kwargs['scheduler'] = kwargs.get('scheduler')
            return DistributedEvaluator
        except ImportError:
            logger.error(f'Please install DistributedEvaluator requirements')
            raise

    def get_multiprocess_evaluator():
        return MultiprocessEvaluator

    def get_none():
        return None

    evaluator_dict = {
        'dask': get_dask_evaluator,
        'distributed': get_distributed_evaluator,
        'multiprocess': get_multiprocess_evaluator,
        'none': get_none,
        'None': get_none,
        'default': get_none,
    }

    if evaluator_str:
        if evaluator_str in evaluator_dict:
            evaluator_type = evaluator_dict[evaluator_str]()
            if evaluator_type:
                logger.debug(f'Creating evaluator {evaluator_type.__name__} with kwargs: {evaluator_kwargs}')
                return evaluator_type(**evaluator_kwargs)
        else:
            raise ValueError(f'Invalid evaluator {evaluator_str} (please choose one of these: {evaluator_dict.keys()})')

    return None


def get_algorithm(
        algorithm_str: str, *,
        problem: Problem,
        variables: int,
        population_size: int,
        max_evaluations: int,
        evaluator: Evaluator[S],
        kwargs: Dict[str, Any]
) -> Algorithm:
    logger = logging.getLogger(__name__)

    algorithm_kwargs = {}
    if evaluator:
        algorithm_kwargs['evaluator'] = evaluator

    def get_hpsea():
        nonlocal algorithm_kwargs
        if 'offspring_size' in kwargs and kwargs.get('offspring_size'):
            algorithm_kwargs['offspring_size'] = kwargs.get('offspring_size')
        return HPSEA[FloatSolution, List[FloatSolution]]

    def get_nsgaii():
        return NSGAII[FloatSolution, List[FloatSolution]]

    def get_ssnsgaii():
        return SSNSGAII[FloatSolution, List[FloatSolution]]

    algorithm_dict = {
        'hpsea': get_hpsea,
        'nsgaii': get_nsgaii,
        'ssnsgaii': get_ssnsgaii,
    }

    if algorithm_str in algorithm_dict:
        algorithm_type = algorithm_dict[algorithm_str]()
        logger.debug(f'Creating algorithm {algorithm_type} with kwargs: {algorithm_kwargs}')
        return algorithm_type(
            problem=problem,
            population_size=population_size,
            max_evaluations=max_evaluations,
            mutation=Polynomial(min(3.3 / variables, 1.0), distribution_index=20),
            crossover=SBX(0.9, distribution_index=15),
            selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
            **algorithm_kwargs
        )
    else:
        raise ValueError(f'Invalid algorithm {algorithm_str} (please choose one of these: {algorithm_dict.keys()})')


def get_plot_server(
        plot_server_str: Optional[str], *, plot_server_interface: PlotServerInterface
) -> Optional[PlotServer]:
    logger = logging.getLogger(__name__)

    def get_bokeh_plot_server():
        try:
            from .plot_server.bokeh_plot_server import BokehPlotServer
            return BokehPlotServer
        except ImportError:
            logger.error(f'Please install BokehPlotServer requirements')
            raise

    def get_dash_plot_server():
        try:
            from .plot_server.dash_plot_server import DashPlotServer
            return DashPlotServer
        except ImportError:
            logger.error(f'Please install DashPlotServer requirements')
            raise

    def get_mpl_plot_server():
        try:
            from .plot_server.mpl_plot_server import MplPlotServer
            return MplPlotServer
        except ImportError:
            logger.error(f'Please install MplPlotServer requirements')
            raise

    def get_none():
        return None

    plot_server_dict = {
        'bokeh': get_bokeh_plot_server,
        'dash': get_dash_plot_server,
        'mpl': get_mpl_plot_server,
        'none': get_none,
        'None': get_none,
        'default': get_bokeh_plot_server,
    }

    if plot_server_str:
        if plot_server_str in plot_server_dict:
            plot_server_type = plot_server_dict[plot_server_str]()
            if plot_server_type:
                logger.debug(f'Creating plot server {plot_server_type.__name__}')
                return plot_server_type(plot_server_interface=plot_server_interface)
        else:
            raise ValueError(
                f'Invalid plot server {plot_server_str} (please choose one of these: {plot_server_dict.keys()})')

    return None


class DepositionOptimizer(PlotServerInterface):
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
            evaluator_str: Optional[str] = 'dask',
            algorithm_str: str = 'hpsea',
            plot_server_str: Optional[str] = 'bokeh',
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

        self.logger.debug('Creating evaluator observer')
        self.evaluator_observer = HoardingEvaluatorObserver()

        self.evaluator = get_evaluator(evaluator_str, kwargs=kwargs, evaluator_observer=self.evaluator_observer)

        self.algorithm = get_algorithm(
            algorithm_str, problem=self.problem, variables=variables, population_size=population_size,
            max_evaluations=max_evaluations, evaluator=self.evaluator, kwargs=kwargs
        )

        self.logger.debug('Creating algorithm observer')
        self.algorithm_observer = VerboseHoardingAlgorithmObserver()
        self.algorithm.observable.register(self.algorithm_observer)

        self.plot_server = get_plot_server(plot_server_str, plot_server_interface=self)

    def run(self) -> None:
        if self.plot_server:
            self.plot_server.serve_background()

        self.logger.debug('Running algorithm')
        self.algorithm.run()
        self.logger.debug('Algorithm finished')

        if self.plot_server:
            self.plot_server.stop_background()

        if self.evaluator:
            evaluator_stop = getattr(self.evaluator, 'stop', None)
            if callable(evaluator_stop):
                evaluator_stop()

    def get_new_solutions(self, start: int) -> Dict[str, List[float]]:
        if self.evaluator_observer:
            return self.evaluator_observer.get_new_solutions(start)
        return {}

    def get_population(self) -> Dict[str, List[float]]:
        if self.algorithm_observer:
            return self.algorithm_observer.get_population()
        return {}

    def get_path(self, path_id: int) -> List[float]:
        if self.evaluator_observer:
            return self.evaluator_observer.get_path(path_id)
        return []

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
