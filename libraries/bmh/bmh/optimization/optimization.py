import logging
import math
from typing import List, Optional, Dict, Any

from bmh.optimization.jmetal_ext.problem.multiobjective.solution_generator import RandomSolutionGenerator
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
from .jmetal_ext.component.multiprocess_evaluator import MultiprocessEvaluator
from .jmetal_ext.component.observable_evaluator import EvaluatorObserver
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem, process_material_deposition
from .optimization_result import OptimizationResult
from .plot_server.plot_server import PlotServer, PlotServerInterface
from ..benchmark.material_deposition import Material, Deposition, DepositionMeta


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
            f'best: {best.objectives}'
        )
        self.last_evaluations = evaluations
        self.last_computing_time = computing_time

    def get_population(self) -> Dict[str, List[float]]:
        return {
            'f1': [o.objectives[0] for o in self.population],
            'f2': [o.objectives[1] for o in self.population]
        }

    def reset(self):
        self.population = []
        self.last_evaluations = None
        self.last_computing_time = None


class HoardingEvaluatorObserver(EvaluatorObserver):
    def __init__(self):
        self.solutions: List[S] = []

    def notify(self, solution_list: List[S]):
        for solution in solution_list:
            self.solutions.append(solution)

    def get_new_solutions(self, start: int) -> Dict[str, List[float]]:
        new_solutions = self.solutions[start:]

        return {
            'f1': [o.objectives[0] for o in new_solutions],
            'f2': [o.objectives[1] for o in new_solutions]
        }

    def get_solution(self, solution_id: int) -> S:
        if 0 <= solution_id < len(self.solutions):
            return self.solutions[solution_id]

        raise ValueError(f'Invalid solution ID {solution_id}')

    def reset(self):
        self.solutions = []


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
        plot_server_str: Optional[str], *, plot_server_interface: PlotServerInterface, port: int
) -> Optional[PlotServer]:
    logger = logging.getLogger(__name__)

    def get_bokeh_plot_server():
        try:
            from .plot_server.bokeh_plot_server import BokehPlotServer
            return BokehPlotServer
        except ImportError:
            logger.error(f'Please install BokehPlotServer requirements')
            raise

    def get_none():
        return None

    plot_server_dict = {
        'bokeh': get_bokeh_plot_server,
        'none': get_none,
        'None': get_none,
        'default': get_none,
    }

    if plot_server_str:
        if plot_server_str in plot_server_dict:
            plot_server_type = plot_server_dict[plot_server_str]()
            if plot_server_type:
                logger.debug(f'Creating plot server {plot_server_type.__name__}')
                return plot_server_type(plot_server_interface=plot_server_interface, port=port)
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
            population_size: int = 250,
            max_evaluations: int = 25000,
            evaluator_str: Optional[str] = 'dask',
            algorithm_str: str = 'hpsea',
            plot_server_str: Optional[str] = 'none',
            plot_server_port: int = PlotServer.DEFAULT_PORT,
            auto_start: bool = True,
            v_max: float,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Initializing optimization problem')

        # Copy arguments
        self.deposition_meta = deposition_meta
        self.x_min = x_min
        self.x_max = x_max
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.evaluator_str = evaluator_str
        self.algorithm_str = algorithm_str
        self.plot_server_str = plot_server_str
        self.auto_start = auto_start
        self.v_max = v_max
        self.kwargs = kwargs

        # Cache
        self.problem: Optional[HomogenizationProblem] = None
        self.algorithm: Optional[Algorithm] = None

        self.algorithm_observer = VerboseHoardingAlgorithmObserver()
        self.evaluator_observer = HoardingEvaluatorObserver()
        self.evaluator = get_evaluator(
            self.evaluator_str, kwargs=self.kwargs, evaluator_observer=self.evaluator_observer
        )
        self.plot_server = get_plot_server(self.plot_server_str, plot_server_interface=self, port=plot_server_port)
        self.deposition_prefix: Deposition = None

    def start(self):
        if self.plot_server:
            self.plot_server.serve_background()

    def run(
            self, *, material: Material, variables: int, deposition_prefix: Deposition = None,
            timestamps: Optional[List[float]] = None,
            solution_generator=RandomSolutionGenerator()
    ) -> None:
        self.algorithm_observer.reset()
        self.evaluator_observer.reset()
        if self.plot_server:
            self.plot_server.reset()

        self.deposition_prefix = deposition_prefix

        self.problem = HomogenizationProblem(
            deposition_meta=self.deposition_meta,
            x_min=self.x_min,
            x_max=self.x_max,
            material=material,
            number_of_variables=variables,
            deposition_prefix=deposition_prefix,
            v_max=self.v_max,
            timestamps=timestamps,
            solution_generator=solution_generator,
        )

        self.algorithm = get_algorithm(
            self.algorithm_str, problem=self.problem, variables=variables, population_size=self.population_size,
            max_evaluations=self.max_evaluations, evaluator=self.evaluator, kwargs=self.kwargs
        )
        self.algorithm.observable.register(self.algorithm_observer)

        if self.auto_start:
            self.logger.debug('Starting DepositionOptimizer')
            self.start()

        self.logger.debug('Running algorithm')
        self.algorithm.run()
        self.logger.debug('Algorithm finished')

        if self.auto_start:
            self.logger.debug('Stopping DepositionOptimizer')
            self.stop()

    def stop(self):
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

    def get_solution(self, solution_id: int) -> OptimizationResult:
        if self.evaluator_observer and self.problem:
            solution = self.evaluator_observer.get_solution(solution_id)
            deposition = self.problem.variables_to_deposition(solution.variables)
            return OptimizationResult(
                deposition=deposition,
                variables=solution.variables,
                objectives=solution.objectives,
                objective_labels=self.problem.get_objective_labels(),
                reclaimed_material=process_material_deposition(material=self.problem.material, deposition=deposition)
            )

        raise RuntimeError('DepositionOptimizer not initialized')

    def get_best_solution(self) -> Optional[OptimizationResult]:
        if self.evaluator_observer and self.problem:
            if len(self.algorithm_observer.population) > 0:
                solution = min(
                    self.algorithm_observer.population, key=lambda r: math.hypot(r.objectives[0], r.objectives[1])
                )
                deposition = self.problem.variables_to_deposition(solution.variables)
                return OptimizationResult(
                    deposition=deposition,
                    variables=solution.variables,
                    objectives=solution.objectives,
                    objective_labels=self.problem.get_objective_labels(),
                    reclaimed_material=process_material_deposition(material=self.problem.material,
                                                                   deposition=deposition)
                )
            else:
                return None

        raise RuntimeError('DepositionOptimizer not initialized')

    def get_reference(self) -> Optional[OptimizationResult]:
        deposition, material, objectives = self.problem.get_reference_relative()
        return OptimizationResult(
            deposition=deposition,
            variables=[],
            objectives=objectives,
            objective_labels=self.problem.get_objective_labels(),
            reclaimed_material=material
        )

    def get_all_results(self) -> List[OptimizationResult]:
        self.logger.debug('Collecting all results')
        return self.solutions_to_optimization_results(self.evaluator_observer.solutions)

    def get_final_results(self) -> List[OptimizationResult]:
        self.logger.debug('Collecting final results')
        return self.solutions_to_optimization_results(self.algorithm.get_result())

    def solutions_to_optimization_results(self, solutions: List[S]) -> List[OptimizationResult]:
        objective_labels = self.problem.get_objective_labels()
        return [OptimizationResult(
            self.problem.variables_to_deposition(s.variables), s.variables, s.objectives, objective_labels
        ) for s in solutions]

    def get_material(self) -> Material:
        return self.problem.material

    def get_progress(self) -> Dict[str, float]:
        if self.deposition_prefix:
            return {
                't_start': self.deposition_prefix.meta.time
            }
