import logging
from typing import List, Optional, Dict, Any

import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.algorithm import Algorithm
from jmetal.core.observer import Observer
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import GenerationalDistance, InvertedGenerationalDistance, HyperVolume
from jmetal.core.solution import FloatSolution
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.evaluator import Evaluator, S
from jmetal.util.generator import Generator
from jmetal.util.observer import WriteFrontToFileObserver
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file, \
    read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetalpy_extensions.util.evaluator import EvaluatorObserver, MultiprocessEvaluator
from jmetalpy_extensions.util.observer import WriteQualityIndicatorsToFileObserver
from .homogenization_problem.homogenization_problem import HomogenizationProblem, process_material_deposition
from .optimization_result import OptimizationResult
from .plot_server.plot_server import PlotServer, PlotServerInterface
from ..benchmark.material_deposition import Material, Deposition, DepositionMeta
from ..helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume


def solutions_to_fitness_values(solutions: List[S], number_of_objectives: int):
    return dict([(f'f{i + 1}', [s.objectives[i] for s in solutions]) for i in range(number_of_objectives)])


class VerboseHoardingAlgorithmObserver(Observer):
    def __init__(self, number_of_objectives: int):
        self.number_of_objectives = number_of_objectives
        self.population = []
        self.last_evaluations: Optional[int] = None
        self.last_computing_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        self.population = kwargs['SOLUTIONS']
        computing_time = kwargs['COMPUTING_TIME']
        e_diff = evaluations - self.last_evaluations if self.last_evaluations else evaluations
        t_diff = computing_time - self.last_computing_time if self.last_computing_time else computing_time
        cps = e_diff / t_diff if t_diff > 0 else '-'
        best = min(self.population, key=lambda s: np.sum(np.square(s.objectives)))
        self.logger.info(
            f'{evaluations} evaluations / {computing_time:.1f}s @{cps:.2f}cps, '
            f'best: {best.objectives}'
        )
        self.last_evaluations = evaluations
        self.last_computing_time = computing_time

    def get_population(self) -> Dict[str, List[float]]:
        return solutions_to_fitness_values(self.population, self.number_of_objectives)

    def reset(self):
        self.population = []
        self.last_evaluations = None
        self.last_computing_time = None


class HoardingEvaluatorObserver(EvaluatorObserver):
    def __init__(self, number_of_objectives: int):
        self.number_of_objectives = number_of_objectives
        self.solutions: List[S] = []

    def notify(self, solution_list: List[S]):
        for solution in solution_list:
            self.solutions.append(solution)

    def get_new_solutions(self, start: int) -> Dict[str, List[float]]:
        return solutions_to_fitness_values(self.solutions[start:], self.number_of_objectives)

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
        population_generator: Generator,
        kwargs: Dict[str, Any]
) -> Algorithm:
    logger = logging.getLogger(__name__)

    algorithm_kwargs = {}
    if evaluator:
        algorithm_kwargs['population_evaluator'] = evaluator

    if population_generator:
        algorithm_kwargs['population_generator'] = population_generator

    def get_hpsea():
        # FIXME HPSEA?
        nonlocal algorithm_kwargs
        if 'offspring_size' in kwargs and kwargs.get('offspring_size'):
            algorithm_kwargs['offspring_population_size'] = kwargs.get('offspring_size')
        else:
            algorithm_kwargs['offspring_population_size'] = 2 * int(0.5 * 0.2 * population_size)
        return NSGAII[FloatSolution, List[FloatSolution]]

    def get_nsgaii():
        return NSGAII[FloatSolution, List[FloatSolution]]

    def get_ssnsgaii():
        nonlocal algorithm_kwargs
        algorithm_kwargs['offspring_population_size'] = 1
        return NSGAII[FloatSolution, List[FloatSolution]]

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
            termination_criterion=StoppingByEvaluations(max_evaluations),
            mutation=PolynomialMutation(min(3.3 / variables, 1.0), distribution_index=20),
            crossover=SBXCrossover(0.9, distribution_index=15),
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
            evaluator_str: Optional[str] = 'multiprocess',
            algorithm_str: str = 'hpsea',
            plot_server_str: Optional[str] = 'none',
            plot_server_port: int = PlotServer.DEFAULT_PORT,
            auto_start: bool = True,
            v_max: float,
            parameter_labels: List[str],
            ppm3: float = 1.0,
            objectives: list[str],
            reference_front_file: str = None,
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
        self.parameter_labels = parameter_labels
        self.ppm3 = ppm3
        self.objectives = objectives
        self.reference_front_file = reference_front_file
        self.kwargs = kwargs

        # Cache
        self.problem: Optional[HomogenizationProblem] = None
        self.algorithm: Optional[Algorithm] = None

        self.algorithm_observer = VerboseHoardingAlgorithmObserver(len(objectives))
        self.evaluator_observer = HoardingEvaluatorObserver(len(objectives))
        self.evaluator = get_evaluator(
            self.evaluator_str, kwargs=self.kwargs, evaluator_observer=self.evaluator_observer
        )
        self.plot_server = get_plot_server(self.plot_server_str, plot_server_interface=self, port=plot_server_port)
        self.deposition_prefix: Optional[Deposition] = None

    def start(self):
        if self.plot_server:
            self.plot_server.serve_background()

    def run(
            self, *, material: Material, variables: int, deposition_prefix: Deposition = None,
            timestamps: Optional[List[float]] = None,
            population_generator: Generator = None
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
            ppm3=self.ppm3,
            timestamps=timestamps,
            objectives=self.objectives
        )

        self.algorithm = get_algorithm(
            self.algorithm_str,
            problem=self.problem,
            variables=variables,
            population_size=self.population_size,
            max_evaluations=self.max_evaluations,
            evaluator=self.evaluator,
            population_generator=population_generator,
            kwargs=self.kwargs
        )
        self.algorithm.observable.register(self.algorithm_observer)
        self.algorithm.observable.register(WriteFrontToFileObserver(output_directory='./fronts'))

        quality_indicators = [
            HyperVolume(reference_point=[1.0] * len(self.objectives)),
        ]
        if self.reference_front_file:
            reference_front = read_solutions(self.reference_front_file)
            reference_front_objectives = [solution.objectives for solution in reference_front]
            quality_indicators.append(GenerationalDistance(reference_front=reference_front_objectives))
            quality_indicators.append(InvertedGenerationalDistance(reference_front=reference_front_objectives))
        self.algorithm.observable.register(WriteQualityIndicatorsToFileObserver(
            output_file='./quality_indicators.csv',
            quality_indicators=quality_indicators
        ))

        if self.auto_start:
            self.logger.debug('Starting DepositionOptimizer')
        self.start()

        self.logger.debug('Running algorithm')
        self.algorithm.run()
        self.logger.debug('Algorithm finished')

        if self.auto_start:
            self.logger.debug('Stopping DepositionOptimizer')
        self.stop()

        front = get_non_dominated_solutions(self.algorithm.get_result())
        print_function_values_to_file(front, 'FUN')
        print_variables_to_file(front, 'VAR')
        with open('OBJ', 'w') as f:
            f.write(f'{self.problem.get_objective_labels()}')

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
                reclaimed_material=process_material_deposition(
                    material=self.problem.material, deposition=deposition, ppm3=self.ppm3
                )
            )

        raise RuntimeError('DepositionOptimizer not initialized')

    def get_best_solution(self) -> Optional[OptimizationResult]:
        if self.evaluator_observer and self.problem:
            if len(self.algorithm_observer.population) > 0:
                solution = min(
                    self.algorithm_observer.population, key=lambda r: np.sum(np.square(r.objectives))
                )
                deposition = self.problem.variables_to_deposition(solution.variables)
                return OptimizationResult(
                    deposition=deposition,
                    variables=solution.variables,
                    objectives=solution.objectives,
                    objective_labels=self.problem.get_objective_labels(),
                    reclaimed_material=process_material_deposition(
                        material=self.problem.material, deposition=deposition, ppm3=self.ppm3
                    )
                )
            else:
                return None

        raise RuntimeError('DepositionOptimizer not initialized')

    def get_reference(self) -> Optional[OptimizationResult]:
        deposition, material, objectives = self.problem.get_reference_relative()
        return OptimizationResult(
            deposition=deposition,
            variables=[],
            objectives=list(objectives.values()),
            objective_labels=list(objectives.keys()),
            reclaimed_material=material
        )

    def get_ideal_reclaimed_material(self) -> Material:
        _, material, _ = self.problem.get_reference_relative()
        ideal = material.copy()
        for p in material.get_parameter_columns():
            avg = np.average(ideal.data[p], weights=ideal.data['volume'])
            ideal.data[p] = avg
        height = get_stockpile_height(ideal.data['volume'].sum(), self.x_max - self.x_min)
        ideal.data['x_diff'] = (ideal.data['x'] - ideal.data['x'].shift(1)).fillna(0.0)
        ideal.data['volume'] = ideal.data.apply(
            lambda row: get_stockpile_slice_volume(
                row['x'], self.x_max - self.x_min, height, self.x_min, row['x_diff']
            ), axis=1
        )
        return ideal

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
        if self.problem:
            return self.problem.material

        raise RuntimeError('DepositionOptimizer not initialized')

    def get_progress(self) -> Dict[str, float]:
        if self.deposition_prefix:
            return {
                't_start': self.deposition_prefix.meta.time
            }

    def get_parameter_labels(self) -> List[str]:
        return self.parameter_labels
