from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator, Evaluator
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RankingAndCrowdingDistanceSelection
from jmetal.util.observable import Observable, DefaultObservable

S = TypeVar('S')
R = TypeVar(List[S])


class SSNSGAII(GenerationalGeneticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 observable: Observable = DefaultObservable(),
                 evaluator: Evaluator[S] = SequentialEvaluator[S]()):
        super(SSNSGAII, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            observable,
            evaluator)

    def selection(self, population: List[S]):
        return [
            self.selection_operator.execute(self.population),
            self.selection_operator.execute(self.population)
        ]

    def reproduction(self, population: List[S]) -> List[S]:
        # Acquire one set of parents
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        parents = [population[j] for j in range(number_of_parents_to_combine)]

        # Generate offspring
        offspring = self.crossover_operator.execute(parents)

        # Use the first offspring only
        self.mutation_operator.execute(offspring[0])
        return [offspring[0]]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[TypeVar('S')]]:
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

    def get_name(self) -> str:
        return "Steady State NSGA-II"

    def get_result(self) -> R:
        return self.population

    def update_progress(self):
        self.evaluations += 1

        observable_data = {'evaluations': self.evaluations,
                           'population': self.population,
                           'computing time': self.get_current_computing_time()}

        self.observable.notify_all(**observable_data)
