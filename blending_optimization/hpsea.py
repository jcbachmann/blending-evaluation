from typing import TypeVar, List

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator, Evaluator
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import RankingAndCrowdingDistanceSelection
from jmetal.util.observable import Observable, DefaultObservable

S = TypeVar('S')
R = TypeVar(List[S])


class HPSEA(GenerationalGeneticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 observable: Observable = DefaultObservable(),
                 evaluator: Evaluator[S] = SequentialEvaluator[S](),
                 offspring_size: int = 20):
        super(HPSEA, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            observable,
            evaluator)
        self.offspring_size = offspring_size

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[TypeVar('S')]]:
        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

    def get_name(self) -> str:
        return 'HPSEA'

    def get_result(self) -> R:
        return self.population

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.offspring_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        offspring_population = []
        for i in range(0, self.offspring_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

        return offspring_population

    def update_progress(self):
        self.evaluations += self.offspring_size

        observable_data = {'evaluations': self.evaluations,
                           'population': self.population,
                           'computing time': self.get_current_computing_time()}

        self.observable.notify_all(**observable_data)
