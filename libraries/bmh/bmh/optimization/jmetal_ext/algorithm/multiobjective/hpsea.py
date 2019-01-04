from typing import TypeVar, List, Optional

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.density_estimator import CrowdingDistance
from jmetal.component.evaluator import SequentialEvaluator, Evaluator
from jmetal.component.ranking import Ranking
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem

S = TypeVar('S')
R = TypeVar(List[S])


# TODO upstream temporary code (huge performance improvement)
class MyFastNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II. """

    def __init__(self):
        super(MyFastNonDominatedRanking, self).__init__()

    def compute_ranking(self, solution_list: List[S]):
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solution_list))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solution_list))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solution_list) + 1)]

        number_of_comparisons = self.number_of_comparisons

        for p in range(len(solution_list) - 1):
            for q in range(p + 1, len(solution_list)):
                # TODO constraint violation not considered for performance reasons
                solution_p = solution_list[p]
                solution_q = solution_list[q]
                better_p = False
                better_q = False

                for i in range(solution_p.number_of_objectives):
                    value_p = solution_p.objectives[i]
                    value_q = solution_q.objectives[i]

                    if value_p < value_q:
                        better_p = True
                    elif value_p > value_q:
                        better_q = True

                if better_p and not better_q:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif better_q and not better_p:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

                number_of_comparisons += 1

        self.number_of_comparisons = number_of_comparisons

        for i in range(len(solution_list)):
            if dominating_ith[i] is 0:
                front[0].append(i)
                solution_list[i].attributes['dominance_ranking'] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] is 0:
                            front[i].append(q)
                            solution_list[q].attributes['dominance_ranking'] = i

        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for k in range(len(front[j])):
                q[k] = solution_list[front[j][k]]
            self.ranked_sublists[j] = q

        return self.ranked_sublists


class MyRankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):

    def __init__(self, max_population_size: int):
        super(MyRankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size

    def execute(self, front: List[S]) -> List[S]:
        ranking = MyFastNonDominatedRanking()
        crowding_distance = CrowdingDistance()
        ranking.compute_ranking(front)

        ranking_index = 0
        new_solution_list = []

        while len(new_solution_list) < self.max_population_size:
            if len(ranking.get_subfront(ranking_index)) < self.max_population_size - len(new_solution_list):
                new_solution_list = new_solution_list + ranking.get_subfront(ranking_index)
                ranking_index += 1
            else:
                subfront = ranking.get_subfront(ranking_index)
                crowding_distance.compute_density_estimator(subfront)
                sorted_subfront = sorted(subfront, key=lambda x: x.attributes['crowding_distance'], reverse=True)
                for i in range((self.max_population_size - len(new_solution_list))):
                    new_solution_list.append(sorted_subfront[i])

        return new_solution_list

    def get_name(self) -> str:
        return 'Ranking and crowding distance selection'


class HPSEA(GenerationalGeneticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 evaluator: Evaluator[S] = SequentialEvaluator[S](),
                 offspring_size: Optional[int] = None):
        super(HPSEA, self).__init__(
            problem,
            population_size,
            max_evaluations,
            mutation,
            crossover,
            selection,
            evaluator)
        self.offspring_size = offspring_size if offspring_size is not None else 2 * int(0.5 * 0.2 * population_size)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[TypeVar('S')]]:
        join_population = population + offspring_population
        return MyRankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

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
                           'computing time': self.get_current_computing_time(),
                           'population': self.population,
                           'reference_front': self.problem.reference_front}

        self.observable.notify_all(**observable_data)
