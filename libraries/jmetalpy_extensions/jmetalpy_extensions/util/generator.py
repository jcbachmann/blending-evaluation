import copy
import random
from typing import Tuple, TypeVar, List

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.generator import Generator

R = TypeVar('R')


class RandomChoiceInjectorGenerator(Generator):
    def __init__(self, solutions: List[Solution]):
        self.solutions = solutions

    def new(self, problem: Problem) -> R:
        return copy.deepcopy(random.choice(self.solutions))


class MultiGenerator(Generator):
    def __init__(self, weighted_generators: List[Tuple[Generator, float]]):
        self.generators, self.weights = zip(*weighted_generators)

    def new(self, problem: Problem) -> R:
        return random.choices(self.generators, weights=self.weights)[0].new(problem)
