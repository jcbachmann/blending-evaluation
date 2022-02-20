import random

from jmetal.core.solution import FloatSolution
from jmetal.util.generator import Generator

from .homogenization_problem import HomogenizationProblem


class FullSpeedGenerator(Generator):
    def new(self, problem: HomogenizationProblem) -> FloatSolution:
        solution = FloatSolution(
            problem.lower_bound,
            problem.upper_bound,
            problem.number_of_objectives,
            problem.number_of_constraints
        )
        starting_side = random.choice([0, 1])
        solution.variables = [float((i + starting_side) % 2) for i in range(problem.number_of_variables)]
        return solution


class RandomEndGenerator(Generator):
    def new(self, problem: HomogenizationProblem) -> FloatSolution:
        solution = FloatSolution(
            problem.lower_bound,
            problem.upper_bound,
            problem.number_of_objectives,
            problem.number_of_constraints
        )
        solution.variables = [random.choice([0.0, 1.0]) for _ in range(problem.number_of_variables)]
        return solution


class FixedRandomSpeedGenerator(Generator):
    def new(self, problem: HomogenizationProblem) -> FloatSolution:
        solution = FloatSolution(
            problem.lower_bound,
            problem.upper_bound,
            problem.number_of_objectives,
            problem.number_of_constraints
        )
        starting_side = random.choice([0, 1])
        speed = random.randint(1, 10)

        def pos(i):
            nonlocal speed

            p = (i % speed) / speed
            return p if (float(int(i / speed) + starting_side) % 2) == 0 else 1 - p

        solution.variables = [pos(i) for i in range(problem.number_of_variables)]
        return solution


class RandomSpeedGenerator(Generator):
    def new(self, problem: HomogenizationProblem) -> FloatSolution:
        solution = FloatSolution(
            problem.lower_bound,
            problem.upper_bound,
            problem.number_of_objectives,
            problem.number_of_constraints
        )
        offset = 0
        speed = random.randint(1, 10)
        start_dir = random.choice([False, True])

        def pos(i):
            nonlocal offset
            nonlocal speed
            nonlocal start_dir

            i_rel = i - offset

            if i_rel > 0 and i_rel % speed == 0:
                offset = i
                speed = random.randint(1, 10)
                start_dir = not start_dir
                i_rel = 0

            p = (i_rel % speed) / speed
            return p if start_dir else 1 - p

        solution.variables = [pos(i) for i in range(problem.number_of_variables)]
        return solution
