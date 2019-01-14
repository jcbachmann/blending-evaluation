import random
from abc import ABC, abstractmethod
from typing import List, Tuple


class SolutionGenerator(ABC):
    @abstractmethod
    def gen(self, v: int) -> List[float]:
        pass


class RandomSolutionGenerator(SolutionGenerator):
    def gen(self, v: int) -> List[float]:
        return [random.uniform(0.0, 1.0) for _ in range(v)]

    def __str__(self):
        return 'RandomSolutionGenerator'


class FullSpeedSolutionGenerator(SolutionGenerator):
    def gen(self, v: int) -> List[float]:
        starting_side = random.choice([0, 1])
        return [float((i + starting_side) % 2) for i in range(v)]

    def __str__(self):
        return 'FullSpeedSolutionGenerator'


class RandomEndSolutionGenerator(SolutionGenerator):
    def gen(self, v: int) -> List[float]:
        return [random.choice([0.0, 1.0]) for _ in range(v)]

    def __str__(self):
        return 'RandomEndSolutionGenerator'


class FixedRandomSpeedSolutionGenerator(SolutionGenerator):
    def gen(self, v: int) -> List[float]:
        starting_side = random.choice([0, 1])
        speed = random.randint(1, 10)

        def pos(i):
            nonlocal speed

            p = (i % speed) / speed
            return p if (float(int(i / speed) + starting_side) % 2) == 0 else 1 - p

        return [pos(i) for i in range(v)]

    def __str__(self):
        return 'FixedRandomSpeedSolutionGenerator'


class RandomSpeedSolutionGenerator(SolutionGenerator):
    def gen(self, v: int) -> List[float]:
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

        return [pos(i) for i in range(v)]

    def __str__(self):
        return 'RandomSpeedSolutionGenerator'


class PoolSolutionGenerator(SolutionGenerator):
    def __init__(self, solution_pool: List[List[float]]):
        self.solution_pool = solution_pool

    def gen(self, v: int) -> List[float]:
        return random.choice(self.solution_pool)

    def __str__(self):
        return 'PoolSolutionGenerator'


class MultiSolutionGenerator(SolutionGenerator):
    def __init__(self, weighted_generators: List[Tuple[SolutionGenerator, int]]):
        self.generators: List[SolutionGenerator] = [c[0] for c in weighted_generators]
        self.weights: List[int] = [c[1] for c in weighted_generators]

    def gen(self, v: int) -> List[float]:
        return random.choices(self.generators, weights=self.weights)[0].gen(v)

    def __str__(self):
        return f'MultiSolutionGenerator{str(self.generators)}'
