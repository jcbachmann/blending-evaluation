import logging
import random
from abc import abstractmethod
from typing import List, Tuple

import numpy as np


class MaterialHandler:
    dot = None

    def __init__(self, label: str, plant, shape=None):
        self.label = label
        self.plant = plant
        self.logger = logging.getLogger(__name__)
        if MaterialHandler.dot is not None:
            MaterialHandler.dot.attr('node', shape=shape if shape else 'ellipse')
            MaterialHandler.dot.node(label, label)

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def gen(self, i: int = 0):
        pass

    def unpack_src_gen(self, src):
        if isinstance(src, MaterialHandler):
            if MaterialHandler.dot is not None:
                MaterialHandler.dot.edge(src.label, self.label)
            return src.gen()
        elif isinstance(src, tuple):
            if MaterialHandler.dot is not None:
                MaterialHandler.dot.edge(src[0].label, self.label)
            return src[0].gen(src[1])
        elif isinstance(src, list):
            if MaterialHandler.dot is not None:
                for s in src:
                    MaterialHandler.dot.edge((s if isinstance(s, MaterialHandler) else s[0]).label, self.label)
            return [(s.gen() if isinstance(s, MaterialHandler) else s[0]) for s in src]
        else:
            raise Exception('Invalid source type passed to MaterialHandler.unpack')


class MaterialSource(MaterialHandler):
    def __init__(self, label: str, plant):
        super().__init__(label, plant, 'doublecircle')

    def sample(self):
        pass

    def gen(self, i: int = 0):
        pass


class MaterialBuffer(MaterialHandler):
    def __init__(self, label, plant, src, steps: int):
        super().__init__(label + ' [' + str(steps) + ']', plant, 'cds')
        self.src_gen = self.unpack_src_gen(src)
        self.buffer: List[Tuple[float, float]] = []
        for _ in range(steps):
            self.buffer.append((0.0, 0.0))
        self._sample = (0.0, 0.0)

    def gen(self, i: int = 0):
        while True:
            self.buffer.append(next(self.src_gen))
            tph, q = self.buffer.pop(0)
            self.logger.debug(
                f'Buffer {self.label}: {", ".join([f"({tph:.1f}, {q:.1f})" for tph, q in self.buffer])}')
            self.logger.debug(f'Buffer {self.label} out: ({tph:.1f}, {q:.1f})')
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]


class MaterialJoiner(MaterialHandler):
    def __init__(self, label, plant, src_x):
        super().__init__(label, plant, 'trapezium')
        self.src_gens = self.unpack_src_gen(src_x)
        self._sample = (0.0, 0.0)

    def gen(self, i: int = 0):
        while True:
            m = [next(src_gen) for src_gen in self.src_gens]
            tph = sum(tph for tph, _ in m)
            q = np.average([q for _, q in m], weights=[tph for tph, _ in m]) if tph > 0 else 0
            self.logger.debug(f'Joiner {self.label}: ({tph:.1f}, {q:.1f})')
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]


class MaterialSplitter(MaterialHandler):
    def __init__(self, label, plant, src, weights):
        super().__init__(label, plant, 'invtrapezium')
        self.src_gen = self.unpack_src_gen(src)
        self.weights = weights
        self.buffer: List[List[Tuple[float, float]]] = [[] for _ in range(len(weights))]
        self._sample = [(0.0, 0.0)] * len(weights)

    def gen(self, i: int = 0):
        while True:
            if len(self.buffer[i]) == 0:
                self.acquire_buffer()
            tph, q = self.buffer[i].pop(0)
            self.logger.debug(f'Splitter {self.label}.{i}: ({tph:.1f}, {q:.1f})')
            yield tph, q

    def acquire_buffer(self):
        tph, q = next(self.src_gen)
        for i, weight in enumerate(self.weights):
            self.buffer[i].append((weight * tph, q))
            self._sample[i] = (weight * tph, q)

    def sample(self):
        return self._sample


class MaterialMux(MaterialHandler):
    def __init__(self, label: str, plant, src_x, weight_matrix, flip_probability: float = 0):
        super().__init__(label, plant, 'pentagon')
        self.src_gens = self.unpack_src_gen(src_x)
        self.weight_matrix = weight_matrix
        self.flip_probability = flip_probability
        self.flip = False
        self.buffer: List[List[Tuple[float, float]]] = [[] for _ in range(len(weight_matrix))]
        self._sample = [(0.0, 0.0)] * len(weight_matrix)

    def sample(self):
        return self._sample

    def gen(self, i: int = 0):
        while True:
            if len(self.buffer[i]) == 0:
                self.acquire_buffer()
            tph, q = self.buffer[i].pop(0)
            self.logger.debug(f'Splitter {self.label}.{i}: ({tph:.1f}, {q:.1f})')
            yield tph, q

    def acquire_buffer(self):
        m = [next(src_gen) for src_gen in self.src_gens]
        in_tphs = [tph for tph, _ in m]
        in_qs = [q for _, q in m]
        out_tphs = [float(np.dot(in_tphs, weights)) for weights in self.weight_matrix]
        out_qs = [
            np.average(in_qs, weights=np.multiply(in_tphs, weights)) if sum(np.multiply(in_tphs, weights)) > 0 else 0
            for i, weights in enumerate(self.weight_matrix)]
        self._sample = list(zip(out_tphs, out_qs))
        if self.flip:
            self._sample.reverse()
        if random.random() < self.flip_probability:
            self.flip = not self.flip
        for i, s in enumerate(self._sample):
            self.buffer[i].append(s)


class MaterialDuplicator(MaterialHandler):
    def __init__(self, label, plant, src, count):
        super().__init__(label, plant, 'Mdiamond')
        self.src_gen = self.unpack_src_gen(src)
        self.buffer: List[List[Tuple[float, float]]] = [[] for _ in range(count)]
        self._sample = [(0.0, 0.0)] * count
        self.count = count

    def gen(self, i: int = 0):
        while True:
            if len(self.buffer[i]) == 0:
                self.acquire_buffer()
            tph, q = self.buffer[i].pop(0)
            self.logger.debug(f'Splitter {self.label}.{i}: ({tph:.1f}, {q:.1f})')
            yield tph, q

    def acquire_buffer(self):
        tph, q = next(self.src_gen)
        for i in range(self.count):
            self.buffer[i].append((tph, q))
            self._sample[i] = (tph, q)

    def sample(self):
        return self._sample


class MaterialOut(MaterialHandler):
    def __init__(self, label, plant, src):
        super().__init__(label, plant, 'doubleoctagon')
        self.src_gen = self.unpack_src_gen(src)
        self._sample = (0.0, 0.0)

    def step(self):
        tph, q = next(self.src_gen)
        self.logger.debug(f'Destination {self.label}: ({tph:.1f}, {q:.1f})')
        # Inconsistently sample input instead of output as there is no output but sampling might be interesting
        self._sample = (tph, q)

    def gen(self, i: int = 0):
        return None

    def sample(self):
        return [self._sample]
