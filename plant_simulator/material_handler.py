import logging
import queue
import random
from abc import abstractmethod

import numpy as np


class MaterialHandler:
    dot = None

    def __init__(self, label: str, shape=None):
        self.label = label
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
            MaterialHandler.dot.edge(src.label, self.label)
            return src.gen()
        elif isinstance(src, tuple):
            MaterialHandler.dot.edge(src[0].label, self.label)
            return src[0].gen(src[1])
        elif isinstance(src, list):
            for s in src:
                MaterialHandler.dot.edge((s if isinstance(s, MaterialHandler) else s[0]).label, self.label)
            return [(s.gen() if isinstance(s, MaterialHandler) else s[0]) for s in src]
        else:
            raise Exception('Invalid source type passed to MaterialHandler.unpack')


class MaterialSource(MaterialHandler):
    def __init__(self, label: str):
        super().__init__(label, 'doublecircle')

    def sample(self):
        pass

    def gen(self, i: int = 0):
        pass


class MaterialBuffer(MaterialHandler):
    def __init__(self, label, src, steps: int):
        super().__init__(label + ' [' + str(steps) + ']', 'cds')
        self.src_gen = self.unpack_src_gen(src)
        self.buffer = queue.Queue()
        for _ in range(steps):
            self.buffer.put((0, 0))
        self._sample = (0, 0)

    def gen(self, i: int = 0):
        while True:
            self.buffer.put(next(self.src_gen))
            tph, q = self.buffer.get()
            logging.debug(
                f'Buffer {self.label}: {", ".join([f"({tph:.1f}, {q:.1f})" for tph, q in self.buffer.queue])}')
            logging.debug(f'Buffer {self.label} out: ({tph:.1f}, {q:.1f})')
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]


class MaterialJoiner(MaterialHandler):
    def __init__(self, label, src_x):
        super().__init__(label, 'trapezium')
        self.src_gens = self.unpack_src_gen(src_x)
        self._sample = (0, 0)

    def gen(self, i: int = 0):
        while True:
            m = [next(src_gen) for src_gen in self.src_gens]
            tph = sum(tph for tph, _ in m)
            q = np.average([q for _, q in m], weights=[tph for tph, _ in m]) if tph > 0 else 0
            logging.debug(f'Joiner {self.label}: ({tph:.1f}, {q:.1f})')
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]


class MaterialSplitter(MaterialHandler):
    def __init__(self, label, src, weights):
        super().__init__(label, 'invtrapezium')
        self.src_gen = self.unpack_src_gen(src)
        self.weights = weights
        self.buffer = [queue.Queue() for _ in range(len(weights))]
        self._sample = [(0, 0)] * len(weights)

    def gen(self, i: int = 0):
        while True:
            if self.buffer[i].empty():
                self.acquire_buffer()
            tph, q = self.buffer[i].get()
            logging.debug(f'Splitter {self.label}.{i}: ({tph:.1f}, {q:.1f})')
            yield tph, q

    def acquire_buffer(self):
        tph, q = next(self.src_gen)
        for i, weight in enumerate(self.weights):
            self.buffer[i].put((weight * tph, q))
            self._sample[i] = (weight * tph, q)

    def sample(self):
        return self._sample


class MaterialMux(MaterialHandler):
    def __init__(self, label: str, src_x, weight_matrix, flip_probability: float = 0):
        super().__init__(label, 'pentagon')
        self.src_gens = self.unpack_src_gen(src_x)
        self.weight_matrix = weight_matrix
        self.flip_probability = flip_probability
        self.flip = False
        self.buffer = [queue.Queue() for _ in range(len(weight_matrix))]
        self._sample = [(0, 0)] * len(weight_matrix)

    def sample(self):
        return self._sample

    def gen(self, i: int = 0):
        while True:
            if self.buffer[i].empty():
                self.acquire_buffer()
            tph, q = self.buffer[i].get()
            logging.debug(f'Splitter {self.label}.{i}: ({tph:.1f}, {q:.1f})')
            yield tph, q

    def acquire_buffer(self):
        m = [next(src_gen) for src_gen in self.src_gens]
        in_tphs = [tph for tph, _ in m]
        in_qs = [q for _, q in m]
        out_tphs = [np.dot(in_tphs, weights) for weights in self.weight_matrix]
        out_qs = [
            np.average(in_qs, weights=np.multiply(in_tphs, weights)) if sum(np.multiply(in_tphs, weights)) > 0 else 0
            for i, weights in enumerate(self.weight_matrix)]
        self._sample = list(zip(out_tphs, out_qs))
        if self.flip:
            self._sample.reverse()
        if random.random() < self.flip_probability:
            self.flip = not self.flip
        for i, s in enumerate(self._sample):
            self.buffer[i].put(s)


class MaterialOut(MaterialHandler):
    def __init__(self, label, src):
        super().__init__(label, 'doubleoctagon')
        self.src_gen = self.unpack_src_gen(src)
        self._sample = (0, 0)

    def step(self):
        tph, q = next(self.src_gen)
        logging.debug(f'Destination {self.label}: ({tph:.1f}, {q:.1f})')
        # Inconsistently sample input instead of output as there is no output but sampling might be interesting
        self._sample = (tph, q)

    def gen(self, i: int = 0):
        return None

    def sample(self):
        return [self._sample]
