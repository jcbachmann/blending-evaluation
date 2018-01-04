#!/usr/bin/env python
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MaterialSource:
    def __init__(self, label, max_tph, availability, q_min, q_exp, q_max):
        self.label = label
        self.max_tph = max_tph
        self.availability = availability
        self.currently_unavailable = False

        self.q_min = q_min
        self.q_exp = q_exp
        self.q_max = q_max

        self.tph_current = self.max_tph * random.triangular(0, self.max_tph, self.max_tph)
        self.q_current = self.q_exp

        self.failure_probability = 0.0005
        if availability >= 1.0:
            self.repair_probability = 1.0
        else:
            self.repair_probability = min(self.failure_probability / (1 - availability), 1.0)

    def get_material(self):
        return self.tph_current, None if self.currently_unavailable else self.q_current

    def step(self, t, t_diff):
        if self.currently_unavailable:
            if random.random() < self.repair_probability:
                self.currently_unavailable = False
        else:
            if random.random() < self.failure_probability:
                self.currently_unavailable = True

        if self.currently_unavailable:
            self.tph_current = 0
        else:
            self.tph_current = self.max_tph * random.triangular(0, self.max_tph, self.max_tph)

        max_change = 0.003
        limit = self.q_max if self.q_current > self.q_exp else self.q_min
        comp = 0.5 + 0.5 * (self.q_current - self.q_exp) / abs(limit - self.q_exp)
        change = (max_change if random.random() > comp else -max_change) * random.random()
        self.q_current = self.q_current + change


class MaterialStreamProvider:
    def __init__(self):
        self.sources = []
        self.t = 0

    def add_source(self, source: MaterialSource):
        self.sources.append(source)

    def initialize(self):
        pass

    def step(self, t, t_diff):
        self.t = t
        for source in self.sources:
            source.step(t, t_diff)

    def get_materials(self):
        df = pd.DataFrame()
        df['time'] = [self.t]
        for source in self.sources:
            tph, quality = source.get_material()
            df[source.label + ' tph'] = [tph]
            df[source.label + ' quality'] = [quality]
        return df


def main(args):
    msp = MaterialStreamProvider()
    msp.add_source(MaterialSource('Great Mine', max_tph=4000, availability=0.9, q_min=0.20, q_exp=0.23, q_max=0.30))
    msp.add_source(MaterialSource('Huge Mine', max_tph=11300, availability=0.99, q_min=0.28, q_exp=0.35, q_max=0.40))
    msp.add_source(MaterialSource('Some Mine', max_tph=7000, availability=0.97, q_min=0.25, q_exp=0.31, q_max=0.38))

    df = pd.DataFrame()

    t_diff = 15
    t_max = 86400  # one day
    # t_max = 3600  # one hour
    for t in np.linspace(t_diff, t_max, int(t_max / t_diff + 0.5)):
        msp.step(t, t_diff)
        df = df.append(msp.get_materials())

    df.filter(regex='.* quality').plot(
        subplots=True,
        grid=True,
        kind='hist'
    )
    df.filter(regex='(.* quality|time)').plot(
        x='time',
        layout=(len(msp.sources), 1),
        grid=True,
        kind='line'
    )
    df.filter(regex='(.* tph|time)').plot(
        x='time',
        layout=(len(msp.sources), 1),
        grid=True,
        kind='line'
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best cone shape')

    main(parser.parse_args())
