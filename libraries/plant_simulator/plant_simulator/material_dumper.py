import json
import os
import random
import string

from .material_handler import MaterialHandler
from .plant import Plant


class MaterialDumper(MaterialHandler):
    def __init__(self, label, plant, src, path):
        super().__init__(label, plant)
        self.src_gen = self.unpack_src_gen(src)
        self._sample = (0, 0)
        self.path = path
        self.time_reference = None
        self.time_limit = None
        self.buffer = None
        self.total_volume = None

        self.reset_random_limit()

    def gen(self, i: int = 0):
        while True:
            tph, q = next(self.src_gen)

            timestamp = self.plant.time - self.time_reference
            volume = tph * Plant.TIME_INCREMENT / 3600
            parameter = q * 100
            self.buffer.append([timestamp, volume, parameter])
            self.total_volume += volume

            if self.plant.time > self.time_limit:
                self.save_file()
                self.reset_random_limit()

            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]

    def save_file(self):
        rnd_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        identifier = self.label.replace(' ', '_') + '_' + str(rnd_id)
        file_dir = os.path.join(self.path, 'generated_material', identifier)
        os.makedirs(file_dir)

        with open(os.path.join(file_dir, 'data.csv'), 'w') as f:
            lines = ['\t'.join([str(i) for i in row]) for row in self.buffer]
            f.write('\n'.join(lines))

        json.dump({
            'label': f'{self.label} {rnd_id}',
            'description': f'Generated material stream from MaterialDumper with label {self.label}',
            'category': 'generated',
            'time': self.plant.time - self.time_reference,
            'volume': self.total_volume,
            'data': 'data.csv'
        }, open(os.path.join(file_dir, 'meta.json'), 'w'), indent=4)

    def reset_random_limit(self):
        self.time_limit = self.plant.time + random.uniform(10, 48) * 60 * 60
        self.time_reference = self.plant.time
        self.buffer = [['timestamp', 'volume', 'parameter']]
        self.total_volume = 0
