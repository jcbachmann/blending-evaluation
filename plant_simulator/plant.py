import logging

from .material_sampler import MaterialSampler


class Plant:
    TIME_INCREMENT = 15

    def __init__(self, evaluate, sampler_buffer_size, sample_group_size, stats_size, stats_period):
        self.material_outs = []
        self.time = 0
        self.sampler = MaterialSampler(sampler_buffer_size, sample_group_size, stats_size, stats_period) if evaluate else None

    def simulate_step(self):
        self.time += Plant.TIME_INCREMENT
        logging.debug(f'--- Simulating step at t = {self.time}')
        for material_out in self.material_outs:
            material_out.step()
        if self.sampler:
            self.sampler.sample(self.time)
        logging.debug('-----------------------------------------------------')

    def evaluate(self):
        self.sampler.evaluate()

    def get_diff_live(self, start):
        return self.sampler.get_diff_live(start)

    def get_diff_stats(self, start):
        return self.sampler.get_diff_stats(start)

    def get_columns(self):
        return self.sampler.get_columns()
