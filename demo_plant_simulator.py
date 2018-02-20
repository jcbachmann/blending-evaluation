import argparse
import datetime
import logging
import random

from plant_simulator.material_handler import MaterialBuffer, MaterialOut, MaterialMux
from plant_simulator.plant import Plant
from plant_simulator.simulated_mine import SimulatedMine


class MyDemoPlant(Plant):
    def __init__(self, evaluate):
        super().__init__(evaluate, 5760)

        # Create and link material handlers
        source_a = SimulatedMine('Great Mine', max_tph=4000, availability=0.9, q_min=0.20, q_exp=0.23, q_max=0.30)
        source_b = SimulatedMine('Huge Mine', max_tph=11300, availability=0.99, q_min=0.28, q_exp=0.35, q_max=0.40)
        source_c = SimulatedMine('Some Mine', max_tph=7000, availability=0.97, q_min=0.25, q_exp=0.31, q_max=0.38)

        mux = MaterialMux(
            label='Mux',
            src_x=[
                MaterialBuffer('BSA', source_a.gen(), 4).gen(),
                MaterialBuffer('BSB', source_b.gen(), 2).gen(),
                MaterialBuffer('BSC', source_c.gen(), 1).gen()
            ],
            weight_matrix=[
                [1, 0.5, 0],
                [0, 0.5, 1]
            ],
            flip_probability=0.01
        )

        out_a = MaterialOut('OA', MaterialBuffer('BOA', mux.gen(0), 1).gen())
        out_b = MaterialOut('OB', MaterialBuffer('BOB', mux.gen(1), 2).gen())

        # Specify outs as simulation hooks
        self.material_outs = [out_a, out_b]

        # Set up sampling at interesting sampling points
        self.sampler.put(source_a)
        self.sampler.put(source_b)
        self.sampler.put(source_c)
        self.sampler.put(out_a)
        self.sampler.put(out_b)


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Initializing')

    if args.seed is not None:
        logging.info(f'Setting random seed to {args.seed}')
        random.seed(args.seed)

    plant = MyDemoPlant(args.evaluate)

    time = datetime.timedelta(0, args.max_steps * Plant.TIME_INCREMENT, 0)
    logging.info(f'Starting simulation of {args.max_steps} steps = {time}')
    start = datetime.datetime.now()
    step = 0
    while step < args.max_steps:
        plant.simulate_step()
        step += 1
    end = datetime.datetime.now()
    logging.info(f'Simulation finished in {end - start}')

    if args.evaluate:
        logging.info('Evaluating')
        plant.evaluate()
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plant Simulator')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum simulation steps')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate material sampling')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    main(parser.parse_args())
