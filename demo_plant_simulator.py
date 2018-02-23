import argparse
import datetime
import logging
import random

from graphviz import Digraph

from plant_simulator.material_handler import MaterialBuffer, MaterialOut, MaterialMux, MaterialHandler
from plant_simulator.plant import Plant
from plant_simulator.plot_server import PlotServer
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
                MaterialBuffer('BSA', source_a, steps=4),
                MaterialBuffer('BSB', source_b, steps=2),
                MaterialBuffer('BSC', source_c, steps=1)
            ],
            weight_matrix=[
                [1, 0.5, 0],
                [0, 0.5, 1]
            ],
            flip_probability=0.01
        )

        out_a = MaterialOut('OA', MaterialBuffer('BOA', (mux, 0), steps=1))
        out_b = MaterialOut('OB', MaterialBuffer('BOB', (mux, 1), steps=1))

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

    MaterialHandler.dot = Digraph('material-flow')
    MaterialHandler.dot.attr(rankdir='LR')
    plant = MyDemoPlant(args.evaluate)
    MaterialHandler.dot.render()

    logging.info('Starting background plot server')
    plot_server = PlotServer(plant.get_columns())
    plot_server.set_data_callback(plant.get_diff)
    plot_server.serve_background()

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
