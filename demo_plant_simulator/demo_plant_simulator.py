import argparse
import datetime
import logging
import random

from graphviz import Digraph

from plant_simulator.blending_system import BlendingSystem
from plant_simulator.material_handler import MaterialBuffer, MaterialOut, MaterialMux, MaterialHandler, \
    MaterialDuplicator
from plant_simulator.plant import Plant
from plant_simulator.plot_server import PlotServer
from plant_simulator.simulated_mine import SimulatedMine


class MyDemoPlant(Plant):
    def __init__(self, evaluate: bool):
        super().__init__(evaluate, sampler_buffer_size=57600, sample_group_size=1,
                         stats_size=int(2 * 3600 / (Plant.TIME_INCREMENT * 6)), stats_period=1800)

        # Create and link material handlers
        source_a = SimulatedMine('Great Mine', max_tph=4000, availability=0.9, q_min=0.20, q_exp=0.23, q_max=0.30)
        source_b = SimulatedMine('Huge Mine', max_tph=11300, availability=0.99, q_min=0.28, q_exp=0.35, q_max=0.40)
        source_c = SimulatedMine('Some Mine', max_tph=7000, availability=0.97, q_min=0.25, q_exp=0.31, q_max=0.38)

        mux = MaterialMux(
            label='Mux',
            src_x=[
                MaterialBuffer('Great Mine Transport', source_a, steps=4),
                MaterialBuffer('Huge Mine Transport', source_b, steps=2),
                MaterialBuffer('Some Mine Transport', source_c, steps=1)
            ],
            weight_matrix=[
                [1, 0.5, 0],
                [0, 0.5, 1]
            ],
            flip_probability=0.01
        )

        duplicator_a = MaterialDuplicator('Duplicator A', MaterialBuffer('Mux Transport A', (mux, 0), steps=1), count=3)
        blend_a_a = BlendingSystem('A Blending Strategy A', (duplicator_a, 0), strategy='A')
        blend_a_b = BlendingSystem('A Blending Strategy B', (duplicator_a, 1), strategy='B')
        # blend_a_a = BlendingSystem('A Blending Simulator Fast', (duplicator_a, 0), simulator='fast')
        # blend_a_b = BlendingSystem('A Blending Simulator Simple', (duplicator_a, 1), simulator='mathematical')

        duplicator_b = MaterialDuplicator('Duplicator B', MaterialBuffer('Mux Transport B', (mux, 1), steps=2), count=3)
        blend_b_a = BlendingSystem('B Blending Strategy A', (duplicator_b, 0), strategy='A')
        blend_b_b = BlendingSystem('B Blending Strategy B', (duplicator_b, 1), strategy='B')

        out_a_blend_a = MaterialOut('Out A Blending Strategy A', blend_a_a)
        out_a_blend_b = MaterialOut('Out A Blending Strategy B', blend_a_b)
        out_a_no_blend = MaterialOut('Out A No Blending', (duplicator_a, 2))

        out_b_blend_a = MaterialOut('Out B Blending Strategy A', blend_b_a)
        out_b_blend_b = MaterialOut('Out B Blending Strategy B', blend_b_b)
        out_b_no_blend = MaterialOut('Out B No Blending', (duplicator_b, 2))

        # Specify outs as simulation hooks
        self.material_outs = [
            out_a_blend_a, out_a_blend_b, out_a_no_blend,
            out_b_blend_a, out_b_blend_b, out_b_no_blend
        ]

        if evaluate:
            # Set up sampling at interesting sampling points
            self.sampler.put('Out A, Strategy A', out_a_blend_a)
            self.sampler.put('Out A, Strategy B', out_a_blend_b)
            self.sampler.put('Out A, No Blending', out_a_no_blend)
            self.sampler.put('Out B, Strategy A', out_b_blend_a)
            self.sampler.put('Out B, Strategy B', out_b_blend_b)
            self.sampler.put('Out B, No Blending', out_b_no_blend)


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )
    logging.info('Initializing')

    if args.seed is not None:
        logging.info(f'Setting random seed to {args.seed}')
        random.seed(args.seed)

    MaterialHandler.dot = Digraph('material-flow')
    MaterialHandler.dot.attr(rankdir='LR')
    plant = MyDemoPlant(args.evaluate)
    MaterialHandler.dot.render()

    if args.evaluate:
        logging.info('Starting background plot server')
        plot_server = PlotServer(plant.get_columns())
        plot_server.set_data_callback(plant.get_diff_live, plant.get_diff_stats)
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
