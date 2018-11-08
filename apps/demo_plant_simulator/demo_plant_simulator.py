import argparse
import datetime
import logging
import random

from graphviz import Digraph
from plant_simulator.blending_system.blending_system import BlendingSystem
from plant_simulator.material_dumper import MaterialDumper
from plant_simulator.material_handler import MaterialBuffer, MaterialOut, MaterialMux, MaterialHandler, \
    MaterialDuplicator
from plant_simulator.plant import Plant
from plant_simulator.plot_server import PlotServer
from plant_simulator.simulated_mine import SimulatedMine


class MyDemoPlant(Plant):
    def __init__(self, evaluate: bool, path: str):
        super().__init__(evaluate, sampler_buffer_size=57600, sample_group_size=1,
                         stats_size=int(2 * 3600 / (Plant.TIME_INCREMENT * 6)), stats_period=1800)

        # Create and link material handlers
        source_a = SimulatedMine('Great Mine', max_tph=4000, availability=0.9, q_min=0.20, q_exp=0.23, q_max=0.30,
                                 plant=self)
        source_b = SimulatedMine('Huge Mine', max_tph=11300, availability=0.99, q_min=0.28, q_exp=0.35, q_max=0.40,
                                 plant=self)
        source_c = SimulatedMine('Some Mine', max_tph=7000, availability=0.97, q_min=0.25, q_exp=0.31, q_max=0.38,
                                 plant=self)

        mux = MaterialMux(
            label='Mux',
            src_x=[
                MaterialBuffer('Great Mine Transport', src=source_a, steps=4, plant=self),
                MaterialBuffer('Huge Mine Transport', src=source_b, steps=2, plant=self),
                MaterialBuffer('Some Mine Transport', src=source_c, steps=1, plant=self)
            ],
            weight_matrix=[
                [1, 0.5, 0],
                [0, 0.5, 1]
            ],
            flip_probability=0.01,
            plant=self
        )

        mux_a = MaterialDumper('A Before Blending', src=(mux, 0), path=path, plant=self)
        mux_b = MaterialDumper('B Before Blending', src=(mux, 1), path=path, plant=self)

        duplicator_a = MaterialDuplicator('Duplicator A',
                                          src=MaterialBuffer('Mux Transport A', src=mux_a, steps=1, plant=self),
                                          count=2, plant=self)
        blend_a_chevron_50 = BlendingSystem('Blending Strategy B', src=(duplicator_a, 1), strategy='Chevron',
                                            strategy_params={'layers': 50, 'by': 'mass'}, plant=self)
        blend_a_pile = BlendingSystem('A Blending Strategy A', src=(duplicator_a, 0), strategy='Pile', plant=self)

        out_a_chevron_50 = MaterialOut('Out A Chevron 50', src=blend_a_chevron_50, plant=self)
        out_a_pile = MaterialOut('Out A Pile', src=blend_a_pile, plant=self)
        out_b = MaterialOut('Out B', src=mux_b, plant=self)

        # Specify outs as simulation hooks
        self.material_outs = [
            out_a_chevron_50, out_a_pile, out_b
        ]

        if evaluate:
            # Set up sampling at interesting sampling points
            self.sampler.put('Chevron 50', out_a_chevron_50)
            self.sampler.put('Pile', out_a_pile)


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )
    logging.info('Initializing')

    if args.seed is not None:
        logging.info(f'Setting random seed to {args.seed}')
        random.seed(args.seed)

    if args.graph:
        MaterialHandler.dot = Digraph('material-flow')
        MaterialHandler.dot.attr(rankdir='LR')

    plant = MyDemoPlant(args.evaluate, args.path)

    if args.graph:
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
    parser.add_argument('--graph', action='store_true', help='Render graph of plant layout to PDF')
    parser.add_argument('--path', type=str, default='.', help='Path where output data is stored')

    main(parser.parse_args())
