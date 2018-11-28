import argparse
import datetime
import logging
import random

from graphviz import Digraph
from plant_simulator.material_handler import MaterialHandler
from plant_simulator.plant import Plant
from plant_simulator.plot_server import PlotServer

from .my_demo_plant import MyDemoPlant


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info('Initializing')

    if args.seed is not None:
        logger.info(f'Setting random seed to {args.seed}')
        random.seed(args.seed)

    if args.graph:
        MaterialHandler.dot = Digraph('material-flow')
        MaterialHandler.dot.attr(rankdir='LR')

    plant = MyDemoPlant(args.evaluate, args.path)

    if args.graph:
        MaterialHandler.dot.render()

    if args.evaluate:
        logger.info('Starting background plot server')
        plot_server = PlotServer(plant.get_columns())
        plot_server.set_data_callback(plant.get_diff_live, plant.get_diff_stats)
        plot_server.serve_background()

    time = datetime.timedelta(0, args.max_steps * Plant.TIME_INCREMENT, 0)
    logger.info(f'Starting simulation of {args.max_steps} steps = {time}')
    start = datetime.datetime.now()
    step = 0
    while step < args.max_steps:
        plant.simulate_step()
        step += 1
    end = datetime.datetime.now()
    logger.info(f'Simulation finished in {end - start}')

    if args.evaluate:
        logger.info('Evaluating')
        plant.evaluate()

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plant Simulator')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum simulation steps')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate material sampling')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--graph', action='store_true', help='Render graph of plant layout to PDF')
    parser.add_argument('--path', type=str, default='.', help='Path where output data is stored')

    main(parser.parse_args())
