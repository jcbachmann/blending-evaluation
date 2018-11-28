from plant_simulator.blending_system.blending_system import BlendingSystem
from plant_simulator.material_dumper import MaterialDumper
from plant_simulator.material_handler import MaterialMux, MaterialBuffer, MaterialDuplicator, MaterialOut
from plant_simulator.plant import Plant
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
