import matplotlib.pyplot as plt
import pandas as pd

from .material_handler import MaterialHandler


class MaterialSampler:
    def __init__(self, sampler_buffer_size):
        self.material_handlers = []
        self.samples = []
        self.sampler_buffer_size = sampler_buffer_size

    def put(self, material_handler: MaterialHandler):
        self.material_handlers.append(material_handler)

    def sample(self, time):
        samples_row = [time]
        for material_handler in self.material_handlers:
            material_handler_samples = material_handler.sample()
            for tph, quality in material_handler_samples:
                samples_row.extend([tph, quality])
        self.samples.append(samples_row)
        if len(self.samples) > self.sampler_buffer_size:
            self.samples.pop(0)

    def evaluate(self):
        columns = ['time']
        for material_handler in self.material_handlers:
            n = len(material_handler.sample())
            for i in range(n):
                columns.extend([f'{material_handler.label} tph {i}', f'{material_handler.label} quality {i}'])

        df = pd.DataFrame(data=self.samples, columns=columns)
        df = df.set_index('time')

        axs = df.filter(regex='.* quality').replace([0], value=[None]).plot.hist(
            subplots=True,
            grid=True,
            title='Material Quality Histogram',
            bins=15
        )
        for ax in axs:
            ax.set_xlabel('Quality')
            ax.set_ylabel('Amount')

        df_quality = df.filter(regex='.* quality').replace([0], value=[None])
        ma = df_quality.rolling(20).mean()
        mstd = df_quality.rolling(20).std()
        ax = ma.plot.line(
            layout=(len(self.material_handlers), 1),
            grid=True,
            title='Material Quality Graph 5 Minute Average'
        )
        for col in df_quality.columns:
            ax.fill_between(df_quality.index, ma[col] - 2 * mstd[col], ma[col] + 2 * mstd[col], alpha=0.3, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Quality')
        ax.set_xlim(0, None)

        df_tph = df.filter(regex='.* tph')
        ma = df_tph.rolling(20).mean()
        mstd = df_tph.rolling(20).std()
        ax = ma.plot.line(
            layout=(len(self.material_handlers), 1),
            grid=True,
            title='Material Volume Graph 5 Minute Average'
        )
        for col in df_tph.columns:
            ax.fill_between(df_tph.index, ma[col] - 2 * mstd[col], ma[col] + 2 * mstd[col], alpha=0.3, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Tons per Hour')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.show()
