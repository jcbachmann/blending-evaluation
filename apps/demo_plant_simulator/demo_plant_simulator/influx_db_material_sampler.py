import datetime

from influxdb import InfluxDBClient
from plant_simulator.material_sampler import MaterialSampler


# Experimental influx db sampler
class InfluxDbMaterialSampler(MaterialSampler):
    def __init__(self, buffer_size: int, group_size: int, stats_size: int, stats_period: float):
        super().__init__(buffer_size, group_size, stats_size, stats_period)
        self.influxdb_client = InfluxDBClient('localhost', 8086, 'root', 'root', 'influxdb')

    def sample(self, time):
        timestamp_str = (datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=time)).isoformat()

        json = []
        for label, material_handler in self.material_handlers:
            material_handler_samples = material_handler.sample()
            for tph, quality in material_handler_samples:
                json.append({
                    'measurement': 'sample',
                    'tags': {
                        'material_handler': label
                    },
                    'time': timestamp_str + 'Z',
                    'fields': {
                        'tph': float(tph),
                        'quality': float(quality)
                    }
                })

        self.influxdb_client.write_points(json)
