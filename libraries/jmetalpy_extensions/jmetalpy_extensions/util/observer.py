from typing import List

from jmetal.core.observer import Observer
from jmetal.core.quality_indicator import QualityIndicator


class WriteQualityIndicatorsToFileObserver(Observer):
    def __init__(self, output_file: str, quality_indicators: List[QualityIndicator]) -> None:
        self.output_file = output_file
        self.quality_indicators = quality_indicators

        with open(self.output_file, 'w+') as of:
            of.write(f'{",".join([indicator.get_short_name() for indicator in quality_indicators])}\n')

    def update(self, *args, **kwargs):
        solutions = kwargs['SOLUTIONS']
        if not solutions:
            return

        objectives = [solution.objectives for solution in solutions]
        results = [str(indicator.compute(objectives)) for indicator in self.quality_indicators]

        with open(self.output_file, 'a+') as of:
            of.write(','.join(results) + '\n')
