import math

from scipy.stats import pearsonr

from benchmark_explorer.evaluation import Evaluation
from data_explorer.rating import RaterColorScale, Rater
from data_explorer.testlet import Testlet
from simulator_benchmark.reference_meta import ReferenceMeta


def red_to_green(p):
    return [1 - p, p, 1 - 2 * math.fabs(0.5 - p)]


def get_correlation(standard_reference: ReferenceMeta, evaluation_reference: ReferenceMeta) -> float:
    standard_material = standard_reference.get_reclaimed_material_meta().get_material()
    evaluation_material = evaluation_reference.get_reclaimed_material_meta().get_material()
    a = standard_material.data.copy()
    b = evaluation_material.data.copy()

    if len(a.index) != len(b.index):
        raise ValueError('Index length does not match')

    return pearsonr(a['parameter'].values, b['parameter'].values)[0]


class CorrelationTestlet(Testlet):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def __str__(self):
        return str(self.evaluation)

    def evaluate(self, standard_reference: ReferenceMeta):
        if standard_reference.identifier not in self.evaluation.references:
            return None, f'-'
        value = get_correlation(standard_reference, self.evaluation.references[standard_reference.identifier])
        return value, f'{value:.2f}'

    def get_result_rater(self) -> Rater:
        return RaterColorScale(minimum=0, maximum=1, color_func=red_to_green)


class SimulatorIdentifier(Testlet):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def __str__(self):
        return 'Material'

    def evaluate(self, standard_reference: ReferenceMeta):
        if standard_reference.identifier not in self.evaluation.references:
            return None, f'-'
        value = get_correlation(standard_reference, self.evaluation.references[standard_reference.identifier])
        return value, f'{value:.2f}'


class MaterialIdentifierTestlet(Testlet):
    def __str__(self):
        return 'Material'

    def evaluate(self, standard_reference: ReferenceMeta):
        return str(standard_reference.material), str(standard_reference.material)


class DepositionIdentifierTestlet(Testlet):
    def __str__(self):
        return 'Deposition'

    def evaluate(self, standard_reference: ReferenceMeta):
        return str(standard_reference.deposition), str(standard_reference.deposition)