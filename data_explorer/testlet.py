import logging
import traceback
from typing import Union

import matplotlib.pyplot

from data_explorer.rating import Rater, Rating

matplotlib.pyplot.switch_backend('Qt5Agg')


class TestletResult:
    def __init__(self, model, view, rating: Rating, color=None):
        self.model = model
        self.view = view
        self.rating = rating
        self.color = color


class Testlet:
    def wrapped_evaluate(self, entry):
        try:
            logging.debug(f'evaluating {self} on {entry}')
            result_model, result_view = self.evaluate(entry)
            rating, color = self.get_rating(result_model)
            return TestletResult(result_model, result_view, rating, color)
        except Exception as e:
            logging.error(f'Exception in wrapped_evaluate: {self} {entry} {e} {traceback.format_exc()}')
            return TestletResult(e, str(e), Rating.ERROR)

    def evaluate(self, entry):
        raise RuntimeError(f'{self}.execute() not defined')

    def get_result_rater(self) -> Union[Rater, None]:
        return None

    def get_rating(self, result_model):
        rater = self.get_result_rater()
        if rater is None:
            return Rating.NEUTRAL, None
        else:
            rating = rater.rate(result_model)
            color = rater.get_color(result_model) if rating == Rating.CUSTOM else rating.to_color()
            return rating, color
