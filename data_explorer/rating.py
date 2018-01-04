from enum import Enum


class Rating(Enum):
    NEUTRAL = 1
    GOOD = 2
    WARNING = 3
    BAD = 4
    ERROR = 5
    CUSTOM = 6

    def to_color(self):
        if self == Rating.CUSTOM:
            raise Exception('Illegal call to Rating.to_color() for CUSTOM rating')

        return {
            Rating.NEUTRAL: None,
            Rating.GOOD: [0xC7, 0xF4, 0x64],
            Rating.WARNING: [0xFA, 0xC7, 0x68],
            Rating.BAD: [0xFF, 0x6B, 0x6B],
            Rating.ERROR: [0xC0, 0x3B, 0x3B],
        }[self]


class Rater:
    def rate(self, value):
        raise Exception('abstract')

    def get_color(self, value):
        raise Exception('CUSTOM rating requires implementation of Rater.get_color()')


class RaterCheckValue(Rater):
    def __init__(self, low_bad=None, low_warning=None, high_warning=None, high_bad=None):
        self.low_bad = low_bad
        self.low_warning = low_warning
        self.high_warning = high_warning
        self.high_bad = high_bad

    def rate(self, value):
        if self.low_bad and value <= self.low_bad:
            return Rating.BAD

        if self.high_bad and value >= self.high_bad:
            return Rating.BAD

        if self.low_warning and value <= self.low_warning:
            return Rating.WARNING

        if self.high_warning and value >= self.high_warning:
            return Rating.WARNING

        return Rating.GOOD


class RaterValueEquals(Rater):
    def __init__(self, target, warning_only=False):
        self.target = target
        self.warning_only = warning_only

    def rate(self, value):
        if self.target == value:
            return Rating.GOOD
        else:
            if self.warning_only:
                return Rating.WARNING
            else:
                return Rating.BAD


class RaterColorScale(Rater):
    def __init__(self, minimum, maximum, color_func):
        self.minimum = minimum
        self.maximum = maximum
        self.color_func = color_func

        if minimum >= maximum:
            raise Exception(f'RaterColorScale requires minimum [{minimum}] < maximum [{maximum}]')

    def rate(self, value):
        return Rating.CUSTOM

    def get_color(self, value):
        if value < self.minimum or value > self.maximum:
            raise Exception(f'Range violation minimum [{self.minimum}] <= value [{value}] <= maximum [{self.maximum}]')

        return self.color_func((value - self.minimum) / (self.maximum - self.minimum))
