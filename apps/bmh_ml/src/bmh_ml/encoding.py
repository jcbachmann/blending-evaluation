"""A deposition encoding that lets the stacker stop: positions of the stacker's turning points (knots) with a stop duration at each.

The deposition variables of bmh_ml (`DEPOSITION_LENGTH` positions at equal time steps, within the core of the stockpile) give the stacker no
way to stand still, and the ends of an ideal stockpile need it: Chevron stacking with stops at the turning points
(`bmh.helpers.chevron.stops_path`) builds an almost ideal pile, constant-speed Chevron leaves its ends too low. `StopEncoding` has one
position and one stop duration per knot; the stacker stops at a knot for its duration and then travels to the next one, every travel
segment taking the same time (or, with `travel`, a time of its own). Without stops it is the path of the 20 deposition variables. Its
positions reach beyond the core (`get_stop_limits`), as the turning points of a Chevron with shrinking turning points do. With
`stop_fraction` a last variable ends stacking early: the stacker stacks only that share of the material and the pile is planned for it.
"""

import math
from dataclasses import dataclass

import numpy as np
from bmh.helpers.bed_setup import STACKING_TIME

from bmh_ml.settings import BED_SIZE_X, DEPOSITION_LENGTH, MATERIAL_LENGTH, TOTAL_VOLUME


def get_stop_limits(bed_size_x: float = BED_SIZE_X, total_volume: float = TOTAL_VOLUME, portions: int = MATERIAL_LENGTH) -> tuple[float, float]:
    """The range of the stacker positions that may stop: the bed less the radius of the 45 degree cone of one material portion at each
    end, so that a stop at the end of the range builds a cone of one portion that just reaches the end of the bed (3.6 to 55.4 m on the
    bmh_ml bed). It holds the turning points of every Chevron with shrinking turning points down to one portion per layer."""
    radius = (3.0 * total_volume / portions / math.pi) ** (1.0 / 3.0)
    return radius, bed_size_x - radius


@dataclass(frozen=True)
class StopEncoding:
    """`knots` stacker positions in [x_min, x_max], then `knots` stop durations in [0, max_dwell], then with `travel` the `knots` - 1
    travel times in [min_travel, 1], then with `stop_fraction` (low, high) the share of the material that is stacked. Durations are in
    units of one travel segment (of the longest one with `travel`), the path is scaled to the stacking time (times the stop fraction).
    The defaults hold the Chevron with stops of up to 19 layers (`encode`)."""

    knots: int = DEPOSITION_LENGTH
    x_min: float = get_stop_limits()[0]
    x_max: float = get_stop_limits()[1]
    max_dwell: float = 4.0
    travel: bool = False
    min_travel: float = 0.05
    stop_fraction: tuple[float, float] | None = None

    @property
    def n_var(self) -> int:
        return 2 * self.knots + (self.knots - 1 if self.travel else 0) + (1 if self.stop_fraction else 0)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = [self.x_min] * self.knots + [0.0] * self.knots
        high = [self.x_max] * self.knots + [self.max_dwell] * self.knots
        if self.travel:
            low += [self.min_travel] * (self.knots - 1)
            high += [1.0] * (self.knots - 1)
        if self.stop_fraction:
            low.append(self.stop_fraction[0])
            high.append(self.stop_fraction[1])
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def split(self, variables: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Positions, stop durations, travel times and stop fractions of rows of variables (shape (rows, n_var))."""
        v = np.atleast_2d(np.asarray(variables, dtype=float))
        if v.shape[1] != self.n_var:
            raise ValueError(f"need {self.n_var} variables per row, got {v.shape[1]}")
        k = self.knots
        positions, dwells = v[:, :k], v[:, k : 2 * k]
        travel = v[:, 2 * k : 3 * k - 1] if self.travel else np.ones((len(v), k - 1))
        fractions = v[:, -1] if self.stop_fraction else np.ones(len(v))
        return positions, dwells, travel, fractions

    def decode(self, variables: np.ndarray, duration: float = STACKING_TIME) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """The stacker paths (times, positions; the knots of a piecewise linear path, a stop as two knots at one position) of rows of
        variables and their stop fractions. A path of stop fraction f ends at f `duration`; `bmh_ml.simulation.reclaim_paths` then stacks
        only the material that flows until then."""
        positions, dwells, travel, fractions = self.split(variables)
        paths = []
        for p, d, w, f in zip(positions, dwells, travel, fractions, strict=True):
            # knot j: arrive, stand for d_j, leave; then travel for w_j to knot j + 1
            steps = np.empty(2 * self.knots - 1)
            steps[0::2], steps[1::2] = d, w
            times = np.concatenate([[0.0], np.cumsum(steps)])
            times *= f * duration / times[-1]
            x = np.repeat(p, 2)
            keep = np.concatenate([[True], np.diff(times) > 0])  # a knot without a stop is one knot
            paths.append((times[keep], x[keep]))
        return paths, fractions

    def encode(self, times: np.ndarray, x: np.ndarray, merge: float = 1e-3, duration: float = STACKING_TIME) -> np.ndarray:
        """Variables of a stacker path, e.g. a Chevron schedule (`bmh.helpers.chevron.stops_path`), for a start population or to check
        what the encoding can express. The path is reduced to its turning points and stops; moves shorter than `merge` of the duration (the
        steps between the turning points of a Chevron with shrinking turning points) become part of the stop at their end, and with equal
        travel times every pass takes the mean travel time. Paths with fewer turning points than knots get collinear knots in their
        longest passes. Raises ValueError for more turning points than knots; values beyond the bounds are clipped."""
        times, x = np.asarray(times, dtype=float), np.asarray(x, dtype=float)
        knots_x, dwell, travel = [x[0]], [0.0], []
        direction = 0.0
        for i in range(1, len(times)):
            dt, dx = times[i] - times[i - 1], x[i] - x[i - 1]
            if dt <= 0:
                continue
            if abs(dx) < 1e-9 or dt < merge * duration:
                dwell[-1] += dt
                knots_x[-1] = x[i]
                direction = 0.0 if abs(dx) < 1e-9 else direction
                continue
            sign = math.copysign(1.0, dx)
            if travel and sign == direction and dwell[-1] == 0.0:
                # the same pass continues: move its end
                travel[-1] += dt
                knots_x[-1] = x[i]
            else:
                travel.append(dt)
                knots_x.append(x[i])
                dwell.append(0.0)
            direction = sign
        while len(knots_x) < self.knots:
            j = int(np.argmax(travel))  # split the longest pass in the middle
            knots_x.insert(j + 1, 0.5 * (knots_x[j] + knots_x[j + 1]))
            dwell.insert(j + 1, 0.0)
            travel[j : j + 1] = [0.5 * travel[j], 0.5 * travel[j]]
        if len(knots_x) > self.knots:
            raise ValueError(f"the path has {len(knots_x)} turning points and stops, the encoding {self.knots} knots")
        travel_a, dwell_a = np.array(travel), np.array(dwell)
        unit = travel_a.max() if self.travel else travel_a.mean()
        values = [np.array(knots_x), dwell_a / unit]
        if self.travel:
            values.append(travel_a / unit)
        if self.stop_fraction:
            values.append([(times[-1] - times[0]) / duration])
        low, high = self.bounds
        return np.clip(np.concatenate(values), low, high)


def equal_time_paths(deposition: np.ndarray, duration: float = STACKING_TIME) -> list[tuple[np.ndarray, np.ndarray]]:
    """The paths of the classic deposition variables: positions at equal time steps over the stacking time."""
    deposition = np.atleast_2d(np.asarray(deposition, dtype=float))
    times = np.linspace(0.0, duration, deposition.shape[1])
    return [(times, row) for row in deposition]
