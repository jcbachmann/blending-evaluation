"""Non-dominated filters and hypervolumes of objective vectors, all objectives minimized.

A point dominates another if it is at least as good in every objective and better in one (Pareto dominance); equal points do not
dominate each other, so duplicates of a non-dominated point are all kept. `get_weakly_nondominated` is the weaker filter that only
removes points another point beats in every objective, so it keeps points that are as good as another one in some objectives and worse
in the rest.
"""

import numpy as np

_CHUNK = 5000


def dominated_by(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """For each point, whether one of `others` dominates it."""
    points, others = np.asarray(points, dtype=float), np.asarray(others, dtype=float)
    result = np.zeros(len(points), dtype=bool)
    for chunk in range(0, len(points), _CHUNK):
        p = points[chunk : chunk + _CHUNK, None, :]
        result[chunk : chunk + _CHUNK] = np.any(np.all(others[None] <= p, axis=2) & np.any(others[None] < p, axis=2), axis=1)
    return result


def _beaten_in_every_objective(points: np.ndarray) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for chunk in range(0, len(points), _CHUNK):
        result[chunk : chunk + _CHUNK] = np.any(np.all(points[None] < points[chunk : chunk + _CHUNK, None, :], axis=2), axis=1)
    return result


def _dominated_2d(objectives: np.ndarray, weakly: bool) -> np.ndarray:
    """Both filters for two objectives in O(n log n): sorted by F1, a point is beaten by an earlier point with a lower F1 and an F2 at
    least as low (`weakly`: lower), or by a point of the same F1 with a lower F2."""
    f1, f2 = objectives[:, 0], objectives[:, 1]
    order = np.lexsort((f2, f1))
    s1, s2 = f1[order], f2[order]
    first = np.concatenate([[True], s1[1:] != s1[:-1]])
    group = np.cumsum(first) - 1
    group_min = np.minimum.reduceat(s2, np.flatnonzero(first))
    before = np.concatenate([[np.inf], np.minimum.accumulate(group_min)[:-1]])  # lowest F2 of all points with a lower F1
    if weakly:
        beaten = before[group] < s2
    else:
        beaten = (before[group] <= s2) | (s2 > group_min[group])
    result = np.empty(len(objectives), dtype=bool)
    result[order] = beaten
    return result


def get_nondominated(objectives: np.ndarray) -> np.ndarray:
    """The rows of the points no other point dominates, in increasing order."""
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return np.zeros(0, dtype=int)
    dominated = _dominated_2d(objectives, weakly=False) if objectives.shape[1] == 2 else dominated_by(objectives, objectives)
    return np.flatnonzero(~dominated)


def get_weakly_nondominated(objectives: np.ndarray) -> np.ndarray:
    """The rows of the points no other point beats in every objective, in increasing order. Unlike `get_nondominated` it keeps weakly
    dominated points."""
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return np.zeros(0, dtype=int)
    beaten = _dominated_2d(objectives, weakly=True) if objectives.shape[1] == 2 else _beaten_in_every_objective(objectives)
    return np.flatnonzero(~beaten)


def get_hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """The area that the points dominate up to the reference point, for two objectives. Points that are not better than the reference
    point in both objectives add nothing."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    reference_point = np.asarray(reference_point, dtype=float)
    points = points[np.all(points < reference_point, axis=1)]
    area, lowest_f2 = 0.0, reference_point[1]
    for f1, f2 in points[np.lexsort((points[:, 1], points[:, 0]))]:
        if f2 < lowest_f2:
            area += (reference_point[0] - f1) * (lowest_f2 - f2)
            lowest_f2 = f2
    return float(area)
