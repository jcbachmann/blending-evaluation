import itertools

import numpy as np
import pytest

from ..fronts import dominated_by, get_hypervolume_2d, get_nondominated, get_weakly_nondominated


def brute_force(points: np.ndarray, weakly: bool) -> list[int]:
    def beats(q, p):
        return bool(np.all(q < p)) if weakly else bool(np.all(q <= p) and np.any(q < p))

    return [i for i, p in enumerate(points) if not any(beats(q, p) for q in points)]


def test_dominance_needs_one_better_objective_and_none_worse():
    points = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 3.0], [2.0, 2.0]])

    assert dominated_by(points, points).tolist() == [False, True, True, False]
    assert get_nondominated(points).tolist() == [0, 3]


def test_duplicates_of_a_nondominated_point_are_all_kept():
    points = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])

    assert get_nondominated(points).tolist() == [0, 1, 2]


def test_the_weak_filter_keeps_points_that_tie_in_one_objective():
    points = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 5.0]])

    assert get_weakly_nondominated(points).tolist() == [0, 1, 3, 4]


@pytest.mark.parametrize("objectives", [2, 3])
@pytest.mark.parametrize("weakly", [False, True])
def test_the_filters_agree_with_the_definitions_on_points_with_many_ties(objectives, weakly):
    rng = np.random.default_rng(objectives)
    for _ in range(50):
        points = rng.integers(0, 5, (int(rng.integers(1, 30)), objectives)).astype(float)
        found = get_weakly_nondominated(points) if weakly else get_nondominated(points)
        assert found.tolist() == brute_force(points, weakly)


def test_the_filters_of_two_objectives_agree_with_the_pairwise_comparison_on_large_sets():
    points = np.random.default_rng(0).integers(0, 200, (6_000, 2)).astype(float)

    assert get_nondominated(points).tolist() == np.flatnonzero(~dominated_by(points, points)).tolist()


def test_empty_sets_have_no_front():
    assert get_nondominated(np.zeros((0, 2))).tolist() == []
    assert get_weakly_nondominated(np.zeros((0, 3))).tolist() == []


def test_the_hypervolume_is_the_union_of_the_boxes_up_to_the_reference_point():
    assert get_hypervolume_2d(np.array([[0.5, 0.5]]), np.ones(2)) == pytest.approx(0.25)
    assert get_hypervolume_2d(np.array([[0.5, 0.5], [0.8, 0.2]]), np.ones(2)) == pytest.approx(0.25 + 0.16 - 0.1)
    assert get_hypervolume_2d(np.array([[0.5, 1.2], [1.1, 0.1], [1.0, 0.5]]), np.ones(2)) == 0.0
    assert get_hypervolume_2d(np.zeros((0, 2)), np.ones(2)) == 0.0


def test_the_hypervolume_agrees_with_counting_the_cells_of_the_grid_of_all_coordinates():
    rng = np.random.default_rng(1)
    for _ in range(20):
        points = rng.random((int(rng.integers(1, 15)), 2)) * 1.2
        reference = np.array([1.0, 1.1])
        xs, ys = np.unique(np.r_[points[:, 0], reference[0]]), np.unique(np.r_[points[:, 1], reference[1]])
        area = 0.0
        for (x0, x1), (y0, y1) in itertools.product(itertools.pairwise(xs), itertools.pairwise(ys)):
            if x1 <= reference[0] and y1 <= reference[1] and np.any((points[:, 0] <= x0) & (points[:, 1] <= y0)):
                area += (x1 - x0) * (y1 - y0)
        assert get_hypervolume_2d(points, reference) == pytest.approx(area)
