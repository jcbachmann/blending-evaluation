import base64
import gzip

import numpy as np
import pytest

from bmh_ml.front_study.build_interpolation_page import encode_lines, values_at
from bmh_ml.front_study.build_resolution_page import quantize, spearman
from bmh_ml.front_study.common import OUT_ENVIRONMENT_VARIABLE, get_out


def test_the_output_directory_comes_from_the_environment_and_is_not_created_by_asking(tmp_path, monkeypatch):
    monkeypatch.setenv(OUT_ENVIRONMENT_VARIABLE, str(tmp_path / "study"))
    assert get_out() == tmp_path / "study"
    assert not (tmp_path / "study").exists()


def test_pooling_two_groups_of_simulations_gives_the_mean_and_standard_deviation_of_all():
    deepen = pytest.importorskip("bmh_ml.front_study.deepen")
    rng = np.random.default_rng(1)
    first, second = rng.normal(size=(3, 8, 2)), rng.normal(size=(3, 24, 2))
    mean, sd = deepen.pool(np.full(3, 8), first.mean(axis=1), first.std(axis=1, ddof=1), np.full(3, 24), second.mean(axis=1), second.std(axis=1, ddof=1))
    both = np.concatenate([first, second], axis=1)
    np.testing.assert_allclose(mean, both.mean(axis=1))
    np.testing.assert_allclose(sd, both.std(axis=1, ddof=1))


def decode_lines(encoded: dict, n_pairs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """What decodeLines does in the page: per pair the positions and the quantized objectives."""
    values = np.frombuffer(gzip.decompress(base64.b64decode(encoded["data"])), dtype="<i2").astype(int)
    n = encoded["points"]
    counts, columns = values[:n_pairs], values[n_pairs:].reshape(3, n)
    lines, start = [], 0
    for count in counts:
        part = np.cumsum(columns[:, start : start + count], axis=1)
        lines.append((part[0], part[1:].T))
        start += count
    return lines


def test_the_lines_survive_their_compact_encoding_and_it_does_not_change_between_builds():
    state = {  # two pairs, points not sorted, as refine.py appends them
        "pair": np.array([1, 0, 0, 1, 0, 1]),
        "position": np.array([0, 4, 0, 4, 2, 1]),
        "f": np.array([[0.3, 9.0], [0.12345, 4.5678], [0.1, 3.0], [0.25, 12.0], [0.2, 5.5], [0.31, 8.123]]),
    }
    encoded = encode_lines(state, n_pairs=2, grid=4)
    assert encoded == encode_lines(state, n_pairs=2, grid=4)
    (positions0, values0), (positions1, values1) = decode_lines(encoded, 2)
    assert positions0.tolist() == [0, 2, 4]
    assert values0.tolist() == [[1000, 3000], [2000, 5500], [1234, 4568]]
    assert positions1.tolist() == [0, 1, 4]
    assert values1.tolist() == [[3000, 9000], [3100, 8123], [2500, 12000]]


def test_values_at_a_base_position_are_found_for_every_line():
    state = {"pair": np.array([0, 1, 0, 1]), "position": np.array([2, 2, 0, 0]), "f": np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])}
    assert values_at(state, 2, 2).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_only_segments_that_jump_beyond_the_noise_and_can_still_be_halved_are_refined():
    refine = pytest.importorskip("bmh_ml.front_study.refine")
    unit = np.array([0.01, 0.1])
    state = {
        "pair": np.array([0, 0, 0, 1, 1]),
        "position": np.array([0, 4, 8, 0, 1]),
        "f": np.array([[0.10, 1.0], [0.10, 1.05], [0.30, 1.0], [0.10, 1.0], [0.50, 5.0]]),
    }
    pair, position, jump = refine.candidates(state, unit)
    assert pair.tolist() == [0]  # pair 0: flat from 0 to 4, a jump from 4 to 8; pair 1 jumps, but its segment is one grid step long
    assert position.tolist() == [6]
    assert jump[0] > refine.THRESHOLD


def test_simulations_are_quantized_as_the_pages_decode_them():
    samples = np.array([[[0.12345, 4.5678], [0.2, 5.0]]])
    assert quantize(samples) == [[1234, 4568, 2000, 5000]]
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
