import pytest

from ..stockpile import Stockpile


@pytest.mark.parametrize("simulator", ["fast", "smooth", "mathematical"])
def test_stack_reclaim(simulator: str):
    stockpile = Stockpile(length=40.0, depth=10.0, simulator=simulator)
    stacked_tons = 0.0
    for i in range(30):
        stockpile.stack(timestamp=15.0 * i, x=5.0 + i, z=5.0, tons=10.0, q=float(i % 3))
        stacked_tons += 10.0

    reclaimed_tons = 0.0
    while not stockpile.reclaiming_finished():
        tons, _ = stockpile.reclaim(speed=0.5)
        reclaimed_tons += tons

    assert reclaimed_tons == pytest.approx(stacked_tons)
