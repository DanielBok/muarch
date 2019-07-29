import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from muarch.calibrate import calibrate_data


@pytest.mark.parametrize('target_mean, target_vol', [
    (None, None),
    ([0.3, -0.1], None),
    (None, [0.6, 0.1]),
    ([0.3, 0.1], [0.6, 0.1]),
])
def test_calibrate(target_mean, target_vol):
    np.random.seed(888)
    data = np.random.uniform(-0.05, 0.05, (36, 100, 2))
    calibrated = calibrate_data(data, target_mean, target_vol)

    if target_mean:
        assert_almost_equal(annualized_mean(calibrated), target_mean, 4)
    if target_vol:
        assert_almost_equal(annualized_vol(calibrated), target_vol, 4)

    if target_mean is None and target_vol is None:
        assert_almost_equal(data, calibrated)


def annualized_mean(data: np.ndarray):
    d = (data + 1).prod(0)
    return (np.sign(d) * np.abs(d) ** (1 / 3)).mean(0) - 1


def annualized_vol(data: np.ndarray):
    d = []
    for i in range(3):
        d.append(((data[i * 12: (i + 1) * 12] + 1).prod(0) - 1).std(0))

    return np.mean(d, 0)
