from muarch.calibrate import calibrate_data
import numpy as np
from numpy.testing import assert_almost_equal


def test_calibrate():
    np.random.seed(888)
    data = np.random.normal(0, 0.25, (36, 100, 2))

    target_mean = [0.3, -0.1]
    target_vol = [0.6, 0.1]
    data = calibrate_data(data, target_mean, target_vol)

    assert_almost_equal(annualized_mean(data), target_mean, 4)
    assert_almost_equal(annualized_vol(data), target_vol, 4)


def annualized_mean(data: np.ndarray):
    d = (data + 1).prod(0)
    return (np.sign(d) * np.abs(d) ** (1 / 3)).mean(0) - 1


def annualized_vol(data: np.ndarray):
    d = []
    for i in range(3):
        d.append(((data[i * 12: (i + 1) * 12] + 1).prod(0) - 1).std(0))

    return np.mean(d, 0)
