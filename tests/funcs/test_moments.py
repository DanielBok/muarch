import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from muarch.funcs import get_annualized_kurtosis, get_annualized_mean, get_annualized_sd, get_annualized_skew


@pytest.fixture
def data_cube() -> np.ndarray:
    np.random.seed(888)
    return np.random.uniform(-0.05, 0.05, size=(20, 500, 5))


@pytest.fixture
def frequency():
    return "quarter"


def test_get_annualized_mean(data_cube, frequency):
    assert_almost_equal(get_annualized_mean(data_cube, frequency),
                        [-1.58544394e-04, 3.95923527e-05, 8.10572621e-04, -3.15037014e-03, -8.93454642e-04])


def test_get_annualized_sd(data_cube, frequency):
    assert_almost_equal(get_annualized_sd(data_cube, frequency),
                        [0.05841895, 0.05774354, 0.05773541, 0.05719393, 0.05780493])


def test_get_annualized_skew(data_cube, frequency):
    assert_almost_equal(get_annualized_skew(data_cube, frequency),
                        [0.12223643, 0.18185045, 0.09259107, 0.13573941, 0.11479943])


def test_get_annualized_kurtosis(data_cube, frequency):
    assert_almost_equal(get_annualized_kurtosis(data_cube, frequency),
                        [-0.27932077, -0.33324643, -0.40724923, -0.33458672, -0.2948958])
