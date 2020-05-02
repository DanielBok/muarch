from typing import Collection

import numpy as np
import pytest

from muarch import MUArch, UArch

np.random.seed(8)
N = 3  # number of models
TRAIN_SIZE = 500
HORIZON = 100
REPS = 5
BURN = 500

Y = np.random.normal(size=(TRAIN_SIZE, N))
X = [np.random.normal(size=(TRAIN_SIZE, 1)), None, None]
X_SIM = [np.random.normal(size=(HORIZON + BURN, 1)), None, None]


def random(size):
    size = (*tuple(size), N) if isinstance(size, Collection) else (size, N)
    return np.random.uniform(size=size)


@pytest.fixture
def models():
    m = MUArch(N, 'AR', lags=1, scale=100, dist='skewt')
    m.fit(Y)
    return m


def test_simulation(models):
    sim = models.simulate(HORIZON, BURN)
    assert sim.shape == (HORIZON, N)


def test_simulation_with_exog():
    models = MUArch(N, scale=100)
    models[0] = UArch('LS', scale=100)

    models.fit(Y, X)
    sim = models.simulate(HORIZON, BURN, x=X_SIM)
    assert sim.shape == (HORIZON, N)


@pytest.mark.parametrize('n_jobs', [None, 1, 2])
def test_simulation_mc(models, n_jobs):
    sim = models.simulate_mc(HORIZON, REPS, BURN, n_jobs=n_jobs)
    assert sim.shape == (HORIZON, REPS, N)


def test_simulation_mc_with_rng(models):
    sim = models.simulate_mc(HORIZON, REPS, BURN, custom_dist=random)
    assert sim.shape == (HORIZON, REPS, N)


@pytest.mark.parametrize('kwargs, error', [
    ({'reps': 5.5}, ValueError),
    ({'reps': -1}, ValueError),
    ({'reps': 5, 'n_jobs': 1.5}, ValueError)
])
def test_simulation_mc_errors(models, kwargs, error):
    with pytest.raises(error):
        models.simulate_mc(HORIZON, **kwargs)
