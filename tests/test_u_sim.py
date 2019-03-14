from itertools import product

import numpy as np
import pandas as pd
import pytest

from muarch import UArch

np.random.seed(8)
TRAIN_SIZE = 500
HORIZON = 100
REPS = 5
BURN = 500

Y = np.random.normal(size=TRAIN_SIZE)
X = np.random.normal(size=(TRAIN_SIZE, 1))
X_SIM = np.random.normal(size=(HORIZON + BURN, 1))

known_setup = list(product(
    ('zero', 'constant', 'harx', 'har', 'ar', 'arx', 'ls'),  # mean
    ('arch', 'figarch', 'garch', 'harch', 'constant', 'egarch'),  # vol
    ('normal', 't', 'skewt', 'ged')  # dist
))


@pytest.mark.parametrize("mean, vol, dist", known_setup)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_simulation(mean, vol, dist):
    lags = 1 if mean in ('harx', 'har', 'ar', 'arx') else 0

    x, x_sim = (X, X_SIM) if mean == 'ls' else (None, None)

    model = UArch(mean=mean, lags=lags, vol=vol, dist=dist, scale=100)
    model.fit(Y, x, show_warning=False)  # mask convergence warning
    sim = model.simulate(HORIZON, x=x_sim, burn=BURN, data_only=True)
    assert len(sim) == HORIZON

    sim = model.simulate(HORIZON, x=x_sim)
    assert isinstance(sim, pd.DataFrame)


@pytest.mark.parametrize("mean, vol, dist", known_setup)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_simulation_mc(mean, vol, dist):
    lags = 1 if mean in ('harx', 'har', 'ar', 'arx') else 0

    x, x_sim = (X, X_SIM) if mean == 'ls' else (None, None)

    model = UArch(mean=mean, lags=lags, vol=vol, dist=dist, scale=100)
    model.fit(Y, x, show_warning=False)  # mask convergence warning

    if vol in ('figarch', 'harch'):
        with pytest.raises(NotImplementedError):
            assert model.simulate_mc(HORIZON, REPS, x=x_sim).shape == (HORIZON, REPS)
    else:
        assert model.simulate_mc(HORIZON, REPS, x=x_sim).shape == (HORIZON, REPS)


def test_simulation_mc_errors():
    model = UArch('AR', 1, scale=100)
    model.fit(Y)

    with pytest.raises(ValueError, match='reps must be an integer greater than 0'):
        model.simulate_mc(500, 5.5)
    with pytest.raises(ValueError, match='reps must be an integer greater than 0'):
        model.simulate_mc(500, -2)
