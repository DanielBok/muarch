import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from muarch import UArch
from muarch.datasets import load_etf
from muarch.summary import Summary


@pytest.fixture('module')
def returns():
    return load_etf().iloc[:, 0]


@pytest.fixture('module')
def model(returns):
    model = UArch('AR', dist='skewt')
    model.fit(returns)
    return model


def test_forecast(model, returns):
    horizon, start = 2, 3
    forecast = model.forecast(start=start, horizon=horizon)

    ideal_shape = len(returns), horizon
    assert forecast.residual_variance.shape == ideal_shape
    assert forecast.mean.shape == ideal_shape
    assert forecast.variance.shape == ideal_shape

    # nas for the skipped starts
    assert np.alltrue(forecast.mean.iloc[:start].isna())
    assert np.alltrue(~forecast.mean.iloc[start:].isna())


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_plots(model):
    assert isinstance(model.hedgehog_plot(model.params), plt.Figure)
    assert isinstance(model.residual_plot(), plt.Figure)


def test_residuals(model, returns):
    assert len(model.residuals()) == len(returns)


@pytest.mark.parametrize('mean, vol, dist, err_msg', [
    ('WRONG_MEAN', 'garch', 'normal', "Unknown model type 'wrong_mean' in mean"),
    ('arx', 'WRONG_VOL', 'normal', "Unknown model type 'wrong_vol' in vol"),
    ('arx', 'garch', 'WRONG_DIST', "Unknown model type 'wrong_dist' in dist")
])
def test_wrong_specs_raises(mean, vol, dist, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        UArch(mean=mean, vol=vol, dist=dist)


def test_summary(model):
    assert isinstance(model.summary(short=True), pd.Series)

    summary = model.summary()
    assert isinstance(summary, Summary)
    assert isinstance(summary.as_text(), str)
    assert isinstance(summary.as_csv(), str)
    assert isinstance(summary.as_latex(), str)
    assert isinstance(summary.as_html(), str)


def test__repr_html_(model):
    assert isinstance(model._repr_html_(), str)


def test___repr__(model):
    assert isinstance(repr(model), str)
