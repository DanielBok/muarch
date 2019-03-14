import numpy as np
import pandas as pd
import pytest

from muarch import MUArch, UArch
from muarch.datasets import load_etf
from muarch.summary import SummaryList


@pytest.fixture('module')
def returns():
    return load_etf()


@pytest.fixture('module')
def model(returns):
    model = MUArch(3, 'AR', dist='skewt')
    model.fit(returns)
    return model


@pytest.mark.parametrize('n, err', [
    (3, None),
    ([UArch(), UArch(), UArch()], None),
    ([], ValueError),
    ([UArch(), 213], TypeError),
    (UArch(), TypeError)
])
def test_mu_specs_correctly(n, err):
    if err is None:
        MUArch(3)
    else:
        with pytest.raises(err):
            MUArch(n)


def test_mu_params(model):
    assert isinstance(model.params, pd.DataFrame)


def test_mu_residuals(model):
    assert isinstance(model.residuals(), np.ndarray)


def test_mu_summary(model):
    assert isinstance(model.summary(short=True), pd.DataFrame)
    assert isinstance(model.summary(), SummaryList)


def test__repr_html_(model):
    assert isinstance(model._repr_html_(), str)


def test__repr__(model):
    assert isinstance(repr(model), str)
