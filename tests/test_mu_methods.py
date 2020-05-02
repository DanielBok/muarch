import numpy as np
import pandas as pd
import pytest

from muarch import MUArch, UArch
from muarch.datasets import load_etf
from muarch.summary import SummaryList


@pytest.fixture(scope='module')
def returns():
    return load_etf()


@pytest.fixture(scope='module')
def model(returns):
    model = MUArch(3, 'AR', dist='skewt', scale=0.1)
    model.fit(returns)
    return model


@pytest.mark.parametrize('n, err', [
    (3, None),
    ([UArch(), UArch(), UArch()], None),
    ([], AssertionError),
    ([UArch(), 213], AssertionError),
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

    smry: SummaryList = model.summary()
    smry[0].add_header('Header Text')  # add headers
    smry[0], smry[1] = smry[1], smry[0]  # swaps

    assert isinstance(smry, SummaryList)
    assert isinstance(str(smry), str)
    assert isinstance(repr(smry), str)
    assert isinstance(smry.as_csv(), str)
    assert isinstance(smry.as_html(), str)
    assert isinstance(smry.as_latex(), str)
    assert isinstance(smry.as_text(), str)
    assert isinstance(smry._repr_html_(), str)

    with pytest.raises(ValueError):
        smry.append(0)


def test__repr_html_(model):
    assert isinstance(model._repr_html_(), str)


def test__repr__(model):
    assert isinstance(repr(model), str)
