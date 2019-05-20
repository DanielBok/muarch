import warnings
from typing import Iterable, Optional

import numpy as np
import scipy.optimize as opt

__all__ = ['calibrate_data', 'truncate_outliers']


def calibrate_data(data: np.ndarray, mean: Optional[Iterable[float]] = None, sd: Optional[Iterable[float]] = None,
                   time_unit=12):
    """
    Calibrates the data given the target mean and standard deviation.

    Parameters
    ----------
    data: ndarray
        Data tensor to calibrate

    mean: iterable float, optional
        The target annual mean vector

    sd: iterable float, optional
        The target annual standard deviation (volatility) vector

    time_unit: int, optional
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month

    Returns
    -------
    ndarray
        An instance of the adjusted numpy tensor
    """
    data = data.copy()
    s = data.shape

    def calc_annualized_mean(d: np.ndarray, years: float):
        d = (d + 1).prod(0)
        s = np.sign(d)
        return (s * np.abs(d) ** (1 / years)).mean(0) - 1

    def set_target(d: Optional[Iterable[float]], defaults: np.ndarray):
        if d is None:
            return defaults

        assert s[2] == len(d) == len(defaults), "vector length must be equal to number of assets in data cube"
        for i, v in enumerate(d):
            if v is None:
                d[i] = defaults[i]
        return d

    y = s[0] // time_unit
    target_means = set_target(mean, calc_annualized_mean(data, y))
    target_vols = set_target(sd, ((data.reshape((y, time_unit, *s[1:])) + 1).prod(1) - 1).std(1).mean(0))

    assert len(target_vols) == len(target_means)
    num_assets = s[2]

    sol = np.asarray([opt.root(
        fun=_asset_moments,
        x0=np.random.uniform(0, 0.02, 2),
        args=(data[..., i], tv, tm, time_unit)
    ).x for i, tv, tm in zip(range(num_assets), target_vols, target_means)])

    for i in range(num_assets):
        if sol[i, 0] < 0:
            warnings.warn(f'negative vol adjustment at index {i}. This is a cause of concern as a negative vol '
                          f'multiplier will alter the correlation structure')

        data[..., i] = data[..., i] * sol[i, 0] + sol[i, 1]

    return data


def truncate_outliers(data: np.ndarray, sd=0, replacement='mean'):  # pragma: no cover
    """
    Truncates outliers by replacing it with the mean, median or a specified value.

    Outlier is determined by the number of standard deviations past the mean within the asset group.

    Parameters
    ----------
    data: ndarray
        The tensor (data cube) where the axis represents time, trials and number of asset classes respectively

    sd: float
        The number of standard deviations to consider a point an outlier

    replacement: {float, 'mean', 'median'}
        The value to replace outliers with. Valid values are 'mean', 'median' or a number.

    Returns
    -------
    ndarray
        A data cube with the outliers replaced
    """
    assert sd >= 0, "Standard deviations to determine outliers must be >= 0"

    if isinstance(replacement, str):
        assert replacement.lower() in ('mean', 'median'), "replacement can only be 'mean', 'median' or a float value"
    else:
        assert isinstance(replacement, float), "replacement can only be 'mean', 'median' or a float value"
    if sd == 0:
        return data

    cube = np.copy(data)
    for i in range(data.shape[2]):
        returns = cube[..., i]
        mean, std = returns.mean(), returns.std()
        bottom = mean - sd * std
        top = mean + sd * std

        if isinstance(replacement, str):
            r = np.mean(returns) if replacement == 'mean' else np.median(returns)
        else:
            r = float(replacement)

        returns[returns >= top] = r
        returns[returns <= bottom] = r

    return cube


def _asset_moments(x: np.ndarray, asset: np.ndarray, t_vol: float, t_mean: float, time_unit: int):
    """
    Calculates the first 2 asset moments after an adjustment

    Parameters
    ----------
    x: ndarray
        Adjustment quantity

    asset: ndarray
        Initial asset tensor

    t_vol: float
        Target volatility

    t_mean: float
        Target returns

    time_unit: int
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month

    Returns
    -------
    (float, float)
        Volatility and mean
    """

    t, n = asset.shape  # time step (month), trials
    y = t // time_unit

    calibrated = asset * x[0] + x[1]

    d = (calibrated + 1).prod(0)
    mean = (np.sign(d) * np.abs(d) ** (1 / y)).mean() - t_mean - 1
    vol = ((calibrated.reshape((y, time_unit, n)) + 1).prod(1) - 1).std(1).mean(0) - t_vol

    return vol, mean
