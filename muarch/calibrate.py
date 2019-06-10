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
    assert not np.isnan(data).any(), "data cube must not have nan values"

    y, n, num_assets = data.shape
    y //= time_unit

    def set_target(target_values: Optional[Iterable[float]], typ: str, default: np.ndarray):
        if target_values is None:
            return default, [0 if typ == 'mean' else 1] * num_assets

        best_guess = []
        target_values = np.asarray(target_values)
        assert num_assets == len(target_values) == len(
            default), "vector length must be equal to number of assets in data cube"
        for i, v in enumerate(target_values):
            if v is None:
                target_values[i] = default[i]
                best_guess.append(0 if typ == 'mean' else 1)
            else:
                best_guess.append(target_values[i] - default[i] if typ == 'mean' else default[i] / target_values[i])
        return target_values, best_guess

    d = (data + 1).prod(0)
    default_mean = (np.sign(d) * np.abs(d) ** (1 / y)).mean(0) - 1
    default_vol = ((data + 1).reshape(y, time_unit, n, num_assets).prod(1) - 1).std(1).mean(0)

    target_means, guess_mean = set_target(mean, 'mean', default_mean)
    target_vols, guess_vol = set_target(sd, 'vol', default_vol)

    sol = np.asarray([opt.root(
        fun=_asset_moments,
        x0=[gv, gm],
        args=(data[..., i], tv, tm, time_unit)
    ).x for i, tv, tm, gv, gm in zip(range(num_assets), target_vols, target_means, guess_vol, guess_mean)])

    for i in range(num_assets):
        if sol[i, 0] < 0 or np.isnan(sol[i]).any():
            # negative vol adjustments would alter the correlation between asset classes (flip signs)
            # in such instances, we fall back to using the a simple affine transform where
            # R' = (tv/cv) * r  # adjust vol first
            # R' = (tm - mean(R'))  # adjust mean

            # adjust vol
            cv = default_vol[i]
            tv = sd[i] if sd[i] is not None else tv
            data[..., i] *= tv / cv  # tv / cv

            # adjust mean
            tm, cm = target_means[i], data[..., i].mean()
            data[..., i] += (tm - cm)  # (tm - mean(R'))
        else:
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
