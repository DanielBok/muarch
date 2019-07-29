from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import scipy.optimize as opt

__all__ = ['calibrate_data', 'truncate_outliers']


def calibrate_data(data: np.ndarray, mean: Optional[Iterable[float]] = None, sd: Optional[Iterable[float]] = None,
                   time_unit: Union[int, str] = 12, inplace=False) -> np.ndarray:
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
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    inplace: bool
        If True, calibration will modify the original data. Otherwise, a deep copy of the original data will be
        made before calibration. Deep copy can be time consuming if data is big.

    Returns
    -------
    ndarray
        An instance of the adjusted numpy tensor
    """
    assert not np.isnan(data).any(), "data cube must not have nan values"
    time_unit = _get_integer_time_unit(time_unit)

    if not inplace:
        data = deepcopy(data)

    if mean is not None and sd is not None:
        return _calibrate_mean_and_sd(data, np.asarray(mean), np.asarray(sd), time_unit)

    if mean is not None and sd is None:
        return _calibrate_mean_only(data, np.asarray(mean), time_unit)

    if mean is None and sd is not None:
        return _calibrate_sd_only(data, np.asarray(sd), time_unit)

    return data  # no adjustments


def truncate_outliers(data: np.ndarray, sd=0, replacement='mean', inplace=False):  # pragma: no cover
    """
    Truncates outliers by replacing it with the mean, median or a specified value.

    Outlier is determined by the number of standard deviations past the mean within the asset group.

    Parameters
    ----------
    data: ndarray
        The tensor (data cube) where the axis represents time, trials and number of asset classes respectively

    sd: float
        The number of standard deviations to consider a point an outlier. If :code:`sd` is set to 0, no changes will
        be made

    replacement: {float, 'mean', 'median'}
        The value to replace outliers with. Valid values are 'mean', 'median' or a number.

    inplace: bool
        If True, calibration will modify the original data. Otherwise, a deep copy of the original data will be
        made before calibration. Deep copy can be time consuming if data is big.

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

    if not inplace:
        data = deepcopy(data)

    if sd == 0:
        return data

    def get_replacement(returns_matrix):
        if replacement == "mean":
            return np.mean(returns_matrix)
        elif replacement == "median":
            return np.median(returns_matrix)
        else:
            return float(replacement)

    for i in range(data.shape[2]):
        returns = data[..., i]
        mean, std = returns.mean(), returns.std()
        r = get_replacement(returns)

        returns[returns > (mean + sd * std)] = r
        returns[returns < (mean - sd * std)] = r

    return data


def _annualized_returns_per_asset(data: np.ndarray, time_unit: int):
    years, *_ = _get_data_shape(data, time_unit)
    d = (data + 1).prod(0)
    return (np.sign(d) * np.abs(d) ** (1 / years)).mean(0) - 1


def _annualized_returns_for_1_asset(data: np.ndarray, time_unit: int):
    years, *_ = _get_data_shape(data, time_unit)
    d = (data + 1).prod(0)
    return (np.sign(d) * np.abs(d) ** (1 / years)).mean() - 1


def _annualized_sd_per_asset(data: np.ndarray, time_unit: int):
    years, trials, num_assets = _get_data_shape(data, time_unit)
    return ((data + 1).reshape(years, time_unit, trials, num_assets).prod(1) - 1).std(1).mean(0)


def _annualized_sd_for_1_asset(data: np.ndarray, time_unit: int) -> float:
    years, trials = _get_data_shape(data, time_unit)
    return ((data + 1).reshape(years, time_unit, trials).prod(1) - 1).std(1).mean()


def _best_guess_scalar(f, non_neg=False):
    def make_space():
        if non_neg:
            return 2 ** np.linspace(-15, 6, 150)
        a = 2 ** np.linspace(-15, 4, 75)
        return np.sort([*-a, *a])

    space = make_space()
    f_space = np.frompyfunc(f, 1, 1)(space)

    best, best_diff = None, np.inf
    for i, j, x, y in zip(space[:-1], space[1:], f_space[:-1], f_space[1:]):
        if x * y <= 0 and abs(x - y) < best_diff:
            best_diff = abs(x - y)
            best = i, j

    if best is None:
        raise RuntimeError("Unable to find roots")

    return best


def _calculate_scalar_solutions(f, non_neg=False):
    a, b = _best_guess_scalar(f, non_neg)
    return opt.bisect(f, a, b)


def _calibrate_mean_only(data: np.ndarray, mean: np.ndarray, time_unit: int):
    _validate_target_mean(data, mean)

    def _asset_mean(x: float, asset_returns: np.ndarray, target_mean: float):
        return _annualized_returns_for_1_asset(asset_returns + x, time_unit) - target_mean

    sol = np.array([_calculate_scalar_solutions(lambda x: _asset_mean(x, data[..., i], target))
                    for i, target in enumerate(mean)])

    for i, s in enumerate(sol):
        data[..., i] += s

    return data


def _calibrate_sd_only(data: np.ndarray, sd: np.ndarray, time_unit: int):
    _validate_target_sd(data, sd)

    def _asset_sd(x: float, asset_returns: np.ndarray, target_vol: float):
        return _annualized_sd_for_1_asset(asset_returns * x, time_unit) - target_vol

    sol = np.array([_calculate_scalar_solutions(lambda x: _asset_sd(x, data[..., i], target), non_neg=True)
                    for i, target in enumerate(sd)])

    current_vol = _annualized_sd_per_asset(data, time_unit)
    for i, s in enumerate(sol):
        if np.isfinite(s) and s > 0:
            data[..., i] *= s
        else:
            data[..., i] *= sd[i] / current_vol[i]

    return data


def _calibrate_mean_and_sd(data: np.ndarray, mean: np.ndarray, sd: np.ndarray, time_unit: int):
    _validate_target_mean(data, mean)
    _validate_target_sd(data, sd)

    years, _, num_assets = _get_data_shape(data, time_unit)

    current_mean = _annualized_returns_per_asset(data, time_unit)
    current_vol = _annualized_sd_per_asset(data, time_unit)

    def _asset_moments(x: np.ndarray, asset: np.ndarray, target_vol: float, target_mean: float):
        calibrated = asset * x[0] + x[1]

        return (
            _annualized_sd_for_1_asset(calibrated, time_unit) - target_vol,
            _annualized_returns_for_1_asset(calibrated, time_unit) - target_mean
        )

    def _calculate_solutions(asset_returns: np.ndarray, target_vol: float, target_mean: float):
        def get_initial_point(f, non_neg: bool):
            x, y = _best_guess_scalar(f, non_neg=non_neg)
            return (x + y) / 2

        initial_vol = get_initial_point(
            lambda x: _annualized_sd_for_1_asset(asset_returns * x, time_unit) - target_vol,
            non_neg=True
        )
        initial_mean = get_initial_point(
            lambda x: _annualized_returns_for_1_asset(asset_returns + x, time_unit) - target_mean,
            non_neg=False
        )

        return opt.root(
            _asset_moments,
            x0=np.array([initial_vol, initial_mean]),
            args=(asset_returns, target_vol, target_mean)
        ).x

    sol = np.array([_calculate_solutions(data[..., i], v, m) for i, v, m in zip(range(num_assets), sd, mean)])

    for i in range(num_assets):
        if sol[i, 0] < 0 or np.isnan(sol[i]).any():
            # negative vol adjustments would alter the correlation between asset classes (flip signs)
            # in such instances, we fall back to using the a simple affine transform where
            # R' = (tv / cv) * R + (tm - cm)

            data[..., i] = data[..., i] * (sd[i] / current_vol[i]) + (current_mean[i] - mean[i])
        else:
            data[..., i] = data[..., i] * sol[i, 0] + sol[i, 1]

    return data


def _get_data_shape(data: np.ndarray, time_unit: int):
    def get_years(periods):
        years = periods / time_unit
        assert years.is_integer(), "time_unit must divide data periods to an integer year"
        return int(years)

    def get_3d_shape() -> Tuple[int, int, int]:
        periods, trials, num_assets = data.shape
        return get_years(periods), trials, num_assets

    def get_2d_shape() -> Tuple[int, int]:
        periods, trials = data.shape
        return get_years(periods), trials

    if data.ndim == 2:
        return get_2d_shape()
    else:
        return get_3d_shape()


def _get_integer_time_unit(time_unit: Union[int, str]):
    if isinstance(time_unit, int):
        return time_unit
    if isinstance(time_unit, str):
        if time_unit.lower() in ('week', 'weekly'):
            return 52
        if time_unit.lower() in ('month', 'monthly'):
            return 12
        if time_unit.lower() in ('quarter', 'quarterly'):
            return 4
        if time_unit.lower() in ('semi-annual', 'semi-annually'):
            return 2
        if time_unit.lower() in ('annual', 'annually', 'year', 'yearly'):
            return 1

    raise ValueError(f"Unacceptable time_unit argument, {time_unit}")


def _validate_target_mean(data: np.ndarray, mean: np.ndarray):
    assert len(mean) == data.shape[2], "length of target returns (mean) must equal number of assets"
    assert np.isfinite(np.asarray(mean)).all(), "all target returns (mean) must be finite"


def _validate_target_sd(data: np.ndarray, sd: np.ndarray):
    _, _, num_assets = data.shape
    assert num_assets == len(sd), "length of target vols (sd) must equal number of assets"
    assert np.isfinite(np.asarray(sd)).all(), "all target vols (sd) must be finite"
    assert (np.asarray(sd) >= 0).all(), "target vol must all be >= 0"
