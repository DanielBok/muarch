from copy import deepcopy
from typing import List, Tuple

import numpy as np


def truncate_outliers(data: np.ndarray, *,
                      bounds: List[Tuple[float, float]] = None,
                      sd=0,
                      replacement='mean',
                      inplace=False):  # pragma: no cover
    """
    Truncates outliers by replacing it with the mean, median or a specified value.

    Outlier is determined by the number of standard deviations past the mean within the asset group.

    Parameters
    ----------
    data: ndarray
        The tensor (data cube) where the axis represents time, trials and number of asset classes respectively

    bounds: List of numbers
        A list containing the lower and upper bound for each asset class. If specified, this takes precedence over
        the :code:`sd` parameter. If :code:`sd` is set to 0 and bounds are not specified, no changes will be made

    sd: float
        The number of standard deviations to consider a point an outlier. If :code:`sd` is set to 0 and bounds are not
        specified, no changes will be made

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
    num_assets = data.shape[2]
    if bounds is None and sd == 0:
        return data

    if bounds is None:
        bounds = _form_bounds(data, sd)
    bounds = _sort_bounds(bounds)

    replacement = replacement.lower()

    _validate_data_cube(data)
    _validate_replacement(replacement)
    _validate_bounds(bounds, num_assets)

    if not inplace:
        data = deepcopy(data)

    replacement_values = _get_replacement_values(replacement, bounds=bounds, data=data)

    for i, (lb, ub), r in zip(range(num_assets), bounds, replacement_values):
        returns = data[..., i]

        returns[returns > ub] = r
        returns[returns < lb] = r

    return data


def _form_bounds(data: np.ndarray, sd: float):
    assert sd >= 0, "Standard deviations to determine outliers must be >= 0"

    bounds = []
    for i in range(data.shape[2]):
        returns = data[..., i]
        mean, std = returns.mean(), returns.std()
        bounds.append((mean - sd * std, mean + sd * std))

    return bounds


def _sort_bounds(bounds: List[Tuple[float, float]]):
    return [(min(b), max(b)) for b in bounds]


def _get_replacement_values(replacement: str, *, bounds=None, data: np.ndarray) -> List[float]:
    num_assets = data.shape[2]
    if replacement == "mean":
        return ([np.mean(b) for b in bounds]
                if bounds is not None else
                [data[..., n].mean() for n in range(num_assets)])
    elif replacement == "median":
        return [np.median(data[..., n]) for n in range(num_assets)]
    else:
        return [float(replacement)] * num_assets


def _validate_bounds(bounds: List[Tuple[float, float]], num_assets: int):
    assert len(bounds) == num_assets, "Number of bound ranges do not match number of assets"


def _validate_data_cube(data: np.ndarray):
    assert data.ndim == 3, "data must be a 3D tensor"


def _validate_replacement(replacement: str):
    if isinstance(replacement, str):
        assert replacement.lower() in ('mean', 'median'), \
            "replacement can only be 'mean', 'median' or a float value"
    else:
        assert isinstance(replacement, float), "replacement can only be 'mean', 'median' or a float value"
