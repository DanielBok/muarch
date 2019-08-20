from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np

from muarch.funcs.moments import get_annualized_mean, get_annualized_sd
from muarch.funcs.time_unit import get_integer_time_unit
from ._calibrate_both import calibrate_mean_and_sd
from ._calibrate_mean import calibrate_mean_only
from ._calibrate_sd import calibrate_sd_only


def calibrate_data(data: np.ndarray, mean: Optional[Iterable[float]] = None, sd: Optional[Iterable[float]] = None,
                   time_unit: Union[int, str] = "month", inplace=False, tol=1e-6) -> np.ndarray:
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

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    inplace: bool
        If True, calibration will modify the original data. Otherwise, a deep copy of the original data will be
        made before calibration. Deep copy can be time consuming if data is big.

    tol: float
        Tolerance used to determine if calibrate should be called. For example, if the cube's target annualized
        mean is similar to the actual tolerance, function will skip the mean adjustment.

    Returns
    -------
    ndarray
        An instance of the adjusted numpy tensor
    """
    assert not np.isnan(data).any(), "data cube must not have nan values"
    time_unit = get_integer_time_unit(time_unit)

    if not inplace:
        data = deepcopy(data)

    mean = _set_to_none_if_close(mean, get_annualized_mean(data, time_unit), tol)
    sd = _set_to_none_if_close(sd, get_annualized_sd(data, time_unit), tol)

    return _calibrate(data, mean, sd, time_unit, tol)


def _set_to_none_if_close(actual: Optional[np.ndarray], target: np.ndarray, tol):
    return None if actual is None or np.isclose(actual, target, atol=tol).all() else np.asarray(actual)


def _calibrate(data: np.ndarray, mean: Optional[np.ndarray], sd: Optional[np.ndarray], time_unit: int, tol: float):
    if mean is not None and sd is not None:
        return calibrate_mean_and_sd(data, np.asarray(mean), np.asarray(sd), time_unit)

    if mean is not None and sd is None:
        return calibrate_mean_only(data, np.asarray(mean), time_unit, tol)

    if mean is None and sd is not None:
        return calibrate_sd_only(data, np.asarray(sd), time_unit)

    return data  # no adjustments
