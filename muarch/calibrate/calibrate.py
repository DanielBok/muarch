from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np

from ._calibrate_both import calibrate_mean_and_sd
from ._calibrate_mean import calibrate_mean_only
from ._calibrate_sd import calibrate_sd_only


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
        return calibrate_mean_and_sd(data, np.asarray(mean), np.asarray(sd), time_unit)

    if mean is not None and sd is None:
        return calibrate_mean_only(data, np.asarray(mean), time_unit)

    if mean is None and sd is not None:
        return calibrate_sd_only(data, np.asarray(sd), time_unit)

    return data  # no adjustments


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
