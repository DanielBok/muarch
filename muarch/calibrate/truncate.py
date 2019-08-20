from copy import deepcopy

import numpy as np


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
