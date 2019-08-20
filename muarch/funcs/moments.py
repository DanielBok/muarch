from typing import Union

import numpy as np
from scipy.stats import kurtosis, skew

from .time_unit import get_integer_time_unit

__all__ = ["get_annualized_kurtosis", "get_annualized_mean", "get_annualized_sd", "get_annualized_skew"]


def get_annualized_kurtosis(data: np.ndarray, time_unit: Union[int, str] = 'monthly') -> Union[float, np.ndarray]:
    """
    Gets the annualized kurtosis for each asset class in the data cube

    Parameters
    ----------
    data
        Data matrix or tensor. The axis must represent time, trials and assets respectively where the assets axis
        is valid only if the data is a tensor.

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    Returns
    -------
    ndarray
        The annualized kurtosis for the asset class or an array containing the annualized kurtosis for each
         asset class.
    """
    return np.mean([kurtosis(i) for i in annualize_data(data, time_unit)], 0)


def get_annualized_mean(data: np.ndarray, time_unit: Union[int, str] = 'monthly') -> Union[float, np.ndarray]:
    """
    Gets the annualized mean for each asset class in the data cube

    Parameters
    ----------
    data
        Data matrix or tensor. The axis must represent time, trials and assets respectively where the assets axis
        is valid only if the data is a tensor.

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    Returns
    -------
    float or ndarray
        The annualized mean for the asset class or an array containing the annualized mean for each asset class.
    """
    time_unit = get_integer_time_unit(time_unit)
    years = len(data) // time_unit

    data = (annualize_data(data, time_unit) + 1).prod(0)
    return np.mean(np.sign(data) * np.abs(data) ** (1 / years), 0) - 1


def get_annualized_sd(data: np.ndarray, time_unit: Union[int, str] = 'monthly') -> Union[float, np.ndarray]:
    """
    Gets the annualized standard deviation for each asset class in the data cube

    Parameters
    ----------
    data
        Data matrix or tensor. The axis must represent time, trials and assets respectively where the assets axis
        is valid only if the data is a tensor.

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    Returns
    -------
    float ndarray
        The annualized standard deviation (volatility) for the asset class or an array containing the annualized
        standard deviation for each asset class.
    """
    return annualize_data(data, time_unit).std(1).mean(0)


def get_annualized_skew(data: np.ndarray, time_unit: Union[int, str] = 'monthly') -> Union[float, np.ndarray]:
    """
    Gets the annualized skew for each asset class in the data cube

    Parameters
    ----------
    data
        Data matrix or tensor. The axis must represent time, trials and assets respectively where the assets axis
        is valid only if the data is a tensor.

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    Returns
    -------
    float or ndarray
        The annualized skew for the asset class or an array containing the annualized skew for each asset class.
    """
    return np.mean([skew(i) for i in annualize_data(data, time_unit)], 0)


def annualize_data(data: np.ndarray, time_unit: Union[str, int]) -> np.ndarray:
    """
    Annualizes the data. Note that the dimension of the data will change from this transformation

    Parameters
    ----------
    data
        Data matrix or tensor. The axis must represent time, trials and assets respectively where the assets axis
        is valid only if the data is a tensor.

    time_unit: int or str
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
        a month. Alternatively, you could put in a string name of the time_unit. Accepted values are weekly,
        monthly, quarterly, semi-annually and yearly

    Returns
    -------
    ndarray
        The annualized data.
    """

    data = np.asarray(data)
    time_unit = get_integer_time_unit(time_unit)

    assert data.ndim in (2, 3), "data must be 2 or 3 dimensional"
    if data.ndim == 2:
        t, n = data.shape
        return (data.reshape((t // time_unit, time_unit, n)) + 1).prod(1) - 1
    else:
        t, n, a = data.shape
        return (data.reshape((t // time_unit, time_unit, n, a)) + 1).prod(1) - 1
