from typing import Tuple

import numpy as np

__all__ = ["calc_current_sd", "get_data_shape", "validate_target_mean", "validate_target_sd"]


def calc_current_sd(data: np.ndarray, time_unit: int):
    years, trials, num_assets = get_data_shape(data, time_unit)
    return ((data + 1).reshape(years, time_unit, trials, num_assets).prod(1) - 1).std(1).mean(0)


def get_data_shape(data: np.ndarray, time_unit: int):
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


def validate_target_mean(data: np.ndarray, mean: np.ndarray):
    assert len(mean) == data.shape[2], "length of target returns (mean) must equal number of assets"
    assert np.isfinite(np.asarray(mean)).all(), "all target returns (mean) must be finite"


def validate_target_sd(data: np.ndarray, sd: np.ndarray):
    _, _, num_assets = data.shape
    assert num_assets == len(sd), "length of target vols (sd) must equal number of assets"
    assert np.isfinite(np.asarray(sd)).all(), "all target vols (sd) must be finite"
    assert (np.asarray(sd) >= 0).all(), "target vol must all be >= 0"
