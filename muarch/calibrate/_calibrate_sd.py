import numpy as np
import scipy.optimize as opt

from ._calibrate_utils import calc_current_sd, get_data_shape, validate_target_sd


def calibrate_sd_only(data: np.ndarray, sd: np.ndarray, time_unit: int):
    validate_target_sd(data, sd)
    curr_sd = calc_current_sd(data, time_unit)

    sol = get_solutions(data, time_unit, sd)
    for i, s in enumerate(sol):
        if np.isfinite(s) and s > 0:
            data[..., i] *= s
        else:
            data[..., i] *= sd[i] / curr_sd[i]

    return data


def get_solutions(data: np.ndarray, time_unit: int, sd: np.ndarray):
    asset_mean_list = (AssetSD(data[..., i], time_unit, target) for i, target in enumerate(sd))

    return [f() for f in asset_mean_list]


class AssetSD:
    def __init__(self, data: np.ndarray, time_unit, target_sd: float):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.years, self.trials = get_data_shape(data, time_unit)
        self.target = target_sd
        self.best_guess = self.calc_best_guess()

    def __call__(self) -> float:
        a, b = self.best_guess
        return opt.toms748(self.annualized_sd, a, b)

    def annualized_sd(self, x):
        shape = self.years, self.time_unit, self.trials
        return ((self.data * x + 1).reshape(shape).prod(1) - 1).std(1).mean() - self.target

    def calc_best_guess(self):
        space = 2 ** np.linspace(-15, 6, 150)
        f_space = np.array([self.annualized_sd(x) for x in space])
        mask = (f_space[:-1] * f_space[1:]) <= 0

        if not np.any(mask):
            raise RuntimeError("Unable to find roots")

        index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
        return space[:-1][mask][index], space[1:][mask][index]


def _call(f):
    return f()
