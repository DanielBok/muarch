import numpy as np
import scipy.optimize as opt

from ._calibrate_utils import get_data_shape, validate_target_mean


def calibrate_mean_only(data: np.ndarray, mean: np.ndarray, time_unit: int):
    validate_target_mean(data, mean)

    sol = get_solutions(data, time_unit, mean)
    for i, s in enumerate(sol):
        data[..., i] += s

    return data


def get_solutions(data: np.ndarray, time_unit: int, mean: np.ndarray):
    asset_list = (AssetMean(data[..., i], time_unit, target) for i, target in enumerate(mean))

    return [f() for f in asset_list]


class AssetMean:
    def __init__(self, data: np.ndarray, time_unit, target_mean: float):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.years, _ = get_data_shape(data, time_unit)
        self.target = target_mean
        self.best_guess = self.calc_best_guess()

    def __call__(self) -> float:
        a, b = self.best_guess
        return opt.toms748(self.annualized_mean, a, b)

    def annualized_mean(self, x):
        d = (self.data + x + 1).prod(0)
        return (np.sign(d) * np.abs(d) ** (1 / self.years)).mean() - 1 - self.target

    def calc_best_guess(self):
        space = 2 ** np.linspace(-15, 4, 20)
        space = np.sort([*-space, *space])

        f_space = np.array([self.annualized_mean(x) for x in space])
        mask = (f_space[:-1] * f_space[1:]) <= 0

        if not np.any(mask):
            raise RuntimeError("Unable to find roots")

        index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
        return space[:-1][mask][index], space[1:][mask][index]
