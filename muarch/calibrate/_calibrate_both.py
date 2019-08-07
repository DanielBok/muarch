import numpy as np
import scipy.optimize as opt

from ._calibrate_utils import get_data_shape, validate_target_mean, validate_target_sd


def calibrate_mean_and_sd(data: np.ndarray, mean: np.ndarray, sd: np.ndarray, time_unit: int):
    validate_target_mean(data, mean)
    validate_target_sd(data, sd)

    sol = get_solutions(data, time_unit, mean, sd)

    for i in range(data.shape[2]):
        if sol[i, 0] > 0 and np.isfinite(sol[i]).all():
            data[..., i] = data[..., i] * sol[i, 1] + sol[i, 0]

    return data


def get_solutions(data: np.ndarray, time_unit: int, mean: np.ndarray, sd: np.ndarray):
    asset_list = (AssetMeanAndSD(data[..., i], time_unit, m, s) for i, (m, s) in enumerate(zip(mean, sd)))

    return np.array([f() for f in asset_list])


class AssetMeanAndSD:
    def __init__(self, data: np.ndarray, time_unit, target_mean: float, target_sd: float):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.years, self.trials = get_data_shape(data, time_unit)
        self.target_mean = target_mean
        self.target_sd = target_sd

        self.best_guess = np.array([self.calc_best_guess_mean(), self.calc_best_guess_sd()])

    def __call__(self) -> float:
        return opt.root(self.annualized_moments, x0=self.best_guess).x

    def annualized_moments(self, x: np.ndarray):
        calibrated_data = self.data * x[1] + x[0]
        return self.annualized_mean(calibrated_data), self.annualized_sd(calibrated_data)

    def annualized_mean(self, data) -> float:
        d = (data + 1).prod(0)
        return (np.sign(d) * np.abs(d) ** (1 / self.years)).mean() - 1 - self.target_mean

    def annualized_sd(self, data) -> float:
        return ((data + 1).reshape(self.years, self.time_unit, self.trials).prod(1) - 1).std(1).mean() - self.target_sd

    def calc_best_guess_mean(self):
        space = 2 ** np.linspace(-15, 4, 20)
        space = np.sort([*-space, *space])

        f_space = np.array([self.annualized_mean(self.data + x) for x in space])
        mask = (f_space[:-1] * f_space[1:]) <= 0

        if not np.any(mask):
            raise RuntimeError("Unable to find roots")

        index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
        return (space[:-1][mask][index] + space[1:][mask][index]) / 2

    def calc_best_guess_sd(self):
        space = 2 ** np.linspace(-15, 6, 30)

        f_space = np.array([self.annualized_sd(self.data * x) for x in space])
        mask = (f_space[:-1] * f_space[1:]) <= 0

        if not np.any(mask):
            raise RuntimeError("Unable to find roots")

        index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
        return (space[:-1][mask][index] + space[1:][mask][index]) / 2


def _call(f):
    return f()
