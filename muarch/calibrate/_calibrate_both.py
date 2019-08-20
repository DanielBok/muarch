import numpy as np
import scipy.optimize as opt

from muarch.funcs import get_annualized_mean, get_annualized_sd
from ._calibrate_utils import validate_target_mean, validate_target_sd


def calibrate_mean_and_sd(data: np.ndarray, mean: np.ndarray, sd: np.ndarray, time_unit: int, tol=1e-6):
    validate_target_mean(data, mean)
    validate_target_sd(data, sd)

    sol = [RootFinder(data[..., i], time_unit, tol).find_roots(m, s) for i, (m, s) in enumerate(zip(mean, sd))]

    for i, (m, s) in enumerate(sol):
        if s > 0 and np.isfinite([m, s]).all():
            data[..., i] = data[..., i] * s + m

    return data


class RootFinder:
    def __init__(self, data: np.ndarray, time_unit: int, tol=1e-6):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.tol = tol

    def annualized_moments(self, x: np.ndarray, mean: float, sd: float):
        calibrated_data = self.data * x[1] + x[0]
        return (get_annualized_mean(calibrated_data, self.time_unit) - mean,
                get_annualized_sd(calibrated_data, self.time_unit) - sd)

    def find_roots(self, mean: float, sd: float) -> np.ndarray:
        if self.is_similar(mean, sd):
            return np.array([0, 1])

        return opt.root(self.annualized_moments, self.initial_guess(mean, sd), args=(mean, sd)).x

    def initial_guess(self, mean: float, sd: float):
        def get_by_bisection(space: np.ndarray, f_space: np.ndarray, mask: np.ndarray):
            i = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
            return (space[:-1][mask][i] + space[1:][mask][i]) / 2

        def get_closest_to_0(space: np.ndarray, f_space: np.ndarray):
            return float(space[np.argmin(np.abs(f_space))])

        def mean_best_guess():
            space = 2 ** np.linspace(-15, 4, 30)
            space = np.sort([*-space, *space])
            f_space = np.array([get_annualized_mean(self.data + x, self.time_unit) - mean for x in space])
            mask: np.ndarray = (f_space[:-1] * f_space[1:]) <= 0

            return get_by_bisection(space, f_space, mask) if any(mask) else get_closest_to_0(space, f_space)

        def sd_best_guess():
            space = 2 ** np.linspace(-15, 6, 30)
            f_space = np.array([get_annualized_sd(self.data * x, self.time_unit) - sd for x in space])
            mask: np.ndarray = (f_space[:-1] * f_space[1:]) <= 0

            return get_by_bisection(space, f_space, mask) if any(mask) else get_closest_to_0(space, f_space)

        return np.array([mean_best_guess(), sd_best_guess()])

    def is_similar(self, mean: float, sd: float):
        return all(np.isclose(self.annualized_moments(np.array([0, 1]), mean, sd), [mean, sd], atol=self.tol))
