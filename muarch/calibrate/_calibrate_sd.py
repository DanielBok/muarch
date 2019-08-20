import numpy as np
import scipy.optimize as opt
from muarch.funcs import get_annualized_sd

from ._calibrate_utils import get_data_shape, validate_target_sd


def calibrate_sd_only(data: np.ndarray, sd: np.ndarray, time_unit: int, tol=1e-6):
    validate_target_sd(data, sd)
    curr_sd = get_annualized_sd(data, time_unit)

    sol = [RootFinder(data[..., i], time_unit, tol).find_root(target) for i, target in enumerate(sd)]
    for i, s in enumerate(sol):
        if np.isfinite(s) and s > 0:
            data[..., i] *= s
        else:
            data[..., i] *= sd[i] / curr_sd[i]

    return data


class RootFinder:
    def __init__(self, data: np.ndarray, time_unit: int, tol=1e-6):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.years, self.trials = get_data_shape(data, time_unit)
        self.tol = tol

    def annualized_sd(self, x: float, target: float):
        shape = self.years, self.time_unit, self.data.shape[-1]
        return ((self.data * x + 1).reshape(shape).prod(1) - 1).std(1).mean() - target

    def annualized_sd_der(self, x: float, target: float):
        return (self.annualized_sd(x + self.tol, target) -
                self.annualized_sd(x - self.tol, target)) / (2 * self.tol)

    def find_root(self, target: float) -> float:
        if self.is_similar(target):
            return 0

        return opt.newton(self.annualized_sd, self.initial_guess(target), self.annualized_sd_der,
                          args=(target,), tol=self.tol, maxiter=1000)

    def initial_guess(self, target: float):
        def get_by_bisection():
            index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
            a, b = space[:-1][mask][index], space[1:][mask][index]
            return (a + b) / 2

        def get_closest_to_0():
            return float(space[np.argmin(np.abs(f_space))])

        space = 2 ** np.linspace(-15, 6, 30)
        f_space = np.array([self.annualized_sd(x, target) for x in space])
        mask: np.ndarray = (f_space[:-1] * f_space[1:]) <= 0

        if any(mask):
            return get_by_bisection()
        else:
            return get_closest_to_0()

    def is_similar(self, target: float):
        return np.isclose(self.annualized_sd(0, target), 0, atol=self.tol)
