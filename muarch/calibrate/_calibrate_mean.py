import numpy as np
import scipy.optimize as opt

from muarch.funcs import get_annualized_mean
from ._calibrate_utils import validate_target_mean


def calibrate_mean_only(data: np.ndarray, mean: np.ndarray, time_unit: int, tol=1e-6):
    validate_target_mean(data, mean)

    sol = [RootFinder(data[..., i], time_unit, tol).find_root(target) for i, target in enumerate(mean)]
    for i, s in enumerate(sol):
        data[..., i] += s

    return data


class RootFinder:
    def __init__(self, data: np.ndarray, time_unit: int, tol=1e-6):
        assert data.ndim == 2
        self.data = data
        self.time_unit = time_unit
        self.tol = tol

    def annualized_mean(self, x: float, target: float):
        return get_annualized_mean(self.data + x) - target

    def annualized_mean_der(self, x: float, target: float):
        return (self.annualized_mean(x + self.tol, target) -
                self.annualized_mean(x - self.tol, target)) / (2 * self.tol)

    def find_root(self, target: float) -> float:
        if self.is_similar(target):
            return 0

        return opt.newton(self.annualized_mean, self.initial_guess(target), self.annualized_mean_der,
                          args=(target,), tol=self.tol, maxiter=1000)

    def initial_guess(self, target: float):
        def get_by_bisection():
            index = np.argmin(np.abs(f_space[:-1] - f_space[1:])[mask])  # index with best root character
            a, b = space[:-1][mask][index], space[1:][mask][index]
            return (a + b) / 2

        def get_closest_to_0():
            return float(space[np.argmin(np.abs(f_space))])

        space = 2 ** np.linspace(-15, 4, 30)
        space = np.sort([*-space, *space])

        f_space = np.array([self.annualized_mean(x, target) for x in space])
        mask: np.ndarray = (f_space[:-1] * f_space[1:]) <= 0

        if any(mask):
            return get_by_bisection()
        else:
            return get_closest_to_0()

    def is_similar(self, target: float):
        return np.isclose(self.annualized_mean(0, target), 0, atol=self.tol)
