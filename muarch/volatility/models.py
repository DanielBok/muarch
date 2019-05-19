from warnings import warn

import numpy as np
from arch.univariate.volatility import (
    ARCH as A,
    ConstantVariance as C,
    EGARCH as E,
    FIGARCH as F,
    GARCH as G,
    HARCH as H
)
from arch.utility.exceptions import InitialValueWarning, initial_value_warning

from muarch.volatility import _vol_simulations as vs
from .simulators import RNG, garch_simulator, garch_simulator_mc

try:
    from arch.univariate.recursions import figarch_weights
except ImportError:
    from arch.univariate.recursions_python import figarch_weights


class ARCH(A):
    @garch_simulator
    def simulate(self, parameters: np.ndarray, nobs: int, rng: RNG, burn=500, initial_value=None):
        pass

    @garch_simulator_mc
    def simulate_mc(self, parameters: np.ndarray, nobs: int, reps: int, rng: RNG, burn=500, initial_value=None):
        pass


class ConstantVariance(C):
    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        errors = rng(nobs + burn)
        sigma2 = np.ones(nobs + burn) * parameters[0]
        data = np.sqrt(sigma2) * errors
        return data[burn:], sigma2[burn:]

    def simulate_mc(self, parameters: np.ndarray, nobs: int, reps: int, rng: RNG, burn=500, initial_value=None):
        errors = rng(nobs + burn, reps)
        sigma2 = np.ones((nobs + burn, reps)) * parameters[0]
        data = np.sqrt(sigma2) * errors
        return data[burn:]


class EGARCH(E):
    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q = self.p, self.o, self.q
        errors = rng(nobs + burn)
        max_lag = max(p, o, q)

        sigma2, data, lnsigma2 = self._simulation_setup(nobs + burn, parameters, initial_value, errors)

        return vs.egarch_simulate(p, o, q, nobs, burn, max_lag, parameters, data, sigma2, lnsigma2, errors)

    def simulate_mc(self, parameters: np.ndarray, nobs: int, reps: int, rng: RNG, burn=500, initial_value=None):
        p, o, q = self.p, self.o, self.q
        errors = rng(nobs + burn, reps)

        sigma2, data, lnsigma2 = self._simulation_setup((nobs + burn, reps), parameters, initial_value, errors)

        return vs.egarch_simulate_mc(p, o, q, nobs, burn, reps, parameters, sigma2, lnsigma2, errors)

    def _simulation_setup(self, shape, parameters, initial_value, errors):
        p, o, q = self.p, self.o, self.q
        max_lag = max(p, o, q)

        if initial_value is None:
            if q > 0:
                beta_sum = np.sum(parameters[p + o + 1:])
            else:
                beta_sum = 0.0

            if beta_sum < 1:
                initial_value = parameters[0] / (1.0 - beta_sum)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(shape)
        data = np.zeros(shape)
        lnsigma2 = np.zeros(shape)

        lnsigma2[:max_lag] = initial_value
        sigma2[:max_lag] = np.exp(initial_value)
        data[:max_lag] = errors[:max_lag] * np.sqrt(sigma2[:max_lag])

        return sigma2, data, lnsigma2


class FIGARCH(F):
    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        truncation = self.truncation
        p, q, power = self.p, self.q, self.power
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        errors = rng(truncation + nobs + burn)

        if initial_value is None:
            persistence = np.sum(lam)
            beta = parameters[-1] if q else 0.0

            initial_value = parameters[0]
            if beta < 1:
                initial_value /= (1 - beta)
            if persistence < 1:
                initial_value /= (1 - persistence)
            if persistence >= 1.0 or beta >= 1.0:
                warn(initial_value_warning, InitialValueWarning)

        sigma2 = np.empty(truncation + nobs + burn)
        data = np.empty(truncation + nobs + burn)
        fsigma = np.empty(truncation + nobs + burn)
        fdata = np.empty(truncation + nobs + burn)

        fsigma[:truncation] = initial_value
        sigma2[:truncation] = initial_value ** (2.0 / power)
        data[:truncation] = np.sqrt(sigma2[:truncation]) * errors[:truncation]
        fdata[:truncation] = abs(data[:truncation]) ** power
        omega = parameters[0]
        beta = parameters[-1] if q else 0
        omega_tilde = omega / (1 - beta)

        return vs.figarch_simulate(nobs, burn, truncation, power, omega_tilde, data, fdata, fsigma, sigma2, lam_rev,
                                   errors)

    def simulate_mc(self, parameters: np.ndarray, reps: int, nobs: int, rng: RNG, burn=500, initial_value=None):
        raise NotImplementedError


class GARCH(G):
    @garch_simulator
    def simulate(self, parameters: np.ndarray, nobs: int, rng: RNG, burn=500, initial_value=None):
        pass

    @garch_simulator_mc
    def simulate_mc(self, parameters: np.ndarray, reps: int, nobs: int, rng: RNG, burn=500, initial_value=None):
        pass


class HARCH(H):
    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        lags = self.lags
        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - np.sum(parameters[1:])) > 0:
                initial_value = parameters[0] / (1.0 - np.sum(parameters[1:]))
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.empty(nobs + burn)
        data = np.empty(nobs + burn)
        max_lag = np.max(lags)
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(initial_value)

        return vs.harch_simulate(nobs, burn, max_lag, parameters, data, sigma2, errors, lags.astype(np.int64))

    def simulate_mc(self, parameters: np.ndarray, reps: int, nobs: int, rng: RNG, burn=500, initial_value=None):
        raise NotImplementedError
