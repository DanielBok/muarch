from functools import wraps
from typing import Callable, Optional, Union
from warnings import warn

import numpy as np
from arch.utility.exceptions import InitialValueWarning, initial_value_warning

from ._vol_simulations import garch_simulate, garch_simulate_mc

RNG = Union[Callable[[int], np.ndarray], Callable[[int, Optional[int]], np.ndarray]]


def garch_simulator(class_func):
    @wraps(class_func)
    def decorator(model, parameters: np.ndarray, nobs: int, rng: RNG, burn=500, initial_value=None):
        p, o, q, power = model.p, model.o, model.q, model.power
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = np.ones_like(parameters)
            scale[p + 1:p + o + 1] = 0.5

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        fsigma = np.zeros(nobs + burn)
        fdata = np.zeros(nobs + burn)

        max_lag = np.max([p, o, q])
        fsigma[:max_lag] = initial_value
        sigma2[:max_lag] = initial_value ** (2.0 / power)
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]
        fdata[:max_lag] = abs(data[:max_lag]) ** power

        return garch_simulate(p, o, q, power, parameters, nobs, burn, max_lag, fsigma, fdata, data, sigma2, errors)

    return decorator


def garch_simulator_mc(class_func):
    @wraps(class_func)
    def decorator(model, parameters: np.ndarray, nobs: int, reps: int, rng: RNG, burn=500, initial_value=None):
        p, o, q, power = model.p, model.o, model.q, model.power
        errors = rng(nobs + burn, reps)

        if initial_value is None:
            scale = np.ones_like(parameters)
            scale[p + 1:p + o + 1] = 0.5

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                initial_value = parameters[0]

        return garch_simulate_mc(p, o, q, power, reps, nobs, burn, parameters, initial_value, errors)

    return decorator
