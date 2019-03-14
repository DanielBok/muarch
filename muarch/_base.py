from typing import Callable, Iterable, Optional, Union

import numpy as np
from arch.univariate.base import ARCHModelResult

from muarch.mean import ARX, ConstantMean, HARX, LS, ZeroMean

__all__ = ['_ArchBase', 'PARAMS', 'RNG_GEN']

RNG_GEN = Callable[[Union[int, Iterable[int]]], np.ndarray]
PARAMS = Optional[np.ndarray]


class _ArchBase:
    def __init__(self):
        self._model: Union[HARX, ARX, ConstantMean, LS, ZeroMean] = None
        self._fit_model: ARCHModelResult = None

    def simulation_horizon_required(self, nobs: int, burn: int):
        """
        Calculates the number of random generations needed for simulation

        Parameters
        ----------
        nobs: int
            number of observations

        burn: int
            number of observations burnt in simulation

        Returns
        -------
        int
            number of random generations required
        """
        num = nobs + burn * 2
        if self._model.distribution.__class__.__name__ == 'GeneralizedError':
            num *= 2
        return num
