from typing import Optional

import numpy as np
from arch.univariate.distribution import SkewStudent as SS
from scipy.stats import t, uniform

from ._base import DistributionMixin, _format_simulator


class SkewStudent(DistributionMixin, SS):
    def __init__(self, random_state=None):
        DistributionMixin.__init__(self)
        SS.__init__(self, random_state)

    @_format_simulator
    def _simulator(self, size: int, reps: Optional[int] = None) -> np.ndarray:
        if self.custom_dist is None:
            if reps is not None:
                size = size, reps
            return self.ppf(uniform.rvs(size=size), self._parameters)
        else:
            self.check_dist_size(size)
            ppf = self.ppf(self.custom_dist[:size], nu)
            self.custom_dist = None  # reset simulator

            return ppf

    def ppf(self, pits, parameters=None):
        self._check_constraints(parameters)

        scalar = np.isscalar(pits)
        if scalar:
            pits = np.array([pits])
        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        cond = pits < (1 - lam) / 2

        # slight speed up for really large problems
        icdf1 = t._ppf(pits[cond] / (1 - lam), eta)
        icdf2 = t._ppf(.5 + (pits[~cond] - (1 - lam) / 2) / (1 + lam), eta)
        icdf = -999.99 * np.ones_like(pits)
        icdf[cond] = icdf1
        icdf[~cond] = icdf2
        icdf = (icdf * (1 + np.sign(pits - (1 - lam) / 2) * lam) * (1 - 2 / eta) ** .5 - a)
        icdf = icdf / b

        if scalar:
            icdf = icdf[0]
        return icdf
