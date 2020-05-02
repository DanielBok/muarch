from typing import Optional

import numpy as np
import scipy.special as sp
from arch.univariate.distribution import GeneralizedError as GE
from scipy.stats import gamma

from ._base import DistributionMixin, _format_simulator


class GeneralizedError(GE, DistributionMixin):
    def __init__(self, random_state=None):
        DistributionMixin.__init__(self)
        GE.__init__(self, random_state)

    @_format_simulator
    def _simulator(self, size: int, reps: Optional[int] = None):
        _parameters = self._parameters
        if self.custom_dist is None:
            if reps is not None:
                size = size, reps

            nu, *_ = _parameters
            randoms = self._random_state.standard_gamma(1 / nu, size) ** (1.0 / nu)
            randoms *= 2 * self._random_state.randint(0, 2, size) - 1
            scale = np.sqrt(sp.gamma(3.0 / nu) / sp.gamma(1.0 / nu))

            return randoms / scale
        else:
            self.derive_dist_size(size * 2)

            nu, *_ = _parameters

            randoms = gamma.ppf(self.custom_dist[:size], 1 / nu) ** (1. / nu)
            randoms *= 2 * np.asarray(self.custom_dist[size:2 * size] > 0.5, np.float) - 1
            scale = np.sqrt(sp.gamma(3.0 / nu) / sp.gamma(1.0 / nu))

            self.custom_dist = None  # reset simulator

            return randoms / scale
