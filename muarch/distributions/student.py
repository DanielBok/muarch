from typing import Optional

import numpy as np
from arch.univariate.distribution import StudentsT as T

from ._base import DistributionMixin, _format_simulator


class StudentsT(T, DistributionMixin):
    def __init__(self, random_state=None):
        DistributionMixin.__init__(self)
        T.__init__(self, random_state)

    @_format_simulator
    def _simulator(self, size: int, reps: Optional[int] = None):
        nu = self._parameters[0]
        std_dev = np.sqrt(nu / (nu - 2))

        if self.custom_dist is None:
            if reps is not None:
                size = size, reps
            return self._random_state.standard_t(nu, size=size) / std_dev
        else:
            self.derive_dist_size(size)
            ppf = self.ppf(self.custom_dist[:size], nu)
            self.custom_dist = None  # reset simulator

            return ppf / std_dev
