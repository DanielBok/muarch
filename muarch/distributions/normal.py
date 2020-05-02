from typing import Optional

from arch.univariate.distribution import Normal as N
from scipy.stats import norm

from ._base import DistributionMixin, _format_simulator


class Normal(N, DistributionMixin):
    def __init__(self, random_state=None):
        DistributionMixin.__init__(self)
        N.__init__(self, random_state)

    @_format_simulator
    def _simulator(self, size: int, reps: Optional[int] = None):
        if self.custom_dist is None:
            if reps is not None:
                size = size, reps
            return self._random_state.standard_normal(size)

        else:
            self.derive_dist_size(size)
            ppf = norm.ppf(self.custom_dist[:size])
            self.custom_dist = None  # reset simulator

            return ppf
