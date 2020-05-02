from typing import Callable, Collection, Optional, Union

import numpy as np
import pandas as pd

__all__ = ['Endog', 'Exog', 'Params', 'RngGen']

Params = Optional[np.ndarray]
RngGen = Callable[[Union[int, Collection[int]]], np.ndarray]

# Fit Options
Endog = Union[pd.DataFrame, np.ndarray]
Exog = Optional[Union[Collection[Optional[np.ndarray]], np.ndarray]]
