Utility Functions
~~~~~~~~~~~~~~~~~

The utility functions help in adjusting the simulated data cube. There are some assumptions about the cube. Namely, assuming that we are running a Monte-Carlo simulation of asset returns, the axis will be 3 dimensional where each axis represents the time, trials and asset class respectively.

Calibrate Data
--------------
.. autofunction:: muarch.calibrate.calibrate_data

Truncate Outliers
-----------------
.. autofunction:: muarch.calibrate.truncate_outliers

Basic Statistics
----------------
.. autofunction:: muarch.funcs.moments.get_annualized_mean
.. autofunction:: muarch.funcs.moments.get_annualized_sd
.. autofunction:: muarch.funcs.moments.get_annualized_skew
.. autofunction:: muarch.funcs.moments.get_annualized_kurtosis
