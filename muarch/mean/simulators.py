from functools import wraps

import numpy as np
import pandas as pd
from arch.utility.array import ensure1d

from ._mean_simulations import simulate_harx, simulate_harx_mc

__all__ = ['harx_simulator', 'harx_mc_simulator']


def harx_simulator(class_func):
    class_func.__docs__ = """
        Simulates data from a linear regression, AR or HAR models
        
        Parameters
        ----------
        params: ndarray
            Parameters to use when simulating the model. Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.
            
        nobs: int
            Length of series to simulate

        burn: int, default 500
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used

        x: {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the simulation. This should be a 2D matrix

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        data_only: bool, default True
            If True, this returns only the simulated data, omits the volatility and error. In this case, it will return
            as a numpy array. Otherwise, it returns a data frame with the data, volatility and error

        Returns
        -------
        simulated_data : Union[DataFrame, numpy array]
            DataFrame with columns data containing the simulated values, volatility, containing the conditional
            volatility and errors containing the errors used in the simulation.
            If data_only, it returns the 'data' column as a numpy array
        
        Examples
        --------
        >>> import numpy as np
        >>> from muarch import UArch
        >>> from muarch.volatility import GARCH
        >>> harx = UArch(lags=[1, 5, 22])
        >>> harx.volatility = GARCH()
        >>> harx_params = np.array([1, 0.2, 0.3, 0.4])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((harx_params, garch_params))
        >>> sim_data = harx.simulate(params, 1000)

        Simulating models with exogenous regressors requires the regressors
        to have nobs plus burn data points

        >>> nobs = 100
        >>> burn = 200
        >>> x = np.random.randn(nobs + burn, 2)
        >>> x_params = np.array([1.0, 2.0])
        >>> params = np.concatenate((harx_params, x_params, garch_params))
        >>> sim_data = harx.simulate(params, nobs=nobs, burn=burn, x=x)
    """

    @wraps(class_func)
    def decorator(model, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        k_x = 0
        if x is not None:
            x = np.asarray(x)
            k_x = x.shape[1]
            if x.shape[0] != nobs + burn:
                raise ValueError('x must have nobs + burn rows')

        # added model._lags is not None for LS models
        lags = model._lags if model._lags is not None else np.zeros((0, 0), int)

        mc = int(model.constant) + lags.shape[1] + k_x
        vc = model.volatility.num_params
        dc = model.distribution.num_params
        num_params = mc + vc + dc
        params = ensure1d(params, 'params', series=False)
        if params.shape[0] != num_params:
            raise ValueError(f'params has the wrong number of elements. Expected {num_params}, got {params.shape[0]}. '
                             f'Perhaps you forgot to add the exogenous variables?')

        dist_params = [] if dc == 0 else params[-dc:]
        vol_params = params[mc:mc + vc]
        simulator = model.distribution.simulate(dist_params)
        sim_data = model.volatility.simulate(vol_params,
                                             nobs + burn,
                                             simulator,
                                             burn,
                                             initial_value_vol)
        errors = sim_data[0]
        vol = np.sqrt(sim_data[1])

        max_lag = np.max(lags) if lags.size else 0

        if initial_value is None:
            initial_value = 0.0
        elif not np.isscalar(initial_value):
            initial_value = ensure1d(initial_value, 'initial_value')
            if initial_value.shape[0] != max_lag:
                raise ValueError('initial_value has the wrong shape')

        y = np.zeros(nobs + burn)
        y[:max_lag] = initial_value

        simulate_harx(y, nobs + burn, k_x, max_lag, model.constant, x, errors, params, lags.T)

        if data_only:
            return y[burn:]

        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = pd.DataFrame(df)
        return df

    return decorator


def harx_mc_simulator(class_func):
    class_func.__docs__ = """
        Monte Carlo simulation from a linear regression, AR or HAR models
        
        Parameters
        ----------
        params: ndarray
            Parameters to use when simulating the model. Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.

        nobs: int
            Length of series to simulate

        reps: int
            Number of trials in the Monte Carlo simulation

        burn: int, default 500
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used

        x: {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the simulation. This should be a 2D matrix

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : ndarray
            matrix of simulated values

        Examples
        --------
        >>> import numpy as np
        >>> from muarch import MUArch
        >>> AR1_GARCH11= MUArch(3, mean='ARX', lags=1, vol='GARCH', p=1, q=1, scale=100)

        >>> sim_data = AR1_GARCH11.simulate_mc(nobs=36, reps=1000, n_jobs=-1)
        """

    @wraps(class_func)
    def decorator(model, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        k_x = 0
        if x is not None:
            x = np.asarray(x)
            k_x = x.shape[1]
            if x.shape[0] != nobs + burn:
                raise ValueError('x must have nobs + burn rows')

        # added model._lags is not None for LS models
        lags = model._lags if model._lags is not None else np.zeros((0, 0), int)

        mc = int(model.constant) + lags.shape[1] + k_x
        vc = model.volatility.num_params
        dc = model.distribution.num_params
        num_params = mc + vc + dc
        params = ensure1d(params, 'params', series=False)
        if params.shape[0] != num_params:
            raise ValueError(f'params has the wrong number of elements. Expected {num_params}, got {params.shape[0]}. '
                             f'Perhaps you forgot to add the exogenous variables?')

        dist_params = [] if dc == 0 else params[-dc:]
        vol_params = params[mc:mc + vc]
        simulator = model.distribution.simulate(dist_params)
        errors = model.volatility.simulate_mc(vol_params, nobs + burn, reps, simulator, burn, initial_value_vol)

        max_lag = np.max(lags) if lags.size else 0

        if initial_value is None:
            initial_value = 0
        elif np.isscalar(initial_value):
            initial_value = np.repeat(initial_value, max_lag)
        elif not np.isscalar(initial_value):
            initial_value = ensure1d(initial_value, 'initial_value')
            if initial_value.shape[0] != max_lag:
                raise ValueError('initial_value has the wrong shape')

        y = np.zeros((nobs + burn, reps), np.float64)  # y is modified in place
        y[:max_lag, :reps] = initial_value
        return simulate_harx_mc(y, nobs + burn, reps, k_x, max_lag, model.constant, x, errors, params, lags.T)[burn:]

    return decorator
