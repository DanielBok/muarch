from typing import Callable

import numpy as np
import pandas as pd
from arch.univariate.mean import ARX as A, ConstantMean as C, HARX as H, LS as _LS, ZeroMean as Z

from .simulators import harx_mc_simulator, harx_simulator

__all__ = ['ARX', 'ConstantMean', 'HARX', 'LS', 'ZeroMean']

GeneratorFunc = Callable[[int], np.ndarray]


class ARX(A):
    @harx_simulator
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        pass

    @harx_mc_simulator
    def simulate_mc(self, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        pass


class HARX(H):
    @harx_simulator
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        pass

    @harx_mc_simulator
    def simulate_mc(self, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        pass


class LS(_LS):
    @harx_simulator
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        """
        Examples
        --------
        Basic data simulation with a constant mean and volatility

        >>> import numpy as np
        >>> from muarch.mean import LS
        >>> from muarch.volatility import GARCH

        >>> ls = LS()
        >>> ls.volatility = GARCH()
        >>> ls_params = np.array([1, 1.5])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((ls_params, garch_params))
        >>> nobs, burn = 100, 400
        >>> sim_data = ls.simulate(params, nobs, burn,  x = np.random.normal(size=[nobs + burn, 1]))
        """
        pass

    @harx_mc_simulator
    def simulate_mc(self, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from muarch import MUArch
        >>> from muarch.datasets import load_etf

        >>> prices = load_etf()
        >>> CONST_GARCH11= MUArch(3, mean='Constant', lags=1, vol='GARCH', p=1, q=1, scale=100)
        >>> CONST_GARCH11.fit(prices)

        >>> sim_data = CONST_GARCH11.simulate_mc(nobs=36, reps=10, n_jobs=-1)
        """
        pass


class ConstantMean(C):
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        """
        Simulated data from a Constant Mean model

        Parameters
        ----------
        params: ndarray
            Parameters to use when simulating the model. Parameter order is [mean volatility distribution] where the
            parameters of the mean model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ... ex[k-1]] where lag[j]
            indicates the coefficient on the jth lag in the model and ex[j] is the coefficient on the jth exogenous
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

        data_only: bool, optional
            If True, this returns only the simulated data, omits the volatility and error. In this case, it will return
            as a numpy array. Otherwise, it returns a data frame with the data, volatility and error

        Returns
        -------
        simulated_data : {DataFrame, ndarray}
            DataFrame with columns data containing the simulated values, volatility, containing the conditional
            volatility and errors containing the errors used in the simulation.
            If data_only, it returns the 'data' column as a numpy array

        Examples
        --------
        Basic data simulation with a constant mean and volatility

        >>> import numpy as np
        >>> from muarch.mean import ConstantMean
        >>> from muarch.volatility import GARCH

        >>> cm = ConstantMean()
        >>> cm.volatility = GARCH()
        >>> cm_params = np.array([1])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((cm_params, garch_params))
        >>> sim_data = cm.simulate(params, 20)
        """
        if initial_value is not None or x is not None:
            raise ValueError('Both initial value and x must be none when '
                             'simulating a constant mean process.')

        mp, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(vp,
                                              nobs + burn,
                                              self.distribution.simulate(dp),
                                              burn,
                                              initial_value_vol)
        errors = sim_values[0]
        y = errors + mp

        if data_only:
            return y[burn:]

        vol = np.sqrt(sim_values[1])
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        return pd.DataFrame(df)

    def simulate_mc(self, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        """
        Monte Carlo simulation from a Constant Mean model

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
        simulated_data: ndarray
            matrix of simulated values

        Examples
        --------
        >>> import numpy as np
        >>> from muarch import MUArch
        >>> from muarch.datasets import load_etf

        >>> prices = load_etf()
        >>> CONST_GARCH11= MUArch(3, mean='Constant', lags=1, vol='GARCH', p=1, q=1, scale=100)
        >>> CONST_GARCH11.fit(prices)
        >>> sim_data = CONST_GARCH11.simulate_mc(nobs=36, reps=10, n_jobs=-1)
        """
        if initial_value is not None or x is not None:
            raise ValueError('Both initial value and x must be none when '
                             'simulating a constant mean process.')

        mp, vp, dp = self._parse_parameters(params)
        simulator = self.distribution.simulate(dp)
        errors = self.volatility.simulate_mc(vp, nobs + burn, reps, simulator, burn, initial_value_vol)
        y = errors + mp

        return y[burn:]


class ZeroMean(Z):
    def simulate(self, params, nobs, burn=500, initial_value=None, x=None, initial_value_vol=None, data_only=False):
        """
        Simulated data from a Zero Mean model

        params: ndarray
            Parameters to use when simulating the model. Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.

        nobs: int
            Length of series to simulate

        burn: int, optional
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used

        x: {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the simulation. This should be a 2D matrix

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        data_only: bool, optional
            If True, this returns only the simulated data, omits the volatility and error. In this case, it will return
            as a numpy array. Otherwise, it returns a data frame with the data, volatility and error

        Returns
        -------
        simulated_data: {DataFrame ,ndarray}
            DataFrame with columns data containing the simulated values, volatility, containing the conditional
            volatility and errors containing the errors used in the simulation.
            If data_only, it returns the 'data' column as a numpy array

        Examples
        --------
        Basic data simulation with no mean and constant volatility

        >>> from muarch.mean import ZeroMean
        >>> zm = ZeroMean()
        >>> sim_data = zm.simulate([1.0], 20)

        Simulating data with a non-trivial volatility process

        >>> from muarch.volatility import GARCH
        >>> zm.volatility = GARCH(p=1, o=1, q=1)
        >>> sim_data = zm.simulate([0.05, 0.1, 0.1, 0.8], 300)
        """
        if initial_value is not None or x is not None:
            raise ValueError('Both initial value and x must be none when simulating a constant mean process.')

        _, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(vp,
                                              nobs + burn,
                                              self.distribution.simulate(dp),
                                              burn,
                                              initial_value_vol)
        errors = sim_values[0]
        y = errors

        if data_only:
            return np.asarray(y[burn:])

        vol = np.sqrt(sim_values[1])
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        return pd.DataFrame(df)

    def simulate_mc(self, params, nobs, reps, burn=500, initial_value=None, x=None, initial_value_vol=None):
        """
        Monte Carlo simulation from a constant mean model

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

        burn: int, optional
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
        simulated_data: ndarray
            matrix of simulated values

        Examples
        --------
        >>> import numpy as np
        >>> from muarch import MUArch
        >>> from muarch.datasets import load_etf

        >>> prices = load_etf()
        >>> CONST_GARCH11= MUArch(3, mean='Constant', lags=1, vol='GARCH', p=1, q=1, scale=100)
        >>> CONST_GARCH11.fit(prices)
        >>> sim_data = CONST_GARCH11.simulate_mc(nobs=36, reps=10, n_jobs=-1)
        """
        if initial_value is not None or x is not None:
            raise ValueError('Both initial value and x must be none when simulating a constant mean process.')

        _, vp, dp = self._parse_parameters(params)

        simulator = self.distribution.simulate(dp)
        errors = self.volatility.simulate_mc(vp, nobs + burn, reps, simulator, burn, initial_value_vol)
        return errors[burn:]
