from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ._base import PARAMS, RNG_GEN, _ArchBase
from .distributions import GeneralizedError, Normal, SkewStudent, StudentsT
from .exceptions import NotFittedError
from .mean import ARX, ConstantMean, HARX, LS, ZeroMean
from .summary import Summary
from .volatility import ARCH, ConstantVariance, EGARCH, FIGARCH, GARCH, HARCH

__all__ = ['UArch']


class UArch(_ArchBase):
    """
    Univariate ARCH model that wraps on top of Mean, Volatility and Distribution classes defined in the `arch` package.
    Mainly, this class combines the original model and fitted model in the `arch` package for convenience. It has also
    some additional methods such as :code:`simulate_mc` for Monte Carlo simulations.
    """

    def __init__(self, mean='Constant', lags=0, vol='GARCH', p=1, o=0, q=1, power=2.0, dist='Normal', hold_back=None,
                 scale=1):
        """
        Creates the wrapping arch model

        Parameters
        ----------
        mean: { 'zero', 'constant', 'harx', 'har', 'ar', 'arx', 'ls' }, optional
            Name of the mean model.  Currently supported options are:

            * 'Constant' - Constant mean model (default)
            * 'Zero' - Zero mean model
            * 'AR' - Autoregression model
            * 'ARX' - Autoregression model with exogenous regressors. Falls back to 'AR' if no exogenous regressors
            * 'HAR' - Heterogeneous Autoregression model
            * 'HARX' - Heterogeneous Autoregressions with exogenous regressors

            For more information on the different models, check out the documentation at
            https://arch.readthedocs.io/en/latest/univariate/mean.html

        lags: int or list (int), optional
            Either a scalar integer value indicating lag length or a list of integers specifying lag locations.

        vol: { 'GARCH', 'ARCH', 'EGARCH', 'FIGARCH' and 'HARCH' }, optional
            Name of the volatility model.  Currently supported options are:
            'GARCH' (default), 'ARCH', 'EGARCH', 'FIGARCH' and 'HARCH'

        p: int, optional
            Lag order of the symmetric innovation

        o: int, optional
            Lag order of the asymmetric innovation

        q: int, optional
            Lag order of lagged volatility or equivalent

        power: float, optional
            Power to use with GARCH and related models

        dist: { 'normal', 'gaussian', 'studentst', 't', 'skewstudent', 'skewt', 'ged', 'generalized error' }, optional
            Name of the error distribution.  Currently supported options are:
            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

        hold_back: int, optional
            Number of observations at the start of the sample to exclude when estimating model parameters. Used when
            comparing models with different lag lengths to estimate on the common sample.

        scale: float
            Factor to scale data up or down by. This is useful when your data is too small leading to numerical errors
            when fitting. It will be used to scale simulation data
        """

        super(UArch, self).__init__()

        known_mean = ('zero', 'constant', 'harx', 'har', 'ar', 'arx', 'ls')
        known_vol = ('arch', 'figarch', 'garch', 'harch', 'constant', 'egarch')
        known_dist = ('normal', 'gaussian', 'studentst', 't', 'skewstudent',
                      'skewt', 'ged', 'generalized error')

        mean = mean.lower()
        vol = vol.lower()
        dist = dist.lower()
        if mean not in known_mean:
            raise ValueError(f"Unknown model type '{mean}' in mean")
        if vol.lower() not in known_vol:
            raise ValueError(f"Unknown model type '{vol}' in vol")
        if dist.lower() not in known_dist:
            raise ValueError(f"Unknown model type '{dist}' in dist")

        self._scale = scale

        self._model_setup = {
            'mean': mean,
            'lags': lags,
            'vol': vol,
            'p': p,
            'o': o,
            'q': q,
            'power': power,
            'dist': dist,
            'hold_back': hold_back
        }
        self._form_model(None)

    def fit(self, y, x=None, update_freq=1, disp='off', starting_values=None, cov_type='robust', show_warning=True,
            first_obs=None, last_obs=None, tol=None, options=None, backcast=None):
        r"""
        Fits the model given a nobs by 1 vector of sigma2 values

        Parameters
        ----------
        y: {ndarray, Series}
            The dependent variable

        x: {ndarray, DataFrame}, optional
            Exogenous regressors.  Ignored if model does not permit exogenous regressors.

        update_freq: int, optional
            Frequency of iteration updates.  Output is generated every `update_freq` iterations. Set to 0 to disable
            iterative output

        disp: 'final' or 'off' (default)
            Either 'final' to print optimization result or 'off' to display nothing

        starting_values: ndarray, optional
            Array of starting values to use.  If not provided, starting values are constructed by the model components

        cov_type:  str, optional
            Estimation method of parameter covariance.  Supported options are 'robust', which does not assume the
            Information Matrix Equality holds and 'classic' which does.  In the ARCH literature, 'robust' corresponds
            to Bollerslev-Wooldridge covariance estimator.

        show_warning: bool, optional
            Flag indicating whether convergence warnings should be shown.

        first_obs:  {int, str, datetime, Timestamp}
            First observation to use when estimating model

        last_obs: {int, str, datetime, Timestamp}
            Last observation to use when estimating model

        tol: float, optional
            Tolerance for termination

        options: dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries include 'ftol', 'eps', 'disp', and 'maxiter'

        backcast: float, optional
            Value to use as backcast. Should be measure :math:`\sigma^2_0`  since model-specific non-linear
            transformations are applied to value before computing the variance recursions.

        Returns
        -------
        UArch
            Fitted UArch instance
        """
        self._form_model(y, x)

        self._fit_model = self._model.fit(update_freq, disp, starting_values, cov_type, show_warning, first_obs,
                                          last_obs, tol, options, backcast)
        return self

    def forecast(self, params=None, horizon=1, start=None, align='origin', method='analytic',
                 simulations=1000, rng=None):
        """
        Construct forecasts from estimated model

        Parameters
        ----------
        params : ndarray, optional
            Alternative parameters to use.  If not provided, the parameters estimated when fitting the model are used.
            Must be identical in shape to the parameters computed by fitting the model.

        horizon : int, optional
           Number of steps to forecast

        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to produce the forecast for. Datetimes can
            only be used with pandas inputs that have a datetime index. Strings must be convertible to a date time,
            such as in '1945-01-01'.

        align : {'origin', 'target'}, optional
            When set to 'origin', the t-th row of forecasts contains the forecasts for t+1, t+2, ..., t+h.

            When set to 'target', the t-th row contains the 1-step ahead forecast from time t-1, the 2 step from
            time t-2, ..., and the h-step from time t-h. 'target' simplifies computing forecast errors since the
            realization and h-step forecast are aligned.

        method : {'analytic', 'simulation', 'bootstrap'}, optional
            Method to use when producing the forecast. The default is 'analytic'. The method only affects the
            variance forecast generation.  Not all volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not support the 'analytic' method
            for horizons > 1.

        simulations : int, optional
            Number of simulations to run when computing the forecast using either simulation or bootstrap.

        rng : {callable, ndarray}, optional
            If using a custom random number generator to for simulation-based forecasts, function must produce random
            samples using the syntax `rng(size)` where size is a 2-element tuple (simulations, horizon).

            Else, if a numpy array is passed in, array must have shape (simulation x horizon).

        Returns
        -------
        forecasts : ARCHModelForecast
            t by h data frame containing the forecasts.  The alignment of the forecasts is controlled by `align`.

        Notes
        -----
        The most basic 1-step ahead forecast will return a vector with the same length as the original data, where
        the t-th value will be the time-t forecast for time t + 1.  When the horizon is > 1, and when using the
        default value for `align`, the forecast value in position [t, h] is the time-t, h+1 step ahead forecast.

        If model contains exogenous variables (`model.x is not None`), then only 1-step ahead forecasts are available.
        Using horizon > 1 will produce a warning and all columns, except the first, will be nan-filled.

        If `align` is 'origin', forecast[t,h] contains the forecast made using y[:t] (that is, up to but not
        including t) for horizon h + 1.  For example, y[100,2] contains the 3-step ahead forecast using the first
        100 data points, which will correspond to the realization y[100 + 2]. If `align` is 'target', then the same
        forecast is in location [102, 2], so that it is aligned with the observation to use when evaluating,
        but still in the same column.
        """
        if params is None:
            params = self.params
        else:
            if params.size != np.array(self.params).size or params.ndim != self.params.ndim:
                raise ValueError('`params` have incorrect dimensions')

        if rng is not None and not callable(rng):
            rng = np.asarray(rng)
            if rng.shape != (simulations, horizon):
                raise ValueError(f"`rng` array should have dimensions ({(simulations, horizon)})")
            rng = lambda _: rng

        return self._fit_model.forecast(params, horizon, start, align, method, simulations, rng)

    def hedgehog_plot(self, params=None, horizon=10, step=10, start=None, type_='volatility', method='analytic',
                      simulations=1000):
        """
        Plot forecasts from estimated model

        Parameters
        ----------
        params: {Series, ndarray}, optional
            Alternative parameters to use.  If not provided, the parameters
            computed by fitting the model are used.  Must be 1-d and identical
            in shape to the parameters computed by fitting the model

        horizon: int, optional
            Number of steps to forecast

        step: int, optional
            Non-negative number of forecasts to skip between spines

        start: int, datetime or str, optional
            An integer, datetime or str indicating the first observation to produce the forecast for. Datetimes can only
            be used with pandas inputs that have a datetime index. Strings must be convertible to a date time, such as
            in '1945-01-01'. If not provided, the start is set to the earliest forecastable date

        type_: {'volatility', 'mean'}
            Quantity to plot, the forecast volatility or the forecast mean

        method: {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic. The method only affects the variance
            forecast generation.  Not all volatility models support all methods. In particular, volatility models that
            do not evolve in squares such as EGARCH or TARCH do not support the 'analytic' method for horizons > 1

        simulations: int
            Number of simulations to run when computing the forecast using either simulation or bootstrap

        Returns
        -------
        figure
            Handle to the figure
        """
        if params is None:
            params = self.params
        return self._fit_model.hedgehog_plot(params, horizon, step, start, type_, method, simulations)

    @property
    def params(self) -> pd.Series:
        """Model Parameters"""
        self._ensure_is_fitted()
        return self._fit_model.params

    def residual_plot(self, annualize=None, scale=None):
        """
        Plot standardized residuals and conditional volatility

        Parameters
        ----------
        annualize: str, optional
            String containing frequency of data that indicates plot should contain annualized volatility. Supported
            values are 'D' (daily), 'W' (weekly) and 'M' (monthly), which scale variance by 252, 52, and 12 respectively

        scale: float, optional
            Value to use when scaling returns to annualize.  If scale is provides, annualize is ignored and the value
            in scale is used.

        Returns
        -------
        figure
            Handle to the figure
        """
        return self._fit_model.plot(annualize, scale)

    def residuals(self, standardize=True) -> np.ndarray:
        """
        Model residuals

        Parameters
        ----------
        standardize: bool, optional
            Whether to standardize residuals. Residuals are standardized by dividing it with the conditional volatility

        Returns
        -------
        ndarray
            Residuals
        """
        model = self._fit_model
        lags = self._model_setup['lags']
        r = np.asarray(model.resid) / self._scale

        if standardize:
            r /= model.conditional_volatility
        return r[lags:]

    def simulate(self,
                 nobs: int,
                 burn=500,
                 initial_value: Union[np.ndarray, float] = None,
                 x: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 initial_value_vol: Union[np.ndarray, float] = None,
                 data_only=False,
                 params: PARAMS = None,
                 custom_dist: Optional[Union[RNG_GEN, np.ndarray]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Simulates data from a ARMA-GARCH model

        Parameters
        ----------
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

        data_only: bool, default True
            If True, this returns only the simulated data, omits the volatility and error. In this case, it will return
            as a numpy array. Otherwise, it returns a data frame with the data, volatility and error

        params: ndarray, optional
            If not None, model will use the parameters supplied to generate simulations. Otherwise, it will use the
            fitted parameters.

        custom_dist: {ndarray, Callable}, optional
            Optional density from which to simulate the innovations (Distribution) in the GARCH models. This is useful
            when working with the copula-GARCH model where each univariate model innovations has dependence on others.
            It is assumed that the values supplied are standardized [0, 1] innovations instead of the unstandardized
            residuals.

            The shape of the array must be at least as long as the simulation size required after accounting for burn
            and type of innovation process. If unsure, use :code:`simulation_size_required` to check.

            If a random number generator function is passed in, ensure that it only takes only argument and returns
            a numpy array. The argument can be an integer or a tuple of integers. In this case, the size will be
            automatically derived to save the user the trouble.

        Returns
        -------
        DataFrame or ndarray
            DataFrame with columns data containing the simulated values, volatility, containing the conditional
            volatility and errors containing the errors used in the simulation.
            If data_only, it returns the 'data' column as a numpy array
        """
        self._set_custom_dist(custom_dist, nobs, burn)
        if params is None:
            params = self.params

        return self._model.simulate(params, nobs, burn, initial_value, x, initial_value_vol, data_only) / self._scale

    def simulate_mc(self,
                    nobs: int,
                    reps: int,
                    burn=500,
                    initial_value: Union[np.ndarray, float] = None,
                    x: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                    initial_value_vol: Union[np.ndarray, float] = None,
                    params: PARAMS = None,
                    custom_dist: Optional[Union[RNG_GEN, np.ndarray]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Simulates data from a ARMA-GARCH model with multiple repetitions.

        This is used for Monte Carlo simulations.

        Parameters
        ----------
        nobs: int
            Length of series to simulate

        reps: int
            Number of repetitions in Monte Carlo simulation

        burn: int, optional
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used

        x: {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the simulation. This should be a 2D matrix

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        params: {Series, ndarray}, optional
            If not None, model will use the parameters supplied to generate simulations. Otherwise, it will use the
            fitted parameters.

        custom_dist: {ndarray, Callable}, optional
            Optional density from which to simulate the innovations (Distribution) in the GARCH models. This is useful
            when working with the copula-GARCH model where each univariate model innovations has dependence on others.
            It is assumed that the values supplied are standardized [0, 1] innovations instead of the unstandardized
            residuals.

            The shape of the array must be at least as long as the simulation size required after accounting for burn
            and type of innovation process. If unsure, use :code:`simulation_size_required` to check.

            If a random number generator function is passed in, he size will be automatically derived to save the user
            the trouble. However, the function must:

            * take as it first argument an integer or a tuple of integer
            * have other parameters that are optional
            * return a numpy array

        Returns
        -------
        simulated_data: ndarray
            Array containing the simulated values

        See Also
        --------
        UArch.simulation_horizon_required: Calculates the simulation size required
        """
        if not isinstance(reps, int) or reps < 1:
            raise ValueError('reps must be an integer greater than 0')

        self._set_custom_dist(custom_dist, nobs, burn, reps)
        if params is None:
            params = self.params

        return self._model.simulate_mc(params, nobs, reps, burn, initial_value, x, initial_value_vol) / self._scale

    def summary(self, short=False, dp=4) -> Union[pd.Series, Summary]:
        """
        Summary of fitted model

        Parameters
        ----------
        short: bool, optional
            Whether to show short summary or full summary.

        dp: int, optional
            Number of decimal places to show in short summary

        Returns
        -------
        Summary
            Model Summary
        """
        self._ensure_is_fitted()
        if short:
            params = self._fit_model.params.round(dp)
            se = self._fit_model.std_err.round(dp)
            return pd.Series(['{p:.{dp}f} ± {s:.{dp}f}'.format(p=p, s=se[i], dp=dp) for i, p in enumerate(params)],
                             index=params.index)
        else:
            return Summary(self._fit_model.summary())

    def _ensure_is_fitted(self):
        """Ensures model is fitted. Else raises error"""
        if self._fit_model is None:
            raise NotFittedError(f"This ArchModel instance is not fitted yet")

    def _form_model(self, y, x=None):
        """
        Forms the actual ARCH model. This is a convenience method so that users can specify data at run time
        instead of at class instantiation
        """
        mean: str = self._model_setup['mean']
        lags: Union[int, List[int]] = self._model_setup['lags']
        vol: str = self._model_setup['vol']
        p: int = self._model_setup['p']
        o: int = self._model_setup['o']
        q: int = self._model_setup['q']
        power: float = self._model_setup['power']
        dist: str = self._model_setup['dist']
        hold_back: Optional[int] = self._model_setup['hold_back']

        y = y * self._scale if y is not None else None
        if x is not None:
            x = np.asarray(x) * self._scale
            if x.ndim == 1:
                x = x.reshape(-1, 1)

        if mean == 'zero':
            am = ZeroMean(y, hold_back=hold_back)
        elif mean == 'constant':
            am = ConstantMean(y, hold_back=hold_back)

        elif mean == 'arx':
            am = ARX(y, x, lags, hold_back=hold_back)
        elif mean == 'ar':
            am = ARX(y, None, lags, hold_back=hold_back)
        elif mean == 'harx':
            am = HARX(y, x, lags, hold_back=hold_back)
        elif mean == 'har':
            am = HARX(y, None, lags, hold_back=hold_back)
        else:  # mean == 'ls'
            am = LS(y, x, hold_back=hold_back)

        if vol == 'constant':
            v = ConstantVariance()
        elif vol == 'arch':
            v = ARCH(p=p)
        elif vol == 'figarch':
            v = FIGARCH(p=p, q=q)
        elif vol == 'garch':
            v = GARCH(p=p, o=o, q=q, power=power)
        elif vol == 'egarch':
            v = EGARCH(p=p, o=o, q=q)
        else:  # vol == 'harch'
            v = HARCH(lags=p)

        if dist in ('skewstudent', 'skewt'):
            d = SkewStudent()
        elif dist in ('studentst', 't'):
            d = StudentsT()
        elif dist in ('ged', 'generalized error'):
            d = GeneralizedError()
        else:  # ('gaussian', 'normal')
            d = Normal()

        am.volatility = v
        am.distribution = d
        self._model = am

    def __model_description__(self, include_lags=True, include_fitted_stats=False):
        """Generates the model description for use by __str__ and related functions"""
        desc = self._model._model_description(include_lags)

        if self._fit_model is not None and include_fitted_stats:
            pass

        return desc

    def _repr_html_(self):
        """HTML representation for IPython Notebook"""
        desc = self.__model_description__(include_fitted_stats=True)

        html = f"<strong>{self._model.name}</strong>"

        fragment = ",\n".join(f"<strong>{key}: </strong> {val}" for key, val in desc.items())
        html = f"{html}(\n{fragment}\n<strong>ID: </strong>: {hex(id(self))})"

        return html

    def _set_custom_dist(self, custom_dist, nobs, burn, reps=None):
        horizon = self.simulation_horizon_required(nobs, burn)
        if isinstance(reps, int):  # for monte carlo
            size = horizon, reps
        else:
            size = horizon

        if custom_dist is not None:
            if callable(custom_dist):
                self._model.distribution.custom_dist = custom_dist(size)
            else:
                custom_dist = np.asarray(custom_dist)
                if len(custom_dist) < horizon:
                    raise ValueError(f"`custom_dist` array size should at least be {horizon} long")
                if reps is not None and reps != custom_dist.shape[1]:
                    raise ValueError(f"number of columns `custom_dist` array should equals {reps}")

                self._model.distribution.custom_dist = custom_dist

    def __repr__(self):
        txt = self.__str__()
        return txt + f', {hex(id(self))}'

    def __str__(self):
        desc = self.__model_description__()
        return f"{self._model.name}({', '.join(f'{k}: {v}' for k, v in desc.items() if k and v)})"
