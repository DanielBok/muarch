import os
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from ._base import RNG_GEN
from .summary import SummaryList
from .uarch import UArch

Exog = Optional[Union[List[Union[None, np.ndarray]], np.ndarray]]


class MUArch:
    """
    Multi-univariate ARCH model. Unlike a multivariate ARCH model, this fits each univariate time series individually.
    Any simulations returns simulations of each univariate series column bound together.
    """

    __n: int

    def __init__(self, n: Union[int, Iterable[UArch]], mean='Constant', lags=0, vol='GARCH', p=1, o=0, q=1, power=2.0,
                 dist='Normal', hold_back=None, scale=1):
        """
        Initializes the MUArch model.

        The MUArch model holds multiple univariate models which are determined during fitting. If the models are not
        specified, the global default options will be used. Models can be individually specified after initializing
        the MUArch instance.

        Parameters
        ----------
        n:  int or list of UArch models
            Number of univariate models to fit. Alternatively, a list of UArch (univariate) models can be specified.

        mean: { 'zero', 'constant', 'harx', 'har', 'ar', 'arx', 'ls' }, optional
            Name of the global default mean model.  Currently supported options are:

            * 'Constant' - Constant mean model
            * 'Zero' - Zero mean model
            * 'AR' - Autoregression model
            * 'ARX' - Autoregression model with exogenous regressors. Falls back to 'AR' if no exogenous regressors
            * 'HAR' - Heterogeneous Autoregression model
            * 'HARX' - Heterogeneous Autoregressions with exogenous regressors

            For more information on the different models, check out the documentation at
            https://muarch.readthedocs.io/en/latest/univariate/mean.html

        lags: int or list (int), optional
            Global default lag. Either a scalar integer value indicating lag length or a list of integers specifying
            lag locations.

        vol: { 'GARCH', 'ARCH', 'EGARCH', 'FIGARCH' and 'HARCH' }, optional
            Name of the global default volatility model.  Currently supported options are:
            'GARCH' (default), 'ARCH', 'EGARCH', 'FIGARCH' and 'HARCH'

        p: int, optional
            Global default lag order of the symmetric innovation

        o: int, optional
            Global default lag order of the asymmetric innovation

        q: int, optional
            Global default lag order of lagged volatility or equivalent

        power: float, optional
            Global default power to use with GARCH and related models

        dist:  { 'normal', 'gaussian', 'studentst', 't', 'skewstudent', 'skewt', 'ged', 'generalized error' }, optional
            Name of the global default error distribution.  Currently supported options are:
            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

        hold_back: int
            Global default. Number of observations at the start of the sample to exclude when estimating model
            parameters. Used when comparing models with different lag lengths to estimate on the common sample.

        scale: float
            Global default factor to scale data up or down by. This is useful when your data is too small leading to
            numerical errors when fitting. It will be used to scale simulation data
        """
        super(MUArch, self).__init__()

        if isinstance(n, abc.Iterable):
            self.__models = list(n)
            self.__n = len(self.__models)

            if self.__n == 0:
                raise ValueError('If passing in a list of UArch models, list cannot be empty!')

            for i, m in enumerate(self.__models):
                if not isinstance(m, UArch):
                    raise TypeError(f'The model in index {i} must be an instance of UArch')

        elif type(n) is int:
            self.__n = n
            self.__models: List[UArch] = [
                UArch(mean, lags, vol, p, o, q, power, dist, hold_back, scale)
                for i in range(n)]
        else:
            raise TypeError('`n` must either be an integer specifying the number of models or a list of UArch models')

        self._model_names = [str(i) for i in range(n)]

    def fit(self, y: Union[pd.DataFrame, np.ndarray], x: Exog = None, update_freq=1, disp='off', cov_type='robust',
            show_warning=True, tol=None, options=None):
        """
        Fits the MUArch model.

        If finer control over the MUArch models is required, set the UArch models separately. Otherwise, method will
        set the default parameters.

        Parameters
        ----------
        y: {ndarray, Series}
            The dependent variable. If a vector is passed in, it is assumed that the same vector (endog) is used for
            all models. Otherwise, the last value of the shape must match the number of models

        x: list of {ndarray, None}, optional
            Exogenous regressors.  Ignored if model does not permit exogenous regressors. If passed in, the first shape
            must match the number of models.

        update_freq: int, optional
            Frequency of iteration updates.  Output is generated every `update_freq` iterations. Set to 0 to disable
            iterative output

        disp: 'final' or 'off' (default)
            Either 'final' to print optimization result or 'off' to display nothing

        cov_type: str, optional
            Estimation method of parameter covariance.  Supported options are 'robust', which does not assume the
            Information Matrix Equality holds and 'classic' which does. In the ARCH literature, 'robust' corresponds
            to Bollerslev-Wooldridge covariance estimator.

        show_warning: bool, optional
            Flag indicating whether convergence warnings should be shown.

        tol: float, optional
            Tolerance for termination

        options: dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries include 'ftol', 'eps', 'disp', and 'maxiter'

        Returns
        -------
        MUArch
            Fitted self instance
        """

        if y.shape[1] != self.__n:  # checks on input data
            raise ValueError("number of columns in data 'y' does not match expected dimension of MUArch object")

        if isinstance(y, pd.DataFrame):  # setting names for params later
            self._model_names = list(y.columns)
            y = y.values
        else:
            self._model_names = [str(i) for i in range(self.__n)]

        if y.ndim == 1:
            y = np.tile(y, self.__n)
        elif y.ndim > 2:
            raise ValueError("Dependent variable `y` must either be 1 or 2 dimensional")

        for i in range(self.__n):
            yy = y[:, i]
            xx = x[i] if x is not None else None
            if xx is not None:
                xx = np.asarray(xx)
                if xx.ndim == 1:
                    xx = xx.reshape(len(xx), 1)

            self[i].fit(yy, xx, update_freq=update_freq, disp=disp, cov_type=cov_type, show_warning=show_warning,
                        tol=tol,
                        options=options)

    @property
    def params(self):
        return pd.DataFrame([m.params for m in self], self._model_names)

    def residuals(self, standardize=True) -> np.ndarray:
        """
        Model residuals

        The residuals will be burnt by the maximum lag of the underlying models. For example, given 3 models - AR(1),
        AR(10), Constant with 400 data points each, the residuals will be 399, 390 and 400 long. The function will
        cut off the first 10 data points in this instance.

        Parameters
        ----------
        standardize: bool, optional
            Whether to standardize residuals. Residuals are standardized by dividing it with the conditional volatility

        Returns
        -------
        ndarray
            Residuals
        """
        residuals = [mm.residuals(standardize) for mm in self]
        length = min([len(r) for r in residuals])

        return np.asarray([r[-length:] for r in residuals]).T

    def simulate(self,
                 nobs,
                 burn=500,
                 initial_value=None,
                 x=None,
                 initial_value_vol=None,
                 data_only=True,
                 custom_dist: Optional[Union[RNG_GEN, np.ndarray]] = None):
        """
        Simulates data from the multiple ARMA-GARCH models

        Parameters
        ----------
        nobs: int
            Length of series to simulate

        burn: int, optional
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used. If array, the last column must be of the same size as the number of models

        x: {ndarray, list of ndarray}, optional
            If supplied as a list, this list should have the same number of elements as the number of models in the
            MUArch model. Each array inside is the specified exogenous variable for that  particular model and
            this must be a nobs + burn by k matrix of exogenous variables to include in the simulation. Otherwise,
            leave the value as `None` to indicate no exogenous variables are used for simulation in the model.

            If an array is supplied directly, it means every model has an exogenous variable associated with it. In
            this case, it should be a 3 dimensional tensor where the first dimension represents the number of models.

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process. If array, the last column must be of the
            same size as the number of models

        data_only: bool, default True
            If True, this returns only the simulated data, omits the volatility and error. In this case, it will return
            as a numpy array. Otherwise, it returns a data frame with the data, volatility and error

        custom_dist: {ndarray, Callable}, optional
            Optional density from which to simulate the innovations (Distribution) in the GARCH models. This is useful
            when working with the copula-GARCH model where each univariate model innovations has dependence on others.
            It is assumed that the values supplied are standardized [0, 1] innovations instead of the unstandardized
            residuals.

            The shape of the array must be at least as long as the simulation size required after accounting for burn
            and type of innovation process. If unsure, use :code:`simulation_size_required` to check. It must also
            account for the number of dimensions of the MUArch model. For example, if MUArch model is simulating a
            horizon of 120 time steps, 10000 trials and has 5 UArch models, the shape of the numpy array should be
            (120, 10000, 5).

            If a random number generator function is passed in, ensure that it only takes only argument and returns
            a numpy array. The argument can be an integer or a tuple of integers. In this case, the size will be
            automatically derived to save the user the trouble.

        Returns
        -------
        simulated_data : {List[DataFrame], ndarray}
            List of DataFrame with columns data containing the simulated values, volatility, containing the conditional
            volatility and errors containing the errors used in the simulation

        See Also
        --------
        UArch.simulation_horizon_required: Calculates the simulation size required
        """

        initial_value, initial_value_vol, x, custom_dist = self._format_simulation_parameters(nobs,
                                                                                              burn,
                                                                                              None,
                                                                                              initial_value,
                                                                                              initial_value_vol,
                                                                                              x,
                                                                                              custom_dist)

        sims = []
        for i in range(self.__n):
            r = custom_dist[..., i] if custom_dist is not None else None
            xx = x[i]
            iv = initial_value[i]
            ivv = initial_value_vol[i]

            sims.append(self[i].simulate(nobs, burn, iv, xx, ivv, data_only, None, r))

        if data_only:
            sims = np.asarray(sims).T

        return sims

    def simulate_mc(self,
                    nobs,
                    reps,
                    burn=500,
                    initial_value=None,
                    x=None,
                    initial_value_vol=None,
                    custom_dist: Optional[Union[RNG_GEN, np.ndarray]] = None,
                    n_jobs: Optional[int] = None):
        """
        Simulates data from the multiple ARCH-GARCH models.

        This function is specially crafted for Monte-Carlo simulations.

        Parameters
        ----------
        nobs: int
            Length of series to simulate

        reps: int
            Number of repetitions

        burn: int, optional
            Number of values to simulate to initialize the model and remove dependence on initial values

        initial_value: {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to use when initializing the model.
            If omitted, 0.0 is used. If array, the last column must be of the same size as the number of models

        x: {ndarray, list of ndarray}, optional
            If supplied as a list, this list should have the same number of elements as the number of models in the
            MUArch model. Each array inside is the specified exogenous variable for that  particular model and
            this must be a nobs + burn by k matrix of exogenous variables to include in the simulation. Otherwise,
            leave the value as `None` to indicate no exogenous variables are used for simulation in the model.

            If an array is supplied directly, it means every model has an exogenous variable associated with it. In
            this case, it should be a 3 dimensional tensor where the first dimension represents the number of models.

        initial_value_vol: {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process. If array, the last column must be of the
            same size as the number of models

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

        n_jobs: int or None, optional
            The number of jobs to run in parallel for simulation. This is particularly useful when simulating large
            number of repetitions with more than 1 dimension. None defaults to using 1 processor. Any numbers less or
            equal to 0 means to use all processors. Even if a large number is used, it will be capped at the
            maximum number of processors available.


        Returns
        -------
        simulated_data : numpy array
            Array containing simulated data from the Monte Carlo Simulation
        """

        if not isinstance(reps, int) or reps < 1:
            raise ValueError('reps must be an integer greater than 0')

        # setting up job processes
        if n_jobs is None:
            n_jobs = 1
        elif not isinstance(n_jobs, int):
            raise ValueError('`n_jobs` must be an integer')
        else:
            n_jobs = min(os.cpu_count(), os.cpu_count() if n_jobs <= 0 else n_jobs)

        initial_value, initial_value_vol, x, custom_dist = self._format_simulation_parameters(nobs,
                                                                                              burn,
                                                                                              reps,
                                                                                              initial_value,
                                                                                              initial_value_vol,
                                                                                              x,
                                                                                              custom_dist)

        # formulate function calls to get simulated data
        functions = []
        for i in range(self.__n):
            r = custom_dist[..., i] if custom_dist is not None else None
            xx = x[i]
            iv = initial_value[i]
            ivv = initial_value_vol[i]

            func = partial(self[i].simulate_mc, nobs, reps, burn, iv, xx, ivv, None, r)
            functions.append(func)

        # switch between sequential and multiprocessing
        if n_jobs == 1:  # sequential
            sims = np.zeros((nobs, reps, self.__n))
            for i, func in enumerate(functions):
                sims[:, :, i] = func()

            return sims
        else:  # multiprocessing
            results = []

            with ProcessPoolExecutor(n_jobs) as P:
                for func in functions:
                    results.append(P.submit(func))

            results = np.asarray([r.result() for r in results])
            return np.moveaxis(results, 0, -1)

    def summary(self, short=False, dp=4):
        """
        Summary of fitted models

        :param short: bool, default False
            Whether to show short summary or full summary.
        :param dp: int, default 4
            Number of decimal places to show in short summary
        :return: SummaryList
            summary of fitted models
        """
        if short:
            return pd.DataFrame([m.summary(short, dp) for m in self], self._model_names)
        else:
            smry = SummaryList()
            for i, name in enumerate(self._model_names):
                s = self[i].summary(short, dp)
                s.add_header(name)
                smry.append(s)
            return smry

    def _format_simulation_parameters(self,
                                      nobs,
                                      burn,
                                      reps,
                                      initial_value,
                                      initial_value_vol,
                                      x,
                                      custom_dist):
        """Ensures the simulation are okay and return the formatted parameters"""

        # Format initial value
        if initial_value is None or isinstance(initial_value, (float, int)):
            initial_value = np.asarray([initial_value] * self.__n)

        # Format initial volatility value
        if initial_value_vol is None or isinstance(initial_value, (float, int)):
            initial_value_vol = np.asarray([initial_value_vol] * self.__n)

        # Format exogenous regressors
        if x is not None:
            if isinstance(x, np.ndarray):
                assert x.ndim == 3, ("If numpy array passed in as MUArch `x` (exog), make sure it is a 3 "
                                     "dimensional array where the last dimension is the number of elements")
                x = [x[i] for i in range(self.__n)]
            else:
                x = list(x)

                assert len(x) == self.__n, ("Exogenous variable's list should have the same number of elements as the "
                                            "number of models")

                for i, xx in enumerate(x):
                    if xx is None:
                        continue
                    elif isinstance(x[0], pd.DataFrame):
                        # convert all data frames to numpy array then move the model axis (first) to last
                        xx = np.moveaxis(np.array([df.values for df in x]), 0, -1)
                    else:
                        xx = np.asarray(xx)

                    if xx.ndim == 1:
                        xx = xx.reshape(-1, 1)
                    x[i] = xx
        else:
            x = [None] * self.__n

        # Format custom distribution
        if custom_dist is not None:
            horizon = max(m.simulation_horizon_required(nobs, burn) for m in self)
            if isinstance(reps, int):  # for monte carlo
                size = horizon, reps
            else:
                size = horizon

            if callable(custom_dist):
                custom_dist = np.asarray(custom_dist(size))
            else:
                custom_dist = np.asarray(custom_dist)

            if custom_dist.ndim == 1:
                custom_dist = custom_dist.reshape(1, -1)

            dim = custom_dist.shape[-1]
            assert dim == self.__n, (f"expected generator to return array with {self.__n} vectors in the last axis "
                                     f"(of shape) but got {dim}")

        return initial_value, initial_value_vol, x, custom_dist

    def _repr_html_(self):
        header = '<h2>Arch Models</h2>'
        models = '<br/>'.join(m._repr_html_() for m in self)
        return header + models

    def __getitem__(self, i: int):
        return self.__models[i]

    def __setitem__(self, i: int, model: UArch):
        assert isinstance(model, UArch), 'only UArch models are allowed to be set as a component of the MUArch object'
        self.__models[i] = model

    def __repr__(self):
        return self.__str__() + f', {hex(id(self))}'

    def __str__(self):
        txt = f'{"Arch Models":^80}\n' + '=' * 80 + '\n\n'
        txt += '\n\n\n'.join(str(m) for m in self)
        return txt

    def __len__(self):
        return self.__n
