import numpy as np

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simulate_harx(
        double[:] y,
        int nobs,
        int k_x,
        int max_lag,
        bint constant,
        double[:, :] x,
        double[:] errors,
        double[:] params,
        int[:, :] lags):

    cdef:
        int t, i, j, ind
        int lag_start, lag_end
        double ar
        int[:] lag

    for t in range(max_lag, nobs):
        ind = 0
        if constant:
            y[t] = params[ind]
            ind += 1

        for j in range(len(lags)):
            ar, lag = 0.0, lags[j]
            for i in range(t-lag[1], t-lag[0]):
                ar += y[i]
            ar /= (lag[1] - lag[0])

            y[t] += params[ind] * ar
            ind += 1

        for i in range(k_x):
            y[t] += params[ind] * x[t, i]
        y[t] += errors[t]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simulate_harx_mc(
        double[:, :] y,
        int nobs,
        int reps,
        int k_x,
        int max_lag,
        bint constant,
        double[:, :] x,
        double[:, :] errors,
        double[:] params,
        int[:, :] lags):

    cdef double[:, :] data = _simulate_harx_mc(y, nobs, reps, k_x, max_lag, constant, x, errors, params, lags)
    return np.asarray(data, np.float64)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:, :] _simulate_harx_mc(
    double[:, :] y,
    int nobs,
    int reps,
    int k_x,
    int max_lag,
    bint constant,
    double[:, :] x,
    double[:, :] errors,
    double[:] params,
    int[:, :] lags
):
    cdef:
        int i, j, r, t, ind
        int[:] lag
        double ar

    for r in range(reps):
        for t in range(max_lag, nobs):
            ind = 0
            if constant:
                y[t, r] = params[ind]
                ind += 1

            for j in range(len(lags)):
                ar, lag = 0.0, lags[j]
                for i in range(t-lag[1], t-lag[0]):
                    ar += y[i, r]
                ar /= (lag[1] - lag[0])

                y[t, r] += params[ind] * ar
                ind += 1

            for i in range(k_x):
                y[t, r] += params[ind] * x[t, i]
            y[t, r] += errors[t, r]

    return y
