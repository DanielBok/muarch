cimport cython

from libc.math cimport sqrt, pi
import numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def garch_simulate(int p,
                   int o,
                   int q,
                   double power,
                   double[:] parameters,
                   int nobs,
                   int burn,
                   int max_lag,
                   double[:] fsigma,
                   double[:] fdata,
                   double[:] data,
                   double[:] sigma2,
                   double[:] errors):
    cdef:
        int j, t, loc

    for t in range(max_lag, nobs + burn):
        loc = 0
        fsigma[t] = parameters[loc]
        loc += 1
        for j in range(p):
            fsigma[t] += parameters[loc] * fdata[t - 1 - j]
            loc += 1
        for j in range(o):
            fsigma[t] += parameters[loc] * fdata[t - 1 - j] * (data[t - 1 - j] < 0)
            loc += 1
        for j in range(q):
            fsigma[t] += parameters[loc] * fsigma[t - 1 - j]
            loc += 1

        sigma2[t] = fsigma[t] ** (2.0 / power)
        data[t] = errors[t] * sqrt(sigma2[t])
        fdata[t] = abs(data[t]) ** power

    return data[burn:], sigma2[burn:]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def egarch_simulate(int p,
                    int o,
                    int q,
                    int nobs,
                    int burn,
                    int max_lag,
                    double[:] parameters,
                    double[:] data,
                    double[:] sigma2,
                    double[:] lnsigma2,
                    double[:] errors):
    cdef:
        int j, t, loc
        double[:] abserrors = np.abs(errors)
        double norm_const = (2 / pi) ** 0.5

    for t in range(max_lag, nobs + burn):
        lnsigma2[t] = parameters[0]
        loc = 1
        for j in range(p):
            lnsigma2[t] += parameters[loc] * (abserrors[t - 1 - j] - norm_const)
            loc += 1
        for j in range(o):
            lnsigma2[t] += parameters[loc] * errors[t - 1 - j]
            loc += 1
        for j in range(q):
            lnsigma2[t] += parameters[loc] * lnsigma2[t - 1 - j]
            loc += 1

    sigma2 = np.exp(lnsigma2)
    data = errors * np.sqrt(sigma2)

    return data[burn:], sigma2[burn:]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def figarch_simulate(int nobs,
                     int burn,
                     int truncation,
                     double power,
                     double omega_tilde,
                     double[:] data,
                     double[:] fdata,
                     double[:] fsigma,
                     double[:] sigma2,
                     double[:] lam_rev,
                     double[:] errors):
    cdef int i, t

    for t in range(truncation, truncation + nobs + burn):
        fsigma[t] = omega_tilde
        for i in range(t - truncation, t):  # dot product
            fsigma[t] += lam_rev[i] * fdata[i]
        # fsigma[t] = omega_tilde + lam_rev.dot(fdata[t - truncation:t])

        sigma2[t] = fsigma[t] ** (2.0 / power)
        data[t] = errors[t] * sqrt(sigma2[t])
        fdata[t] = abs(data[t]) ** power

    return data[truncation + burn:], sigma2[truncation + burn:]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def harch_simulate(int nobs,
                   int burn,
                   int max_lag,
                   double[:] parameters,
                   double[:] data,
                   double[:] sigma2,
                   double[:] errors,
                   int[:] lags):
    cdef:
        int i, j, t
        int[:] lag
        double param

    for t in range(max_lag, nobs + burn):
        sigma2[t] = parameters[0]
        for i in range(len(lags)):
            param = parameters[1 + i] / lags[i]
            for j in range(lags[i]):
                sigma2[t] += param * data[t - 1 - j] ** 2.0
        data[t] = errors[t] * sqrt(sigma2[t])

    return data[burn:], sigma2[burn:]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def garch_simulate_mc(int p,
                      int o,
                      int q,
                      double power,
                      int reps,
                      int nobs,
                      int burn,
                      double[::1] parameters,
                      double initial_value,
                      double[:, ::1] errors):
    cdef:
        int max_lag = max([p, o, q])
        int turns = nobs + burn
        int r, t
        double[:, ::1] sigma2 = np.zeros((turns, reps), dtype=np.float64)
        double[:, ::1] data = np.zeros((turns, reps), dtype=np.float64)
        double[:, ::1] fsigma = np.zeros((turns, reps), dtype=np.float64)
        double[:, ::1] fdata = np.zeros((turns, reps), dtype=np.float64)

    fsigma[:max_lag] = initial_value
    sigma2[:max_lag] = initial_value ** (2.0 / power)

    for r in range(reps):
        for t in range(max_lag):
            data[t, r] = errors[t, r] * sigma2[t, r] ** 0.5
            fdata[t, r] = abs(data[t, r]) ** power

    return np.asarray(
        _garch_simulate_mc(p, o, q, power, reps, nobs, burn, parameters, fsigma, sigma2, fdata, data, errors),
        dtype=np.float64
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:, ::1] _garch_simulate_mc(int p,
                                       int o,
                                       int q,
                                       double power,
                                       int reps,
                                       int nobs,
                                       int burn,
                                       double[::1] parameters,
                                       double[:, ::1] fsigma,
                                       double[:, ::1] sigma2,
                                       double[:, ::1] fdata,
                                       double[:, ::1] data,
                                       double[:, ::1] errors):
    cdef:
        int j, r, t, loc
        int max_lag = max([p, o, q])

    for r in range(reps):
        for t in range(max_lag, nobs + burn):
            loc = 0
            fsigma[t, r] = parameters[loc]
            loc += 1
            for j in range(p):
                fsigma[t, r] += parameters[loc] * fdata[t - 1 - j, r]
                loc += 1
            for j in range(o):
                fsigma[t, r] += parameters[loc] * fdata[t - 1 - j, r] * (data[t - 1 - j, r] < 0)
                loc += 1
            for j in range(q):
                fsigma[t, r] += parameters[loc] * fsigma[t - 1 - j, r]
                loc += 1

            sigma2[t, r] = fsigma[t, r] ** (2.0 / power)
            data[t, r] = errors[t, r] * sqrt(sigma2[t, r])
            fdata[t, r] = abs(data[t, r]) ** power

    return data[burn:]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def egarch_simulate_mc(int p,
                       int o,
                       int q,
                       int nobs,
                       int burn,
                       int reps,
                       double[::1] parameters,
                       double[:, ::1] sigma2,
                       double[:, ::1] lnsigma2,
                       double[:, ::1] errors):
    cdef:
        int r, max_lag = max([p, o, q])
        double[:, ::1] abserrors = np.abs(errors)
        double[:, ::1] data

    for r in range(reps):
        _egarch_simulate_mc(p, o, q, max_lag, nobs + burn, parameters, abserrors[:, r], lnsigma2[:, r], errors[:, r])

    sigma2 = np.exp(lnsigma2)
    data = errors * np.sqrt(sigma2)
    return np.asarray(data[burn:])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _egarch_simulate_mc(int p,
                         int o,
                         int q,
                         int start,
                         int end,
                         double[:] parameters,
                         double[:] abserrors,
                         double[:] lnsigma2,
                         double[:] errors):
    cdef:
        int j, t, loc
        double norm_const = (2 / pi) ** 0.5

    for t in range(start, end):
        lnsigma2[t] = parameters[0]
        loc = 1

        for j in range(p):
            lnsigma2[t] += parameters[loc] * (abserrors[t - 1 - j] - norm_const)
            loc += 1
        for j in range(o):
            lnsigma2[t] += parameters[loc] * errors[t - 1 - j]
            loc += 1
        for j in range(q):
            lnsigma2[t] += parameters[loc] * lnsigma2[t - 1 - j]
            loc += 1
