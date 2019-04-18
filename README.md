MUArch
======

###### Continuous Integration

[![Build Status](https://travis-ci.com/DanielBok/muarch.svg?branch=master)](https://travis-ci.com/DanielBok/muarch)
[![Build status](https://ci.appveyor.com/api/projects/status/i6wylcc8syvdbkih?svg=true)](https://ci.appveyor.com/project/DanielBok/muarch)

###### Documentation

[![Documentation Status](https://readthedocs.org/projects/muarch/badge/?version=latest)](https://muarch.readthedocs.io/en/latest/?badge=latest)

###### Coverage

[![Coverage Status](https://coveralls.io/repos/github/DanielBok/muarch/badge.svg?branch=master)](https://coveralls.io/github/DanielBok/muarch?branch=master)

## Installing

Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/) and on conda.

This is a wrapper on top of Kevin Sheppard's [ARCH](https://github.com/bashtage/arch) package. The purpose of which are to:  

1. Enable faster Monte Carlo simulation
2. Simulate innovations through copula marginals

In the package, there are 2 classes to aid you - `UArch` and `MUArch`. The `UArch` class can be defined using a similar API to `arch_model` in the original `arch` package. The `MUArch` is a collection of these `UArch` models. 

Thus, if you have a function that generates uniform marginals, like a copula, you can create a dependence structure among the different marginals when simulating the GARCH processes.

If you need a copula package, I have one [here](https://github.com/DanielBok/copulae). :)

Example
-------

I'll list out a simple procedure to do AR-GARCH-Copula simulations.

```python
from muarch import MUArch, UArch
from muarch.datasets import load_etf
from copulae import NormalCopula


returns = load_etf()  # load returns data
num_assets = returns.shape[1]

# sets up a MUArch model collection where each model defaults to 
# mean: AR(1)
# vol: GARCH(1, 1)
# dist: normal 
models = MUArch(num_assets, mean='AR', lags=1) 

# set first model to AR(1)-GARCH(1, 1) with skewt innovations  
models[0] = UArch('AR', lags=1, dist='skewt')  

# fit model, if you get complaints regarding non-convergence, you can scale the data up 
# using the scale parameter in the UArch or MUArch. i.e. UArch(..., scale=100). This will
# reduce numerical errors. Don't worry, I'll rescale the simulation values subsequently
models.fit(returns)

# Usually you'll want to fit the residuals to the copula, use the copula to generate the
# residuals and subsequently transform it back to returns 

residuals = models.residuals() # defaults to return the standardized residuals


cop = NormalCopula(dim=num_assets) # use a normal copula, you could of course use a TCopula
cop.fit(residuals)

# simulate 10 steps into the future, over 4 repetitions. This will return a (10 x 4 x 3) array
models.simulate_mc(10, 4, custom_dist=cop.random)
```

Future Works
------------

This is actually a temporary hack so that others can do GARCH copula simulation. Another issue is that an ARFIMA mean model is not so easily specified (and simulated from) with the original `arch` package. You could specify an ARFIMA (or even just an ARMA model for the matter), fit it separately then use the residuals to fit a zero-mean model (pure GARCH). However, in such a way, the simulation is not so straightforward as you'll have to stitch the simulations from GARCH process and the mean model process back.
