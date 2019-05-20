Welcome to MUArch's documentation!
==================================

This is a wrapper on top of Kevin Sheppard's `ARCH <https://github.com/bashtage/arch>`_ package. The purpose of which are to:

1. Enable faster Monte Carlo simulation
2. Simulate innovations through copula marginals

In the package, there are 2 classes to aid you - :class:`UArch` and :class:`MUArch`. The :class:`UArch` class can be defined using a similar API to :func:`arch_model` in the original :class:`arch` package. The :class:`MUArch` is a collection of these :class:`UArch` models.

Thus, if you have a function that generates uniform marginals, like a copula, you can create a dependence structure among the different marginals when simulating the GARCH processes.

If you need a copula package, I have one `here <https://github.com/DanielBok/copulae>`_. :)

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

        Getting Started <getting-started>
        Examples <examples/index>
        API <api/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
