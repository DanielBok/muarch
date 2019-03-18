MUArch Core API
===============

.. module:: muarch
.. py:currentmodule:: muarch

:class:`UArch` is short for Univariate ARCH models and :class:`MUArch` stands for multiple (or many) Univariate ARCH models. In essence :class:`MUArch`, is a list of many :class:`UArch` models. This helps when you need to simulate many univariate ARCH models together. Also, it is helpful when you need to specify the marginals as in a Copula-GARCH model.

UArch
~~~~~

.. autoclass:: UArch
    :members:
    :inherited-members:

MUArch
~~~~~~
.. autoclass:: MUArch
    :members:
    :inherited-members:
