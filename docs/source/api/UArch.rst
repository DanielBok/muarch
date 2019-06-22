UArch
~~~~~

:class:`UArch` is short for Univariate ARCH models and :class:`MUArch` stands for multiple (or many) Univariate ARCH models. In essence :class:`MUArch`, is a list of many :class:`UArch` models. This helps when you need to simulate many univariate ARCH models together. Also, it is helpful when you need to specify the marginals as in a Copula-GARCH model.

.. autoclass:: muarch.uarch.UArch
    :members:
    :inherited-members:

    .. automethod:: __init__
