.. _install:

Installation
============

Conda
-----

https://anaconda.org/conda-forge/pyam

.. code-block:: bash

    conda install -c conda-forge pyam

Pypi
----

https://pypi.org/project/pyam-iamc/

.. warning::  The pyam package is distributed as "pyam-iamc" on pypi.

.. code-block:: bash

    pip install pyam-iamc

Installing from source
----------------------

|pyam| can also be installed from source.

.. code-block:: bash

    pip install -e git+https://github.com/IAMconsortium/pyam.git#egg=pyam-iamc

Dependencies
------------

Like any software project, we stand on the shoulders of giants. Our particular
giants include **pandas** :cite:`pandas`, **matplotlib** :cite:`matplotlib`,
and **numpy** :cite:`numpy`.
See the `setup-configuration`_ for more information.

.. _`setup-configuration`: https://github.com/IAMconsortium/pyam/blob/main/setup.cfg