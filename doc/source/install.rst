.. _install:

Installation
============

Via your favourite Python Package Manager
-----------------------------------------

Conda
~~~~~

.. code-block:: bash

    conda install -c conda-forge pyam

Pip
~~~

.. code-block:: bash

    pip install pyam-iamc

.. warning::  The pyam package is distributed as "pyam-iamc" on pypi.

Installing From Source
----------------------

|pyam| can also be installed from source.

.. code-block:: bash

    pip install -e git+https://github.com/IAMconsortium/pyam.git#egg=pyam

Dependencies
------------

Like any software project, we stand on the shoulders of giants. Our particular
giants include **pandas** :cite:`pandas`, **matplotlib** :cite:`matplotlib`,
and **numpy** :cite:`numpy`.
See the `setup-configuration`_ for more information.

.. _`setup-configuration`: https://github.com/IAMconsortium/pyam/blob/main/setup.cfg