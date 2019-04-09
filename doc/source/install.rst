.. _install:

Install
*******

The most basic installation of :code:`pyam` is trivial.

Via Conda
~~~~~~~~~

.. code-block:: bash

    conda install -c conda-forge pyam

Via Pip
~~~~~~~

:code:`pyam` can also be installed via pip.

.. code-block:: bash

    pip install pyam-iamc

By default, this will not install the optional extras (see `Depedencies`_).
To install the optional extras, execute the following command.

.. code-block:: bash

    pip install pyam-iamc[geoplots]

As a word of warning, if you want to make geospatial plots this may not be the simplest route.
The reason is that many geospatial plotting libraries, including :code:`cartopy`, may not install properly with pip because pip cannot handle the installation of the complicated c-level libraries required.

From Source
~~~~~~~~~~~

:code:`pyam` can also be installed from source.
As with installation via pip, if you want to make geospatial plots this may not be the simplest route.
You will have to handle the installation of any c-level libraries which are required for geospatial plotting yourself.

.. code-block:: bash

    pip install -e git+https://github.com/IAMconsortium/pyam.git#egg=pyam

Depedencies
~~~~~~~~~~~

Like any software project, we stand on the shoulders of giants. Our particular
giants include :code:`numpy` :cite:`numpy`, :code:`matplotlib`
:cite:`matplotlib`, and :code:`pandas` :cite:`pandas`. Explicit requirements are
fully enumerated below.

The required depedencies for :code:`pyam` are:

  .. program-output:: python -c 'import sys; sys.path.append("../.."); import setup; print("\n".join([r for r in setup.REQUIREMENTS]))'

The optional depedencies for :code:`pyam` are:

  .. program-output:: python -c 'import sys; sys.path.append("../.."); import setup; print("\n".join([r for r in setup.EXTRA_REQUIREMENTS["geoplots"]]))'

The depedencies for building this documentation are:

  .. include:: ../requirements.txt
	  :start-line: 0
	  :literal:

References
**********

.. bibliography:: refs.bib
   :style: plain
   :all:
