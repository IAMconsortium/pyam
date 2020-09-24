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


Installing From Source
----------------------

|pyam| can also be installed from source.

.. code-block:: bash

    pip install -e git+https://github.com/IAMconsortium/pyam.git#egg=pyam

Dependencies
------------

Like any software project, we stand on the shoulders of giants. Our particular
giants include :code:`numpy` :cite:`numpy`, :code:`matplotlib`
:cite:`matplotlib`, and :code:`pandas` :cite:`pandas`. Explicit requirements are
fully enumerated below.

The required depedencies for |pyam| are:

  .. program-output:: python -c 'import sys; sys.path.insert(0, "../.."); from setup import REQUIREMENTS; print("\n".join([r for r in REQUIREMENTS]))'

References
----------

.. bibliography:: _bib/install.bib
   :style: plain
   :cited:
