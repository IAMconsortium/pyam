.. _install:

Install
*******

Via Conda
~~~~~~~~~

.. code-block:: bash

    conda install -c conda-forge pyam

Via Pip
~~~~~~~

.. code-block:: bash

    pip install pyam-iamc

From Source
~~~~~~~~~~~

.. code-block:: bash

    pip install -e git+https://github.com/IAMconsortium/pyam.git#egg=pyam

Depedencies
~~~~~~~~~~~

Like any software project, we stand on the shoulders of giants. Our particular
giants include :code:`numpy` :cite:`numpy`, :code:`matplotlib`
:cite:`matplotlib`, and :code:`pandas` :cite:`pandas`. Explicit requirements are
fully enumerated below.

The required depedencies for :code:`pyam` are:

  .. program-output:: python -c 'import sys; sys.path.append("../.."); import requirements; requirements.display()'

The optional depedencies for :code:`pyam` are:

  .. include:: ../../requirements.txt
	  :start-line: 0
	  :literal:

The depedencies for building this documentation are:

  .. include:: ../requirements.txt
	  :start-line: 0
	  :literal:

References
**********

.. bibliography:: refs.bib
   :style: plain 
   :all:
