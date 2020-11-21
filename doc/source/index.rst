**pyam**: analysis and visualization of integrated-assessment scenarios
=======================================================================

Release v\ |version|.

|license| |pypi| |conda| |latest|

|pytest-38| |pytest-37| |codecov| |rtd|

|joss| |doi|

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-black
   :target: https://github.com/IAMconsortium/pyam/blob/master/LICENSE

.. |pypi| image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/

.. |conda| image:: https://anaconda.org/conda-forge/pyam/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |latest| image:: https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |pytest-38| image:: https://github.com/IAMconsortium/pyam/workflows/pytest%20(3.8)/badge.svg?branch=master
   :target: https://github.com/IAMconsortium/pyam/actions?query=workflow%3A%22pytest+%283.8%29%22+branch%3Amaster

.. |pytest-37| image:: https://github.com/IAMconsortium/pyam/workflows/pytest%20(3.7)/badge.svg?branch=master
   :target: https://github.com/IAMconsortium/pyam/actions?query=workflow%3A%22pytest+%283.7%29%22+branch%3Amaster

.. |codecov| image:: https://codecov.io/gh/IAMconsortium/pyam/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/IAMconsortium/pyam

.. |rtd| image:: https://readthedocs.org/projects/pyam-iamc/badge/?version=latest
   :target: https://pyam-iamc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.01095/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.01095

.. |doi| image:: https://zenodo.org/badge/113359260.svg
   :target: https://zenodo.org/badge/latestdoi/113359260

Overview
--------

The open-source Python package |pyam| :cite:`Gidden:2019:pyam`
provides a suite of tools and functions for analyzing and visualizing
input data (i.e., assumptions/parametrization) 
and results (model output) of integrated-assessment scenarios,
energy systems analysis, and sectoral studies.

Key features:
~~~~~~~~~~~~~

 - Simple analysis of timeseries data in the IAMC format (more about it `here`_) |br|
   with an interface similar in feel & style to the widely
   used :class:`pandas.DataFrame`
 - Advanced visualization and plotting functions (see the `gallery`_)
 - Features for scripted validation & processing of scenario data
   and results

The source code for |pyam| is available on `Github`_.

.. _`here`:
   data.html

.. _`gallery`:
   examples/index.html

.. _`Github`:
   https://github.com/IAMconsortium/pyam

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   install
   authors
   contributing
   data
   tutorials
   examples/index
   api

Copyright & License
-------------------

The development of the |pyam| package was started at the IIASA Energy Program,
with contributions from a number of `individuals & institutions`_ over the years.

The package is available under the open-source `Apache License`_.
Refer to the `NOTICE`_ in the GitHub repository for more information.

.. _individuals & institutions: authors.html

.. _Apache License: http://www.apache.org/licenses/LICENSE-2.0.html

.. _NOTICE: https://github.com/IAMconsortium/pyam/blob/master/NOTICE.md

Scientific reference
--------------------

.. bibliography:: _bib/index.bib
   :style: plain
   :all:
