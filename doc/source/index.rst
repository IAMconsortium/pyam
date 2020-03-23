**pyam**: analysis and visualization of integrated-assessment scenarios
=======================================================================

Release v\ |version|.

|pypi| |conda| |license| |latest|

|rtd| |travis| |appveyor| |coveralls|

|joss| |doi|

.. |pypi| image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/

.. |conda| image:: https://anaconda.org/conda-forge/pyam/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |license| image:: https://anaconda.org/conda-forge/pyam/badges/license.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |latest| image:: https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |rtd| image:: https://readthedocs.org/projects/pyam-iamc/badge/?version=latest
   :target: https://pyam-iamc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/IAMconsortium/pyam.svg?branch=master
   :target: https://travis-ci.org/IAMconsortium/pyam

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/qd4taojd2vkqoab4/branch/master?svg=true&passingText=passing&failingText=failing&pendingText=pending
   :target: https://ci.appveyor.com/project/gidden/pyam/branch/master

.. |coveralls| image:: https://coveralls.io/repos/github/IAMconsortium/pyam/badge.svg?branch=master
   :target: https://coveralls.io/github/IAMconsortium/pyam?branch=master

.. |joss| image:: http://joss.theoj.org/papers/356bc013105642ec4e94a3b951836cfe/status.svg
   :target: http://joss.theoj.org/papers/356bc013105642ec4e94a3b951836cfe

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

 - Simple analysis of timeseries data in the IAMC format (more about it `here`_)
   with an interface similar in feel & style to the widely
   used :py:class:`pandas.DataFrame`
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
   contributing
   data
   tutorials
   examples/index
   api

License
-------

|pyam| is available under the open source `Apache License`_.

.. _Apache License: http://www.apache.org/licenses/LICENSE-2.0.html

Scientific reference
--------------------

.. bibliography:: _bib/index.bib
   :style: plain
   :all:
