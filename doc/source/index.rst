**pyam**: analysis and visualization of assessment models
=========================================================

.. |br| raw:: html

    <br>

Release v\ |version|.

|rtd| |pypi| |conda| |license| |latest|

|travis| |appveyor| |coveralls|

|joss| |doi|

.. |rtd| image:: https://readthedocs.org/projects/pyam-iamc/badge/?version=latest
   :target: https://pyam-iamc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |pypi| image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/

.. |conda| image:: https://anaconda.org/conda-forge/pyam/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |license| image:: https://anaconda.org/conda-forge/pyam/badges/license.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |latest| image:: https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/pyam

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

The **pyam** Python package provides a range of diagnostic tools and functions
for analyzing and visualizing data from your favorite assessment model(s).

The source code for **pyam** is available on `Github`_.

.. _`Github`:
   https://github.com/IAMconsortium/pyam

.. _`groups.google.com/d/forum/pyam` :
   https://groups.google.com/d/forum/pyam

Overview
--------

Some of the **pyam** features include:
 - Easily filter and manipulate data in the `IAMC`_ timeseries format
 - An interface similar in feel and style to `pandas.DataFrame`_
 - Advanced visualization and plotting functions.
 - Diagnostic checks for non-reported variables or timeseries values
   to analyze and validate scenario data.
 - Categorization of scenarios according to timeseries data
   or metadata for further analysis.

.. _`IAMC`:
   https://data.ene.iiasa.ac.at/database

.. _`pandas.DataFrame`:
   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

After installing, check out our tutorials or our plotting gallery to get
started.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   data
   tutorials
   examples/index
   api

.. include:: ../../CONTRIBUTING.rst

License
-------

:code:`pyam` is available under the open source `Apache License`_.

.. _Apache License: http://www.apache.org/licenses/LICENSE-2.0.html
