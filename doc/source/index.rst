**pyam**: analysis and visualization of assessment models
=========================================================

.. |br| raw:: html

    <br>

Release v\ |version|.

.. image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/

.. image:: https://anaconda.org/conda-forge/pyam/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyam   

.. image:: https://anaconda.org/conda-forge/pyam/badges/license.svg
   :target: https://anaconda.org/conda-forge/pyam

.. image:: https://zenodo.org/badge/113359260.svg
   :target: https://zenodo.org/badge/latestdoi/113359260

.. image:: https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/pyam

.. image:: https://circleci.com/gh/IAMconsortium/pyam.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/IAMconsortium/pyam

.. image:: https://travis-ci.org/IAMconsortium/pyam.svg?branch=master
   :target: https://travis-ci.org/IAMconsortium/pyam
   
.. image:: https://ci.appveyor.com/api/projects/status/qd4taojd2vkqoab4/branch/master?svg=true&passingText=passing&failingText=failing&pendingText=pending
      :target: https://ci.appveyor.com/project/gidden/pyam/branch/master

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

.. _Apache LIcense: http://www.apache.org/licenses/LICENSE-2.0.html

