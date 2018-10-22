**pyam**: analysis and visualization of assessment models
=========================================================


Release v\ |version|.

.. image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/
   
.. image:: https://img.shields.io/pypi/l/pyam-iamc.svg
    :target: https://pypi.python.org/pypi/pyam-iamc

.. image:: https://circleci.com/gh/IAMconsortium/pyam.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/IAMconsortium/pyam

.. image:: https://travis-ci.org/IAMconsortium/pyam.svg?branch=master
   :target: https://travis-ci.org/IAMconsortium/pyam
   
.. image:: https://ci.appveyor.com/api/projects/status/github/IAMconsortium/pyam?svg=true&passingText=passing&failingText=failing&pendingText=pending
      :target: https://ci.appveyor.com/project/IAMconsortium/pyam

.. image:: https://coveralls.io/repos/github/IAMconsortium/pyam/badge.svg?branch=master
    :target: https://coveralls.io/github/IAMconsortium/pyam?branch=master

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.802832.svg
   :target: https://doi.org/10.5281/zenodo.802832

The **pyam** Python package provides a range of diagnostic tools and functions
for analyzing and visualising data in the IAMC timeseries format.

Overview
--------

Features:
 - Easily filter and manipulate `IAMC`_ compliant data
 - An interface similar in feel and style to `pandas.DataFrame`_
 - Advanced visualization and plotting functions.
 - Diagnostic checks for non-reported variables or timeseries values
   to analyze and validate scenario data.
 - Categorization of scenarios according to timeseries data
   or metadata for further analysis.

An illustrative example of IAMC-style data is shown below;
see https://data.ene.iiasa.ac.at/database for more information.

.. _`pandas.DataFrame`:
   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

.. _`IAMC`:
   http://www.globalchange.umd.edu/iamc/

============  =============  ==========  ==============  ========  ========  ========  ========
**Model**     **Scenario**   **Region**  **Variable**    **Unit**  **2005**  **2010**  **2015**
============  =============  ==========  ==============  ========  ========  ========  ========
MESSAGE V.4   AMPERE3-Base   World       Primary Energy  EJ/y      454.5     479.6     ...
...           ...            ...         ...             ...       ...       ...       ...
============  =============  ==========  ==============  ========  ========  ========  ========

Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   data
   api
   examples/index

