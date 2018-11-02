.. currentmodule:: pyam

Python API
**********

Class IamDataFrame
~~~~~~~~~~~~~~~~~~

.. autoclass:: IamDataFrame
   :members:

Class OpenSCMDataFrame
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OpenSCMDataFrame
   :members:

Useful `pyam` functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: filter_by_meta

.. autofunction:: cumulative

.. autofunction:: fill_series

Class Statistics
~~~~~~~~~~~~~~~~

This class provides a wrapper for generating descriptive summary statistics
for timeseries data using various groupbys or filters.
It uses the `pandas.describe()`_ function internally
and hides the tedious work of filters, groupbys and merging of dataframes.

.. _`pandas.describe()` : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html

.. autoclass:: Statistics
   :members:
