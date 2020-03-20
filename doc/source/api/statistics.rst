.. currentmodule:: pyam

The **Statistics** class
========================

This class provides a wrapper for generating descriptive summary statistics
of timeseries data from a scenario ensemble.
It internally uses the :meth:`pandas.DataFrame.describe` function
and hides the tedious work of filters, groupbys and merging of dataframes.

.. autoclass:: Statistics
   :members:
