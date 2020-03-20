.. currentmodule:: pyam

Timeseries utilities
====================

The |pyam| package includes several utility functions for working
with timeseries data formatted as :class:`pandas.Series` that have
the time dimension as index.

.. warning::

    Not all **pyam** functions currently support continuous-time formats.
    Please reach out via our `mailing list or GitHub issues`_
    if you are not sure whether your use case is supported.

.. _`mailing list or GitHub issues`: ../contributing.html

.. autofunction:: cumulative

.. autofunction:: fill_series

.. autofunction:: cross_threshold
