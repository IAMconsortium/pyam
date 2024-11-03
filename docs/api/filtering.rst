.. currentmodule:: pyam

Filtering and slicing
=====================

Arguments for filtering an :class:`IamDataFrame`
------------------------------------------------

The |pyam| package provides several methods to filter an :class:`IamDataFrame` by its
(timeseries) **data** or **meta** values. Read more about the `Data Model <data.html>`_
that is implemented by an :class:`IamDataFrame`.

The following arguments are available for filtering and can be combined as needed:

Index
^^^^^
- A *column* of the :attr:`IamDataFrame.index`
  (usually '**model**' and '**scenario**'): string or list of strings
- '**index**': list of model/scenario-tuples or a :class:`pandas.MultiIndex`

Timeseries data coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Any *column* of the :attr:`IamDataFrame.coordinates <pyam.IamDataFrame.coordinates>`
  ('**region**', '**variable**', '**unit**'): string or list of strings
- '**measurand**': a tuple (or list of tuples) of '*variable*' and '*unit*'
- '**depth**': the "depth" of entries in the '*variable*' column (number of '|')
- '**level**': the "depth" of entries in the '*variable*' column (number of '|'),
  excluding the strings in the '*variable*' argument (if given)
- '**year**': takes an integer (int/:class:`numpy.int64`), a list of integers or
  a range. Note that the last year of a range is not included,
  so ``range(2010, 2015)`` is interpreted as ``[2010, ..., 2014]``
- '**time_domain**': can be 'year' or 'datetime'
- Arguments for filtering by :class:`datetime.datetime` or :class:`numpy.datetime64`
  ('**month**', '**hour**', '**time**')

Meta indicators and other attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Any *column* of the :attr:`IamDataFrame.meta <pyam.IamDataFrame.meta>` dataframe:
  string, integer, float, or list of these
- '**exclude**' (see :attr:`IamDataFrame.exclude <pyam.IamDataFrame.exclude>`): boolean

.. note::

    In any string filters, '*' is interpreted as wildcard, unless the keyword argument
    *regexp=True* is used; in this case, strings are treated as
    `regular expressions <https://docs.python.org/3/library/re.html>`_.

Methods for filtering and slicing an :class:`IamDataFrame`
----------------------------------------------------------

.. automethod:: pyam.IamDataFrame.filter
   :noindex:

.. automethod:: pyam.IamDataFrame.slice
   :noindex:

The **IamSlice** class
----------------------

This class is an auxiliary feature to streamline the implementation of the
:meth:`IamDataFrame.filter` method.

.. autoclass:: pyam.slice.IamSlice
   :members: dimensions, time, info

Filtering using a proxy :class:`IamDataFrame`
---------------------------------------------

|pyam| includes a function to directly filter a :class:`pandas.DataFrame`
with appropriate columns or index dimensions (i.e.,'*model*' and '*scenario*') using
an :class:`IamDataFrame` and keyword arguments similar to :meth:`IamDataFrame.filter`.

.. autofunction:: pyam.filter_by_meta
