
Data Model
----------

Timeseries data
^^^^^^^^^^^^^^^

A `pyam.IamDataFrame` is a wrapper for two `pandas.DataFrame` instances:

 - `data`: The data table is a dataframe containing the timeseries data in
   "long format". It has the columns `pyam.LONG_IDX = ['model', 'scenario',
   'region', 'unit', 'year', 'value']`.

 - `meta`: The meta table is a dataframe containing categorisation and
   descriptive indicators. It has the index `pyam.META_IDX = ['model',
   'scenario']`.

The standard output format is the IAMC-style "wide format", see the example
above. This format can be accessed using `pd.IamDataFrame.timeseries()`,
which returns a `pandas.DataFrame` with the index `pyam.IAMC_IDX = ['model',
'scenario', 'region', 'variable', 'unit']` and the years as columns.

Filtering
^^^^^^^^^

The `pyam` package provides two methods for filtering timeseries data:

An existing IamDataFrame can be filtered using
`pyam.IamDataFrame.filter(col=...)`_, where `col` can be any column of the
`data` table (i.e., `['model', 'scenario', 'region', 'unit', 'year']) or any
column of the `meta` table. The returned object is a new `pyam.IamDataFrame`
instance.

A `pandas.DataFrame` with columns or index `['model', 'scenario']` can be
filtered by any `meta` columns from a `pyam.IamDataFrame` using
`pyam.filter_by_meta(data, df, col=..., join_meta=False)`_. The returned
object is a `pandas.DataFrame` downselected to those models-and-scenarios where
the `meta` column satisfies the criteria given by `col=...` .
Optionally, the `meta` columns are joined to the returned dataframe.

 .. _`pyam.IamDataFrame.filter(col=...)` : IamDataFrame.html#pyam.IamDataFrame.filter

 .. _`pyam.filter_by_meta(data, df, col=..., join_meta=False)` : pyam_functions.html
