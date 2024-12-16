.. currentmodule:: pyam

Input/output file formats
=========================

DataFrames and xlsx/csv files
-----------------------------

A :class:`pandas.DataFrame` or a path to an :code:`xlsx` or :code:`csv`
with data in the required structure (i.e., index/columns) can be imported
directly by initializing an :class:`IamDataFrame` - see
`this tutorial <../tutorials/data_table_formats.html>`_ for more information.

Exporting to these formats is implemented via the following functions:

.. automethod:: IamDataFrame.as_pandas
   :noindex:

.. automethod:: IamDataFrame.to_excel
   :noindex:

.. automethod:: IamDataFrame.to_csv
   :noindex:

Integration with netcdf files
-----------------------------

`NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ is a powerful file format that
can efficiently store multiple scientific variables sharing the same dimensions.
In climate science, data such as temperature, precipitation and radiation can be stored
in four dimensions: a time dimension and three spatial dimensions (latitude, longitude,
altitude).

The |pyam| package supports reading and writing to netcdf files that have the following
structure:

- **Timeseries data** are stored such that each variable (in the sense of the IAMC
  format) is a separate netcdf-data-variable with the following dimensions *time*,
  *model*, *scenario* and *region*. The *unit* is given as an attribute of the data
  variable. The *long_name* attribute is used as the variable name in the
  :class:`IamDataFrame`. The *time* dimension can be either a datetime format or given
  as years (integer).

- **Meta indicators** are stored as netcdf-data-variables with the dimensions *model*
  and *scenario*.

.. autofunction:: read_netcdf

The frictionless Data Package
-----------------------------

The |pyam| package supports reading and writing to the
`frictionless Data Package <https://frictionlessdata.io>`_.

.. autofunction:: read_datapackage

.. automethod:: IamDataFrame.to_datapackage
   :noindex:
