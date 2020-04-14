.. currentmodule:: pyam

Input/output
============

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

The frictionless Data Package
-----------------------------

The |pyam| package supports reading and writing to the
`frictionless Data Package <https://frictionlessdata.io>`_.

.. autofunction:: read_datapackage

.. automethod:: IamDataFrame.to_datapackage
   :noindex:

Connecting to an IIASA scenario explorer instance
-------------------------------------------------

IIASA's ixmp scenario explorer infrastructure implements a RestAPI
to directly query the database server connected to an explorer instance.
See https://software.ene.iiasa.ac.at/ixmp-server for more information.

The |pyam| package uses this interface to read timeseries data as well as
categorization and quantitative indicators.
The data is returned as an :class:`IamDataFrame`.
See `this tutorial <../tutorials/iiasa_dbs.html>`_ for more information.

.. autofunction:: read_iiasa