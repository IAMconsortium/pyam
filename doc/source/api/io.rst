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

The frictionless Data Package
-----------------------------

The |pyam| package supports reading and writing to the
`frictionless Data Package <https://frictionlessdata.io>`_.

.. autofunction:: read_datapackage

.. automethod:: IamDataFrame.to_datapackage
   :noindex:
