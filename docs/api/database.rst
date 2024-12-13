.. currentmodule:: pyam

Data resources integration
==========================

Connecting to an IIASA database instance
----------------------------------------

IIASA's ixmp Scenario Explorer infrastructure implements a RestAPI
to directly query the database server connected to an explorer instance.
See https://docs.ece.iiasa.ac.at/ for more information.

The |pyam| package uses this interface to read timeseries data as well as
categorization and quantitative meta indicators.
The data is returned as an :class:`IamDataFrame`.
See `this tutorial <../tutorials/iiasa.html>`_ for more information.

.. autofunction:: read_iiasa

.. autofunction:: lazy_read_iiasa

Reading from an |ixmp4| platform
--------------------------------

The |pyam| package provides a simple interface to read timeseries data and meta
indicators from local or remote |ixmp4| platform instances.

.. autofunction:: read_ixmp4

Reading UNFCCC inventory data
-----------------------------

The package :class:`unfccc-di-api`
(`read the docs <https://unfccc-di-api.readthedocs.io/>`_)
provides an interface to the UNFCCC Data Inventory API
(`link <https://di.unfccc.int>`_).
The |pyam| package uses this package to query inventory data and
return the timeseries data directly as an :class:`IamDataFrame`.

.. autofunction:: read_unfccc

Connecting to World Bank data resources
---------------------------------------

The package :class:`pandas-datareader`
(`read the docs <https://pandas-datareader.readthedocs.io>`_)
implements a number of connections to publicly accessible data resources,
e.g., the `World Bank Open Data Catalog <https://datacatalog.worldbank.org>`_.
|pyam| provides a simple utility function to cast the queried timeseries data
directly as an :class:`IamDataFrame`.

.. autofunction:: read_worldbank
