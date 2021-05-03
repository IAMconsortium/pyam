.. currentmodule:: pyam

Data resources integration
==========================

Connecting to an IIASA Scenario Explorer instance
-------------------------------------------------

IIASA's ixmp Scenario Explorer infrastructure implements a RestAPI
to directly query the database server connected to an explorer instance.
See https://software.ece.iiasa.ac.at/ixmp-server for more information.

The |pyam| package uses this interface to read timeseries data as well as
categorization and quantitative indicators.
The data is returned as an :class:`IamDataFrame`.
See `this tutorial <../tutorials/iiasa_dbs.html>`_ for more information.

.. autofunction:: read_iiasa

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
