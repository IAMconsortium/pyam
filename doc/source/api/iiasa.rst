.. currentmodule:: pyam.iiasa

The **Connection** class
========================

IIASA's ixmp Scenario Explorer infrastructure implements a RestAPI
to directly query the database server connected to an explorer instance.
See https://software.ene.iiasa.ac.at/ixmp-server for more information.

The |pyam| package uses this interface to read timeseries data as well as
categorization and quantitative indicators.
The data is returned as an :class:`IamDataFrame`.
See `this tutorial <../tutorials/iiasa_dbs.html>`_ for more information.

.. autoclass:: Connection
   :members:

.. autofunction:: read_iiasa
   :noindex:

.. autofunction:: set_config