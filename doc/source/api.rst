.. currentmodule:: pyam

API Reference
=============

This page gives an overview of the public |pyam| features, objects, functions
and methods.

.. toctree::
   :maxdepth: 1

   api/io
   api/general
   api/iamdataframe
   api/database
   api/filtering
   api/compute
   api/plotting
   api/iiasa
   api/statistics
   api/testing
   api/timeseries
   api/variables

**Notebook logging behaviour**

|pyam| wants to provide sensible defaults for users unfamiliar with
`setting up python's logging library <https://realpython.com/python-logging/#basic-configurations>`_,
and therefore will provide a basic configuration by invoking

.. code-block:: python

   import logging
   logging.basicConfig(level="INFO", format="%(name)s - %(levelname)s: %(message)s")

if (and only if):

1. it determines to be running within a notebook, and
2. logging is still *unconfigured by the time the first logging message by |pyam| is to be emitted*.

**Intersphinx mapping**

To use sphinx.ext.intersphinx_ for generating automatic links from your project
to the documenation of |pyam| classes and functions, please add the following
to your project's :code:`conf.py`:

.. _sphinx.ext.intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

.. code-block:: python

   intersphinx_mapping = {
       'pyam': ('https://pyam-iamc.readthedocs.io/en/stable/', None),
   }
