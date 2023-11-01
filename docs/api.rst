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
   api/slice
   api/filtering
   api/compute
   api/plotting
   api/iiasa
   api/statistics
   api/testing
   api/timeseries
   api/variables

Logging behaviour in Jupyter notebooks
--------------------------------------

The |pyam| package wants to provide sensible defaults for users unfamiliar with setting
up python's logging library (`read more`_), and therefore will add a streamhandler if
(and only if) it determines to be running within a notebook.

.. _`read more` : https://realpython.com/python-logging/#basic-configurations

Intersphinx mapping
-------------------

To use sphinx.ext.intersphinx_ for generating automatic links from your project
to the documentation of |pyam| classes and functions, please add the following
to your project's :code:`conf.py`:

.. _sphinx.ext.intersphinx : https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

.. code-block:: python

   intersphinx_mapping = {
       'pyam': ('https://pyam-iamc.readthedocs.io/en/stable/', None),
   }
