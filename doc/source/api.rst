.. currentmodule:: pyam

API reference
=============

This page gives an overview of the public |pyam| features, objects, functions
and methods.

.. toctree::
   :maxdepth: 1

   api/io
   api/general
   api/iamdataframe
   api/filtering
   api/iiasa
   api/statistics
   api/timeseries
   api/variables


**Intersphinx mapping**

To use sphinx.ext.intersphinx_ for generating automatic links from your project
to the documenation of |pyam| classes and functions, please add the following
to your project's :code:`conf.py`:

.. _sphinx.ext.intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

.. code-block:: python

   intersphinx_mapping = {
       'pyam': ('https://pyam-iamc.readthedocs.io/en/stable/', None),
   }
