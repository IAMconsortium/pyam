Data Model
==========

Scenario data following the IAMC format
---------------------------------------

.. figure:: _static/iamc_logo.jpg
   :width: 120px
   :align: right
   
   `IAMC website`_
   
.. _`IAMC Website`: http://www.globalchange.umd.edu/iamc/

Over the past decade, the Integrated Assessment Modeling Consortium (IAMC)
developed a standardised tabular timeseries format to exchange scenario data.
Previous high-level use cases include reports by the *Intergovernmental Panel
on Climate Change* (`IPCC`_) and model comparison exercises
within the *Energy Modeling Forum* (`EMF`_) hosted by Stanford University.

The table below shows a typical example of integrated-assessment scenario data
following the IAMC format from the `CD-LINKS`_ project.
The |pyam| package is geared for analysis and visualization of any scenario
data provided in this structure.

.. figure:: _static/iamc_template.png

   Illustrative example of IAMC-format timeseries data |br|
   via the `IAMC 1.5°C Scenario Explorer`_ (:cite:`Huppmann:2019:scenario-data`)

.. _`IAMC 1.5°C Scenario Explorer`: https://data.ene.iiasa.ac.at/iamc-1.5c-explorer

Refer to `data.ene.iiasa.ac.at/database`_ for more information on the
IAMC format and a full list of previous use cases.

.. _`IPCC`: https://www.ipcc.ch

.. _`EMF`: https://emf.stanford.edu

.. _`CD-LINKS`: https://www.cd-links.org

.. _`data.ene.iiasa.ac.at/database`: https://data.ene.iiasa.ac.at/database

The :code:`variable` column
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`variable` column implements a "semi-hierarchical" structure
using the :code:`|` character (*pipe*, not l or i) to indicate the *depth*.

Semi-hierarchical means that a hierarchy can be imposed, e.g., one can enforce
that the sum of :code:`Emissions|CO2|Energy` and :code:`Emissions|CO2|Other`
must be equal to :code:`Emissions|CO2`
(if there are no other :code:`Emissions|CO2|…` variables).
However, this is not mandatory, e.g., the sum of :code:`Primary Energy|Coal`,
:code:`Primary Energy|Gas` and :code:`Primary Energy|Fossil` should not be equal
to :code:`Primary Energy` because this would double-count fossil fuels.

Refer to the variable list in the documentation pages of the
`IAMC 1.5°C Scenario Explorer`_ to see the full list of variables used in the
recent *IPCC Special Report on Global Warming of 1.5 ºC* (`SR15`_).

.. _`SR15`: https://www.ipcc.ch/sr15/

The :code:`year` column
~~~~~~~~~~~~~~~~~~~~~~~

In its original design, the IAMC data format (see above) assumed that the
temporal dimension of any scenario data was restricted to full years
represented as integer values.

Two additional use cases are currently supported by :code:`pyam` in development
mode (beta):

 - using representative sub-annual timesteps

 - using continuous time via :class:`pandas.datetime`, replacing the name of
   the :code:`year` column by :code:`time`

Please reach out to the developers to get more information on this
ongoing work.

The :class:`pyam.IamDataFrame` class
------------------------------------

A :class:`pyam.IamDataFrame` instance is a wrapper for
two :class:`pandas.DataFrame` instances (read the `docs`_):

 - :code:`data`: The data table is a dataframe containing the timeseries data
   in "long format". It has the columns of the long data format :code:`['model',
   'scenario', 'region', 'unit', 'year', 'value']`.

 - :code:`meta`: The meta table is a dataframe containing categorisation and
   descriptive indicators. It has the index :code:`pyam.META_IDX = ['model',
   'scenario']`.

The standard output format is the IAMC-style "wide format", see the example
above. This format can be accessed using :meth:`pyam.IamDataFrame.timeseries`,
which returns a :class:`pandas.DataFrame` with the index :code:`pyam.IAMC_IDX =
['model', 'scenario', 'region', 'variable', 'unit']` and the years as columns.

.. _`docs`: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html

Filtering
---------

The |pyam| package provides two methods for filtering scenario data:

An existing `class`:IamDataFrame can be filtered using
:meth:`pyam.IamDataFrame.filter(col=...) <pyam.IamDataFrame.filter>`,
where :code:`col` can be any column of the
:code:`data` table (i.e., `['model', 'scenario', 'region', 'unit', 'year']`)
or any column of the :code:`meta` table. The returned object is
a new :class:`pyam.IamDataFrame` instance.

A :class:`pandas.DataFrame` with columns or index :code:`['model', 'scenario']`
can be filtered by any :code:`meta` columns from a :code:`pyam.IamDataFrame`
using :func:`pyam.filter_by_meta(data, df, col=..., join_meta=False) <pyam.filter_by_meta>`.
The returned object is a :class:`pandas.DataFrame` down-selected to those
models-and-scenarios where the :code:`meta` column satisfies the criteria given
by :code:`col=...` .
Optionally, the :code:`meta` columns are joined to the returned dataframe.

References
----------

.. bibliography:: _bib/data.bib
   :style: plain
   :cited:
