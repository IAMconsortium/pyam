pyam: a Python toolkit for Integrated Assessment Modeling
=========================================================

Overview and scope
------------------

The ``pyam`` package provides a range of diagnostic tools and functions
for analyzing and visualising scenario data in the IAMC timeseries format.

Features:
 - Summary of models, scenarios, variables, and regions included in a snapshot.
 - Display of timeseries data as `pandas.DataFrame`_
   with IAMC-specific filtering options.
 - Advanced visualization and plotting functions.
 - Diagnostic checks for non-reported variables or timeseries values
   to analyze and validate scenario data.
 - Categorization of scenarios according to timeseries data
   or metadata for further analysis.

The package can be used with data that follows the data template convention
of the `Integrated Assessment Modeling Consortium`_ (IAMC).
An illustrative example is shown below;
see `data.ene.iiasa.ac.at/database`_ for more information.

.. _`pandas.DataFrame`:
   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

.. _`Integrated Assessment Modeling Consortium`:
   http://www.globalchange.umd.edu/iamc/

.. _`data.ene.iiasa.ac.at/database`: http://data.ene.iiasa.ac.at/database/

============  =============  ==========  ==============  ========  ========  ========  ========
**Model**     **Scenario**   **Region**  **Variable**    **Unit**  **2005**  **2010**  **2015**
============  =============  ==========  ==============  ========  ========  ========  ========
MESSAGE V.4   AMPERE3-Base   World       Primary Energy  EJ/y      454.5     479.6     ...
...           ...            ...         ...             ...       ...       ...       ...
============  =============  ==========  ==============  ========  ========  ========  ========


License and source code repository
----------------------------------

The `pyam` package is licensed under an `APACHE 2.0 open-source license`_.
See the `LICENSE`_ file included in this repository for the full text.
The source code is available on `github.com/IAMconsortium/pyam`_.

.. _`APACHE 2.0 open-source license`: http://www.apache.org/licenses/LICENSE-2.0

.. _`LICENSE`: https://github.com/IAMconsortium/pyam/blob/master/LICENSE

.. _`github.com/IAMconsortium/pyam`: https://github.com/IAMconsortium/pyam


The `pyam` data model
---------------------

A `pyam.IamDataFrame` is a wrapper for two `pandas.DataFrame` instances:

 - `data`: The data table is a dataframe containing the timeseries data in
   "long format". It has the columns `pyam.LONG_IDX = ['model', 'scenario',
   'region', 'unit', 'year', 'value']`.

 - `meta`: The meta table is a dataframe containing categorisation and
   descriptive indicators. It has the index `pyam.META_IDX = ['model',
   'scenario']`.

The standard output format is the IAMC-style "wide format", see the example
above. This format can be accessed using `pd.IamDataFrame.timeseries()`,
which returns a `pandas.DataFrame` with the index `pyam.IAMC_IDX = ['model',
'scenario', 'region', 'variable', 'unit']` and the years as columns.

`pyam` documentation
--------------------

See `this guide`_ for guidelines on NumPy/SciPy Documentation conventions.

.. _`this guide`: 
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

.. toctree:: 
   :maxdepth: 2 
 
   IamDataFrame
   timeseries
   examples/index

