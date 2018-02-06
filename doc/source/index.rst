pyam: a Python toolkit for Integrated Assessment Modeling
=========================================================

Overview and scope
------------------

The ``pyam`` package provides a range of diagnostic tools and functions  
for analyzing and working with IAMC-format timeseries data. 

Features:
 - Summary of models, scenarios, variables, and regions included in a snapshot.
 - Display of timeseries data as pandas.DataFrame 
   with IAMC-specific filtering options.
 - Simple visualization and plotting functions.
 - Diagnostic checks for non-reported variables or timeseries data 
   to identify outliers and potential reporting issues.
 - Categorization of scenarios according to timeseries data 
   or meta-identifiers for further analysis.

The package can be used with data that follows the data template convention 
of the `Integrated Assessment Modeling Consortium`_ (IAMC).
An illustrative example is shown below; 
see `data.ene.iiasa.ac.at/database`_ for more information.

.. _`Integrated Assessment Modeling Consortium`:
   http://www.globalchange.umd.edu/iamc/

.. _`data.ene.iiasa.ac.at/database`: http://data.ene.iiasa.ac.at/database/

============  =============  ==========  ==============  ========  ========  ========  ========
**model**     **scenario**   **region**  **variable**    **unit**  **2005**  **2010**  **2015**
============  =============  ==========  ==============  ========  ========  ========  ========
MESSAGE V.4   AMPERE3-Base   World       Primary Energy  EJ/y      454.5     479.6     ... 
...           ...            ...         ...             ...       ...       ...       ...
============  =============  ==========  ==============  ========  ========  ========  ========


pyam documentation
------------------

See `this guide`_ for guidelines on NumPy/SciPy Documentation conventions.

.. _`this guide`: 
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

.. toctree:: 
   :maxdepth: 2 
 
   source/IamDataFrame
   source/timeseries
   source/examples/index

