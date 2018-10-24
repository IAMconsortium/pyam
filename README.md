pyam: a Python toolkit for Integrated Assessment Modeling
=========================================================

.. image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/
   
.. image:: https://img.shields.io/pypi/l/pyam-iamc.svg
    :target: https://pypi.python.org/pypi/pyam-iamc

.. image:: https://circleci.com/gh/IAMconsortium/pyam.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/IAMconsortium/pyam

.. image:: https://travis-ci.org/IAMconsortium/pyam.svg?branch=master
   :target: https://travis-ci.org/IAMconsortium/pyam
   
.. image:: https://ci.appveyor.com/api/projects/status/github/IAMconsortium/pyam?svg=true&passingText=passing&failingText=failing&pendingText=pending
      :target: https://ci.appveyor.com/project/IAMconsortium/pyam

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.802832.svg
   :target: https://doi.org/10.5281/zenodo.802832

**Documentation: http://software.ene.iiasa.ac.at/pyam**
**Questions? Start a discussion on our [listserv](https://groups.google.com/forum/#!forum/pyam)**

Overview and scope
------------------

The ``pyam`` package provides a range of diagnostic tools and functions
for analyzing and working with IAMC-format timeseries data.

Features:
- Summary of models, scenarios, variables, and regions included in a snapshot.
- Display of timeseries data as pandas.DataFrame with IAMC-specific filtering
  options.
- Simple visualization and plotting functions.
- Diagnostic checks for non-reported variables or timeseries data to identify
  outliers and potential reporting issues.
- Categorization of scenarios according to timeseries data or meta-identifiers
  for further analysis.

The package can be used with timeseries data that follows the data template
convention of the [Integrated Assessment Modeling Consortium](http://www.globalchange.umd.edu/iamc/) (IAMC).
An illustrative example is shown below;
see [data.ene.iiasa.ac.at/database](http://data.ene.iiasa.ac.at/database/)
for more information.

| **model**    | **scenario** | **region** | **variable**   | **unit** | **2005** | **2010** | **2015** |
|--------------|--------------|------------|----------------|----------|----------|----------|----------|
| MESSAGE V.4  | AMPERE3-Base | World      | Primary Energy | EJ/y     |    454.5 |    479.6 |      ... |
| ...          | ...          | ...        | ...            | ...      |      ... |      ... |      ... |


Tutorial
--------

A comprehensive tutorial for the basic functions is included
in [tutorial/pyam_first_steps](tutorial/pyam_first_steps.ipynb)
using a partial snapshot of the IPCC AR5 scenario database.

Documentation
-------------

The documentation pages can be built locally.
See the instruction in [doc/README](doc/README.md).

Authors
-------

This package was developed and is currently maintained
by Matthew Gidden ([@gidden](https://github.com/gidden))
and Daniel Huppmann ([@danielhuppmann](https://github.com/danielhuppmann/)).

License
-------

Copyright 2017-2018 IIASA Energy Program

The ``pyam`` package is licensed
under the Apache License, Version 2.0 (the "License");
see [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for details.

Python dependencies
-------------------

0. `pandas` v0.21.0 or higher
0. `matplotlib`
0. `seaborn`
0. `geopandas` (optional)
0. `cartopy` (optional)

Documentation Building Depedencies
----------------------------------

0. `Sphinx <http://sphinx-doc.org/>`_ v1.1.2 or higher
0. `sphinxcontrib.bibtex`
0. `sphinxcontrib-fulltoc`
0. `sphinx-gallery`

Installation instructions
-------------------------

0. Fork this repository and clone the forked repository (`<user>/pyam`)
   to your machine. To fork the repository, look for the fork icon in the top
   right at [iiasa/pyam](https://github.com/iiasa/pyam).
   Add `iiasa/pyam` as `upstream` to your clone.

   *We recommend* [GitKraken](https://www.gitkraken.com/) *for users*
   *who prefer a graphical user interface application*
   *to work with Github (as opposed to the command line).*

### Windows Users

0. Double click on `install.bat` in the local folder where you cloned your fork.

### *nix Users

0. In a command prompt, execute the following command

    ```
    python setup.py install
    ```

