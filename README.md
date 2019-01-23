pyam: a Python toolkit for Integrated Assessment Modeling
=========================================================

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
in [the first tutorial](doc/source/tutorials/pyam_first_steps.ipynb)
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

Install
-------

For basic instructions see our
[website](http://software.ene.iiasa.ac.at/pyam/install.html).

To install from source after cloning this repository, simply run

```
pip install -e .
```

### Development

To setup a development environment,

```
# pyam can be replaced with any other name
# you don't have to specify your python version if you don't want
conda create --name pyam pip python=X.Y.Z
conda activate pyam  # may be source activate pyam or just activate pyam
pip install -e .[tests,docs,deploy]
# install other required packages (e.g. on a Unix like system)
conda install -c conda-forge $(cat requirements.txt)
# by hand also works e.g.
conda install -c conda-forge cartopy geopandas
# if you want to write notebooks
pip install notebook nbconvert jupyter_contrib_nbextensions
```

To check everything has installed correctly,

```
pytest tests
```

All the tests should pass.
