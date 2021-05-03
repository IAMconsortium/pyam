pyam: analysis & visualization <br /> of integrated-assessment scenarios
========================================================================

[![license](https://img.shields.io/badge/License-Apache%202.0-black)](https://github.com/IAMconsortium/pyam/blob/main/LICENSE)
[![pypi](https://img.shields.io/pypi/v/pyam-iamc.svg)](https://pypi.python.org/pypi/pyam-iamc/)
[![conda](https://anaconda.org/conda-forge/pyam/badges/version.svg)](https://anaconda.org/conda-forge/pyam)
[![latest](https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/pyam)

<!-- replace python version by dynamic reference to pypi once Python versions are configured there -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9-blue?logo=python&logoColor=white)](https://github.com/IAMconsortium/pyam)
[![pytest](https://github.com/IAMconsortium/pyam/actions/workflows/pytest.yml/badge.svg)](https://github.com/IAMconsortium/pyam/actions/workflows/pytest.yml)
[![ReadTheDocs](https://readthedocs.org/projects/pyam-iamc/badge/?version=latest)](https://pyam-iamc.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/IAMconsortium/pyam/branch/main/graph/badge.svg)](https://codecov.io/gh/IAMconsortium/pyam)

[![doi](https://zenodo.org/badge/113359260.svg)](https://zenodo.org/badge/latestdoi/113359260)
[![joss](https://joss.theoj.org/papers/10.21105/joss.01095/status.svg)](https://joss.theoj.org/papers/10.21105/joss.01095)
[![groups.io](https://img.shields.io/badge/groups.io-pyam-blue)](https://pyam.groups.io/g/forum)
[![slack](https://img.shields.io/badge/slack-@pyam-orange.svg?logo=slack)](https://pyam-iamc.slack.com)

****

Overview and scope
------------------

The open-source Python package **pyam** provides a suite of tools and functions
for analyzing and visualizing input data (i.e., assumptions/parametrization) 
and results (model output) of integrated-assessment models,
macro-energy scenarios, energy systems analysis, and sectoral studies.

### Key features

 - Simple analysis of scenario timeseries data with an interface similar in feel & style
   to the widely used [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
 - Advanced visualization and plotting functions
   (see the [gallery](https://pyam-iamc.readthedocs.io/en/stable/gallery/index.html))
 - Scripted validation and processing of scenario data and results

The comprehensive **documentation** is hosted on [Read the Docs](https://pyam-iamc.readthedocs.io)!

### Timeseries types & data formats

#### Yearly data

The pyam package was initially developed to work with the *IAMC template*,
a timeseries format for *yearly data* developed and used by the
[Integrated Assessment Modeling Consortium](https://www.iamconsortium.org) (IAMC).

| **model** | **scenario** | **region** | **variable**   | **unit** | **2005** | **2010** | **2015** |
|-----------|--------------|------------|----------------|----------|----------|----------|----------|
| MESSAGE   | CD-LINKS 400 | World      | Primary Energy | EJ/y     |    462.5 |    500.7 |      ... |
| ...       | ...          | ...        | ...            | ...      |      ... |      ... |      ... |

*An illustration of the IAMC template using a scenario
from the [CD-LINKS](https://www.cd-links.org) project*  
*via the The [IAMC 1.5°C Scenario Explorer](https://data.ece.iiasa.ac.at/iamc-1.5c-explorer)*

#### Subannual time resolution

The package also supports timeseries data with a *sub-annual time resolution*:
 - Continuous-time data using the Python [datetime format](https://docs.python.org/3/library/datetime.html)
 - "Representative timeslices" (e.g., "winter-night", "summer-day")
   using the pyam *extra-columns* feature 
   
[Read the docs](https://pyam-iamc.readthedocs.io/en/stable/data.html)
for more information about the pyam data model
or look at the [data-table tutorial](pyam-iamc.readthedocs.io/en/stable/tutorials/data_table_formats.ipynb)
to see how to cast from a variety of timeseries formats to a **pyam.IamDataFrame**.

Tutorials
---------

An introduction to the basic functions is shown
in [the "first-steps" notebook](doc/source/tutorials/pyam_first_steps.ipynb).

All tutorials are available in rendered format (i.e., with output) as part of
the [online documentation](https://pyam-iamc.readthedocs.io/en/stable/tutorials.html).
The source code of the tutorials notebooks is available
in the folder [doc/source/tutorials](doc/source/tutorials) of this repository.

Documentation
-------------

The comprehensive documentation is hosted on [Read the Docs](https://pyam-iamc.readthedocs.io).

The documentation pages can be built locally,
refer to the instruction in [doc/README](doc/README.md).

Authors & Contributors
----------------------

This package was initiated and is currently maintained
by Matthew Gidden ([@gidden](https://github.com/gidden))
and Daniel Huppmann ([@danielhuppmann](https://github.com/danielhuppmann/)).

See the complete [list of contributors](AUTHORS.rst).

License
-------

Copyright 2017-2021 IIASA and the pyam developer team

The **pyam** package is licensed
under the Apache License, Version 2.0 (the "License");  
see [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for details.

Install
-------

For basic instructions,
please [read the docs](https://pyam-iamc.readthedocs.io/en/stable/install.html)!

To install from source (including all dependencies)
after cloning this repository, simply run

```
pip install --editable .[tests,optional-io-formats,tutorials]
```

To check that the package was installed correctly, run

```
pytest tests
```
