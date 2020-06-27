pyam: analysis & visualization of integrated-assessment scenarios
===================================================================

[![license](https://anaconda.org/conda-forge/pyam/badges/license.svg)](https://anaconda.org/conda-forge/pyam)
[![pypi](https://img.shields.io/pypi/v/pyam-iamc.svg)](https://pypi.python.org/pypi/pyam-iamc/)
[![conda](https://anaconda.org/conda-forge/pyam/badges/version.svg)](https://anaconda.org/conda-forge/pyam)
[![latest](https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/pyam)

[![pytest-38](https://github.com/IAMconsortium/pyam/workflows/pytest%20(3.8)/badge.svg?branch=master)](https://github.com/IAMconsortium/pyam/actions?query=workflow%3A%22pytest+%283.8%29%22+branch%3Amaster)
[![pytest-37](https://github.com/IAMconsortium/pyam/workflows/pytest%20(3.7)/badge.svg?branch=master)](https://github.com/IAMconsortium/pyam/actions?query=workflow%3A%22pytest+%283.7%29%22+branch%3Amaster)
[![codecov](https://codecov.io/gh/IAMconsortium/pyam/branch/master/graph/badge.svg)](https://codecov.io/gh/IAMconsortium/pyam)
[![ReadTheDocs](https://readthedocs.org/projects/pyam-iamc/badge/?version=latest)](https://pyam-iamc.readthedocs.io/en/latest/?badge=latest)

[![doi](https://zenodo.org/badge/113359260.svg)](https://zenodo.org/badge/latestdoi/113359260)
[![joss](https://joss.theoj.org/papers/10.21105/joss.01095/status.svg)](https://joss.theoj.org/papers/10.21105/joss.01095)

****

**Documentation on [Read the Docs](https://pyam-iamc.readthedocs.io)**

**Questions? Start a discussion on our [mailing list](https://groups.io/g/pyam)**

Overview and scope
------------------

The open-source Python package **pyam** provides a suite of tools and functions
for analyzing and visualizing input data (i.e., assumptions/parametrization) 
and results (model output) of integrated-assessment scenarios.

Key features:

 - Simple analysis of timeseries data in the IAMC format
   (more about it [here](https://pyam-iamc.readthedocs.io/en/stable/data.html))
   with an interface similar in feel and style to the widely
   used [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
 - Advanced visualization and plotting functions
   (see the [gallery](https://pyam-iamc.readthedocs.io/en/stable/examples/index.html))
 - Diagnostic checks for scripted validation of scenario data and results

Data model
----------

An illustrative example of the timeseries format developed by the
[Integrated Assessment Modeling Consortium](https://www.iamconsortium.org) (IAMC)
is shown below.
The row is taken from the [IAMC 1.5°C scenario explorer](https://data.ene.iiasa.ac.at/iamc-1.5c-explorer),
showing a scenario from the [CD-LINKS](https://www.cd-links.org) project.
[Read the docs](https://pyam-iamc.readthedocs.io/en/stable/data.html)
for more information on the IAMC format and the ``pyam`` data model.

| **model** | **scenario** | **region** | **variable**   | **unit** | **2005** | **2010** | **2015** |
|-----------|--------------|------------|----------------|----------|----------|----------|----------|
| MESSAGE   | CD-LINKS 400 | World      | Primary Energy | EJ/y     |    462.5 |    500.7 |      ... |
| ...       | ...          | ...        | ...            | ...      |      ... |      ... |      ... |


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

The complete documentation is hosted on [Read the Docs](https://pyam-iamc.readthedocs.io).

The documentation pages can be built locally,
refer to the instruction in [doc/README](doc/README.md).

Authors
-------

This package was developed and is currently maintained
by Matthew Gidden ([@gidden](https://github.com/gidden))
and Daniel Huppmann ([@danielhuppmann](https://github.com/danielhuppmann/)).

License
-------

Copyright 2017-2020 IIASA Energy Program

The **pyam** package is licensed
under the Apache License, Version 2.0 (the "License");
see [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for details.

Install
-------

For basic instructions,
[read the docs](https://pyam-iamc.readthedocs.io/en/stable/install.html).

To install from source after cloning this repository, simply run

```
pip install -e .
```

### Development

To setup a development environment, the simplest route is to make yourself 
a conda environment and then follow the `Makefile`. 

```sh
# pyam can be replaced with any other name
# you don't have to specify your python version if you don't want
conda create --name pyam pip python=X.Y.Z
conda activate pyam  # may be  simply `source activate pyam` or just `activate pyam`
# use the make file to create your development environment
# (you only require the -B flag the first time, thereafter you can
# just run `make virtual-environment` and it will only update if
# environment definition files have been updaed)
make -B virtual-environment
```

Instead of conda you could also use a pip virtualenv:
```sh
mkdir venv
virtualenv venv -p python3
. venv/bin/activate
pip install -e .[test,optional-io-formats]
```

To check everything has installed correctly, run

```
pytest tests
```

All the tests should pass.
