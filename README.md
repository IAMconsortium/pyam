pyam: analysis and visualization of integrated-assessment scenarios
===================================================================

**Documentation on [Read the Docs](https://pyam-iamc.readthedocs.io)**

**Questions? Start a discussion on our [mailing list](https://groups.io/g/pyam)**

Overview and scope
------------------

The open-source Python package ``pyam`` provides a suite of tools and functions
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

Features:
- Summary of models, scenarios, variables, and regions included in a snapshot.

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

Copyright 2017-2019 IIASA Energy Program

The ``pyam`` package is licensed
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

To check everything has installed correctly,

```
pytest tests
```

All the tests should pass.
