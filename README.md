# Simple diagnostics and visualization tools <br> for Integrated Assessment timeseries data

## Overview of the ``pyam_analysis`` package

This repository contains a Python package with some tools for diagnostics and visualization
of timeseries data from Integrated Assessment models (IAM).

Features:
- Summary of models, scenarios, variables as dataframe or Pivot table
- Simple visualization and plotting of data with flexible filtering
- Diagnostic checks for non-reported variables or timeseries data outside of specified range
- Assignment of scenarios to categories for meta-analysis

The ``pyam_analysis`` package assumes that any data follows the data template convention 
of the [Integrated Assessment Modeling Consortium](http://www.globalchange.umd.edu/iamc/):

| **model**           | **scenario**  | **region** | **variable**   | **unit** | **2005** | **2010** | **2015** |
|---------------------|---------------|------------|----------------|----------|----------|----------|----------|
| MESSAGE-GLOBIOM 1.0 | SSP2-Baseline | World      | Primary Energy | EJ/y     | 463.8    | 500.6    | ...      |
| ...                 | ...           | ...        | ...            | ...      | ...      | ...      | ...      |

## Documentation

Coming soon...

## Python dependencies

0. `Sphinx <http://sphinx-doc.org/>`_ v1.1.2 or higher
0. `sphinxcontrib.bibtex`
0. `sphinxcontrib-fulltoc`
0. `matplotlib`
0. `seaborn`

## Installation instructions

0. Fork this repository and clone the forked repository (`<user>/pyam-analysis`) to
   your machine.  To fork the repository, look for the fork icon in the top right at
   [iiasa/pyam-analysis](https://github.com/iiasa/pyam-analysis).
   Add `iiasa/pyam-analysis` as `upstream` to your clone.

   *We recommend* [GitKraken](https://www.gitkraken.com/) *for users who prefer a graphical user interface application*<br>
   *to work with Github (as opposed to the command line).*
   
### Windows Users

0. Double click on `install.bat` in the local folder in which you saved your fork.

### *nix Users

0. In a command prompt, execute the following command

    ```
    python setup.py install
    ```

