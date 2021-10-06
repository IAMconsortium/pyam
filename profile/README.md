# Profiling

This module provides utility code to run benchmarking on features of pyam.

### *Work in progress!*

This module is in an experimental stage.
We are currently investigating additional/alternative packages for benchmarking
and profiling, and the preferred solution may change at any time.

**To-do** 

Required steps to make this module a full part of the pyam package:

- Include module dependencies as extra-requirements of the installation
- Migrate gitignore from this folder to the main gitignore file

## Dependencies and usage

This module uses the [pytest-monitor](https://pytest-monitor.readthedocs.io) package.

### Installation

In addition to an installation of **pyam** with all optional dependencies,
install the pytest-extension using

```
pip install pytest-monitor
```

### Usage

The **pytest-monitor** package is executed automatically (if installed) when
running **pytest**, writing metrics for each test to a SQLite database (``.pymon``).
To use the profiling module, navigate to the `profile` folder and run pytest.
Then, ``profile_report.py`` prints metrics of the latest session to the command line.

```
pytest .
python profile_report.py
```

## Adding benchmarks and profile tests

To add profiling/benchmarking tests, add any standard pytest-functions to this folder.
