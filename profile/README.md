# Profiling

This module provides utility code to run benchmarking on features of pyam.

## Dependencies and usage

The profiling module uses [pytest-monitor](https://pytest-monitor.readthedocs.io)
package.

### Installation

In addition to an installation of **pyam** with all optional dependencies,
install the pytest-extension using

```
pip install pytest-monitor
```

### Usage

The **pytest-monitor** package is executed automatically (if installed) when
running **pytest**, writing metrics of each test to an SQLite database (``.pymon``).
To use the profiling module, navigate to the `profile` folder and run pytest.
Then, ``profile_report.py`` prints the metrics to the command line.

```
pytest .
python profile_report.py
```

## Adding benchmarks and profile tests


