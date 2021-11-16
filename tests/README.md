# Plotting Tests

Plotting tests are used as regression tests for plotting features. They can be
run locally (see below) and are also run on CI.

## Install Deps

You have to install `pytest-mpl` to run the plotting tests.

## Tests Failing on CI?

Make sure your local version of `matplotlib` and `seaborn` are the same as on
CI. `seaborn` can override default `matplotlib` style sheets, and thus both need
to be the same version.

## Creating Baseline Images

```
pytest --mpl-generate-path=expected_figs test_plotting.py
```

## Running tests

```
pytest --mpl
```
