
# Next Release

- (#128)[https://github.com/IAMconsortium/pyam/pull/128] add ability to directly read data from iiasa data sources


# Release v0.1.0

- (#114)[https://github.com/IAMconsortium/pyam/pull/114] extends `append()` such that data can be added to existing scenarios
- (#111)[https://github.com/IAMconsortium/pyam/pull/111] extends `set_meta()` such that it requires a name (not None)
- (#109)[https://github.com/IAMconsortium/pyam/pull/109] add ability to fill between and add data ranges in `line_plot()`
- (#104)[https://github.com/IAMconsortium/pyam/pull/104] fixes a bug with timeseries utils functions, ensures that years are cast as integers
- (#101)[https://github.com/IAMconsortium/pyam/pull/101] add function `cross_threshold()` to determine years where a timeseries crosses a given threshold
- (#98)[https://github.com/IAMconsortium/pyam/pull/98] add a module to compute and format summary statistics for timeseries data (wrapper for `pd.describe()`
- (#95)[https://github.com/IAMconsortium/pyam/pull/95] add a `scatter()` chart in the plotting library using metadata 
- (#94)[https://github.com/IAMconsortium/pyam/pull/94] `set_meta()` can take pd.DataFrame (with columns `['model', 'scenario']`) as `index` arg
- (#93)[https://github.com/IAMconsortium/pyam/pull/93] IamDataFrame can be initilialzed from pd.DataFrame with index
- (#92)[https://github.com/IAMconsortium/pyam/pull/92] Adding `$` to the pseudo-regexp syntax in `pattern_match()`, adds override option
- (#90)[https://github.com/IAMconsortium/pyam/pull/90] Adding a function to `set_panel_label()` as part of the plotting library
- (#88)[https://github.com/IAMconsortium/pyam/pull/88] Adding `check_aggregate_regions` and `check_internal_consistency` to help with database validation, especially for emissions users
- (#87)[https://github.com/IAMconsortium/pyam/pull/87] Extending `rename()` to work with model and scenario names
- (#85)[https://github.com/IAMconsortium/pyam/pull/85] Improved functionality for importing metadata and bugfix for filtering for strings if `nan` values exist in metadata
- (#83)[https://github.com/IAMconsortium/pyam/pull/83] Extending `filter_by_meta()` to work with non-matching indices between `df` and `data
- (#81)[https://github.com/IAMconsortium/pyam/pull/81] Bug-fix when using `set_meta()` with unnamed pd.Series and no `name` kwarg
- (#80)[https://github.com/IAMconsortium/pyam/pull/80] Extend the pseudo-regexp syntax for filtering in `pattern_match()`
- (#73)[https://github.com/IAMconsortium/pyam/pull/73] Adds ability to remove labels for markers, colors, or linestyles
- (#72)[https://github.com/IAMconsortium/pyam/pull/72] line_plot now drops NaNs so that lines are fully plotted
- (#71)[https://github.com/IAMconsortium/pyam/pull/71] Line plots `legend` keyword can now be a dictionary of legend arguments
- (#70)[https://github.com/IAMconsortium/pyam/pull/70] Support reading of both SSP and RCP data files downloaded from the IIASA database.
- (#66)[https://github.com/IAMconsortium/pyam/pull/66] Fixes a bug in the `interpolate()` function (duplication of data points if already defined)
- (#65)[https://github.com/IAMconsortium/pyam/pull/65] Add a `filter_by_meta()` function to filter/join a pd.DataFrame with an IamDataFrame.meta table
