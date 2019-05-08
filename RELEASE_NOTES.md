
# Next Release

- [#228](https://github.com/IAMconsortium/pyam/pull/228) Update development environment creation instructions and make pandas requirement more specific
- [#219](https://github.com/IAMconsortium/pyam/pull/219) Add ability to query metadata from iiasa data sources
- [#214](https://github.com/IAMconsortium/pyam/pull/214) Tidy up requirements specifications a little
- [#213](https://github.com/IAMconsortium/pyam/pull/213) Add support for IPCC colors, see the new tutorial "Using IPCC Color Palattes"
- [#212](https://github.com/IAMconsortium/pyam/pull/212) Now natively support reading R-style data frames with year columns like "X2015"
- [#207](https://github.com/IAMconsortium/pyam/pull/207) Add a `aggregate_region()` function to sum a variable from subregions and add components that are only defined at the region level
- [#202](https://github.com/IAMconsortium/pyam/pull/202) Extend the `df.rename()` function with a `check_duplicates (default True)` validation option
- [#201](https://github.com/IAMconsortium/pyam/pull/201) Added native support for legends outside of plots with `pyam.plotting.OUTSIDE_LEGEND` with a tutorial
- [#200](https://github.com/IAMconsortium/pyam/pull/200) Bugfix when providing `cmap`  and `color`  arguments to plotting functions
- [#199](https://github.com/IAMconsortium/pyam/pull/199) Initializing an `IamDataFrame` accepts kwargs to fill or create from the data any missing required columns
- [#198](https://github.com/IAMconsortium/pyam/pull/198) Update stack plot functionality and add aggregation tutorial. Also adds a `copy` method to `IamDataFrame`.
- [#197](https://github.com/IAMconsortium/pyam/pull/197) Added a `normalize` function that normalizes all data in a data frame to a specific time period.
- [#195](https://github.com/IAMconsortium/pyam/pull/195) Fix filtering for `time`, `day` and `hour` to use generic `pattern_match()` (if such a column exists) in 'year'-formmatted IamDataFrames
- [#192](https://github.com/IAMconsortium/pyam/pull/192) Extend `utils.find_depth()` to optionally return depth (as list of ints) rather than assert level tests
- [#190](https://github.com/IAMconsortium/pyam/pull/190) Add `concat()` function
- [#189](https://github.com/IAMconsortium/pyam/pull/189) Fix over-zealous `dropna()` in `scatter()`
- [#186](https://github.com/IAMconsortium/pyam/pull/186) Fix over-zealous `dropna()` in `line_plot()`, rework `as_pandas()` to (optionally) discover meta columns to be joined
- [#178](https://github.com/IAMconsortium/pyam/pull/178) Add a kwarg `append` to the function `rename()`, change behaviour of mapping to only apply to data where all given columns are matched
- [#177](https://github.com/IAMconsortium/pyam/pull/177) Modified formatting of time column on init to allow subclasses to avoid pandas limitation (https://stackoverflow.com/a/37226672)
- [#176](https://github.com/IAMconsortium/pyam/pull/176) Corrected title setting operation in line_plot function
- [#175](https://github.com/IAMconsortium/pyam/pull/175) Update link to tutorial in readme.md
- [#174](https://github.com/IAMconsortium/pyam/pull/174) Add a function `difference()` to compare two IamDataFrames
- [#171](https://github.com/IAMconsortium/pyam/pull/171) Fix a bug when reading from an `ixmp.TimeSeries` object, refactor to mitigate circular dependency
- [#162](https://github.com/IAMconsortium/pyam/pull/162) Add a function to sum and append timeseries components to an aggregate variable
- [#152](https://github.com/IAMconsortium/pyam/pull/152) Fix bug where scatter plots did not work with property metadata when using two variables (#136, #152)
- [#151](https://github.com/IAMconsortium/pyam/pull/151) Fix bug where excel files were not being written on Windows and MacOSX (#149)
- [#145](https://github.com/IAMconsortium/pyam/pull/145) Support full semantic and VCS-style versioning with `versioneer`
- [#132](https://github.com/IAMconsortium/pyam/pull/132) support time columns using the `datetime` format and additional `str` columns in `data`

# Release v0.1.2

- [#128](https://github.com/IAMconsortium/pyam/pull/128) add ability to directly read data from iiasa data sources
- [#120](https://github.com/IAMconsortium/pyam/pull/120) update install setup

# Release v0.1.0

- [#114](https://github.com/IAMconsortium/pyam/pull/114) extends `append()` such that data can be added to existing scenarios
- [#111](https://github.com/IAMconsortium/pyam/pull/111) extends `set_meta()` such that it requires a name (not None)
- [#109](https://github.com/IAMconsortium/pyam/pull/109) add ability to fill between and add data ranges in `line_plot()`
- [#104](https://github.com/IAMconsortium/pyam/pull/104) fixes a bug with timeseries utils functions, ensures that years are cast as integers
- [#101](https://github.com/IAMconsortium/pyam/pull/101) add function `cross_threshold()` to determine years where a timeseries crosses a given threshold
- [#98](https://github.com/IAMconsortium/pyam/pull/98) add a module to compute and format summary statistics for timeseries data (wrapper for `pd.describe()`
- [#95](https://github.com/IAMconsortium/pyam/pull/95) add a `scatter()` chart in the plotting library using metadata
- [#94](https://github.com/IAMconsortium/pyam/pull/94) `set_meta()` can take pd.DataFrame (with columns `['model', 'scenario']`) as `index` arg
- [#93](https://github.com/IAMconsortium/pyam/pull/93) IamDataFrame can be initilialzed from pd.DataFrame with index
- [#92](https://github.com/IAMconsortium/pyam/pull/92) Adding `$` to the pseudo-regexp syntax in `pattern_match()`, adds override option
- [#90](https://github.com/IAMconsortium/pyam/pull/90) Adding a function to `set_panel_label()` as part of the plotting library
- [#88](https://github.com/IAMconsortium/pyam/pull/88) Adding `check_aggregate_regions` and `check_internal_consistency` to help with database validation, especially for emissions users
- [#87](https://github.com/IAMconsortium/pyam/pull/87) Extending `rename()` to work with model and scenario names
- [#85](https://github.com/IAMconsortium/pyam/pull/85) Improved functionality for importing metadata and bugfix for filtering for strings if `nan` values exist in metadata
- [#83](https://github.com/IAMconsortium/pyam/pull/83) Extending `filter_by_meta()` to work with non-matching indices between `df` and `data
- [#81](https://github.com/IAMconsortium/pyam/pull/81) Bug-fix when using `set_meta()` with unnamed pd.Series and no `name` kwarg
- [#80](https://github.com/IAMconsortium/pyam/pull/80) Extend the pseudo-regexp syntax for filtering in `pattern_match()`
- [#73](https://github.com/IAMconsortium/pyam/pull/73) Adds ability to remove labels for markers, colors, or linestyles
- [#72](https://github.com/IAMconsortium/pyam/pull/72) line_plot now drops NaNs so that lines are fully plotted
- [#71](https://github.com/IAMconsortium/pyam/pull/71) Line plots `legend` keyword can now be a dictionary of legend arguments
- [#70](https://github.com/IAMconsortium/pyam/pull/70) Support reading of both SSP and RCP data files downloaded from the IIASA database.
- [#66](https://github.com/IAMconsortium/pyam/pull/66) Fixes a bug in the `interpolate()` function (duplication of data points if already defined)
- [#65](https://github.com/IAMconsortium/pyam/pull/65) Add a `filter_by_meta()` function to filter/join a pd.DataFrame with an IamDataFrame.meta table
