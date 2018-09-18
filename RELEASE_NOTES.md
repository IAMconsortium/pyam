
# Next Release

- (#92)[https://github.com/IAMconsortium/pyam/pull/92] Adding `$` to the pseudo-regexp syntax in `pattern_match()`, adds override option
- (#90)[https://github.com/IAMconsortium/pyam/pull/90] Adding a function to `set_panel_label()` as part of the plotting library
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
