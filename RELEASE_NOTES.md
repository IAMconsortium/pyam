# Release v1.3.1

This is a patch release to ensure compatibility with
pandas [v1.4.0](https://pandas.pydata.org/docs/whatsnew/v1.4.0.html),

# Release v1.3.0

## Highlights

- Implement a `compute` module for derived timeseries indicators.
- Add a `diff()` method similar to the corresponding `pandas.DataFrame.diff()`
- Improve error reporting on IamDataFrame initialization

## Individual updates

- [#608](https://github.com/IAMconsortium/pyam/pull/608) The method `assert_iamframe_equals()` passes if an all-nan-col is present
- [#604](https://github.com/IAMconsortium/pyam/pull/604) Add an annualized-growth-rate method
- [#602](https://github.com/IAMconsortium/pyam/pull/602) Add a `compute` module/accessor and a learning-rate method 
- [#600](https://github.com/IAMconsortium/pyam/pull/600) Add a `diff()` method
- [#592](https://github.com/IAMconsortium/pyam/pull/592) Fix for running in jupyter-lab notebooks
- [#590](https://github.com/IAMconsortium/pyam/pull/590) Update expected figures of plotting tests to use matplotlib 3.5
- [#586](https://github.com/IAMconsortium/pyam/pull/586) Improve error reporting for non-numeric data in any value column

# Release v1.2.0

## Highlights

- Update the source code of the manuscript in *Open Research Europe* to reflect changes
  based on reviewer comments
- Increase the performance of the IamDataFrame initialization
- Add an experimental "profiler" module for performance benchmarking

## Dependency changes

The dependencies were updated to require `xlrd>=2.0` (previously `<2.0`) and `openpyxl`
was added as a dependency.

## Individual updates

- [#585](https://github.com/IAMconsortium/pyam/pull/585) Include revisions to the ORE manuscript source code following acceptance/publication 
- [#583](https://github.com/IAMconsortium/pyam/pull/583) Add profiler module for performance benchmarking
- [#579](https://github.com/IAMconsortium/pyam/pull/579) Increase performance of IamDataFrame initialization
- [#572](https://github.com/IAMconsortium/pyam/pull/572) Unpinned the requirements for xlrd and added openpyxl as a requirement to ensure ongoing support of both `.xlsx` and `.xls` files out of the box

# Release v1.1.0

## Highlights

- Update pyam-colors to be consistent with IPCC AR6 palette
- Enable `colors` keyword argument as list in `plot.pie()`
- Fix compatibility with pandas v1.3

## API changes

PR [#559](https://github.com/IAMconsortium/pyam/pull/559) marked
the attribute `_LONG_IDX` as deprecated. Please use `dimensions` instead. 

## Individual updates

- [#566](https://github.com/IAMconsortium/pyam/pull/566) Updated AR6 default color pallet to final version used by WG1
- [#564](https://github.com/IAMconsortium/pyam/pull/564) Add an example with a secondary axis to the plotting gallery 
- [#563](https://github.com/IAMconsortium/pyam/pull/563) Enable `colors` keyword argument as list in `plot.pie()` 
- [#562](https://github.com/IAMconsortium/pyam/pull/562) Add `get_data_column()`, refactor filtering by the time domain
- [#560](https://github.com/IAMconsortium/pyam/pull/560) Add a feature to `swap_year_for_time()`
- [#559](https://github.com/IAMconsortium/pyam/pull/559) Add attribute `dimensions`, fix compatibility with pandas v1.3
- [#557](https://github.com/IAMconsortium/pyam/pull/557) Swap time for year keeping subannual resolution
- [#556](https://github.com/IAMconsortium/pyam/pull/556) Set explicit minimum numpy version (1.19.0)

# Release v1.0.0

This is the first major release of the pyam package.
It coincides with the publication of a manuscript in **Open Research Europe**
(doi: [10.12688/openreseurope.13633.1](http://doi.org/10.12688/openreseurope.13633.1)).

## Notes on prior releases

As part of the release v1.0, several functions and methods were removed
that had been marked as deprecated over several release cycles.
Please refer to [Release v1.0](https://github.com/IAMconsortium/pyam/releases/tag/v1.0.0)
on GitHub for the detailed changes and
the [v0.13 release notes](https://github.com/IAMconsortium/pyam/blob/v0.13.0/RELEASE_NOTES.md)
for information on the release history prior to v1.0.
