# Release v2.1.0

## Highlights

- More flexible and intuitive signature for data validation
- Improved support for region-aggregation with inconsistent data/weight index
- Fix wrong color codes for AR6 Illustrative Pathways

## Individual updates

- [#804](https://github.com/IAMconsortium/pyam/pull/804) Support filters as direct keyword arguments for `validate()` method
- [#801](https://github.com/IAMconsortium/pyam/pull/801) Support initializing with `meta` dataframe in long format
- [#796](https://github.com/IAMconsortium/pyam/pull/796) Raise explicit error message if no connection to IIASA manager service
- [#794](https://github.com/IAMconsortium/pyam/pull/794) Fix wrong color codes for AR6 Illustrative Pathways
- [#792](https://github.com/IAMconsortium/pyam/pull/792) Support region-aggregation with weights-index >> data-index

# Release v2.0.0

## Highlights

- Use **ixmp4** as dependency for better integration with the IIASA Scenario Explorer database infrastructure 
- Refactor the "exclude"-column for validation as own attribute (instead of a column in `meta`)
- Implement a cleaner package structure and reduce exposure of internal methods/modules

## Dependency changes

Support for Python 3.7-3.9 was removed due to an incompatible dependency.  

PR [#766](https://github.com/IAMconsortium/pyam/pull/766) added the **ixmp4** package
for better integration with the IIASA scenario database infrastructure.

## API changes

Credentials to access the IIASA scenario database infrastructure should now be managed
using the **ixmp4** package
(see [here](https://pyam-iamc.readthedocs.io/en/stable/api/iiasa.html)).

The column *exclude* of the `meta` indicators was moved to a new attribute `exclude`.
All validation methods are refactored such that the argument `exclude_on_fail` changes
this new attribute (see PR [#759](https://github.com/IAMconsortium/pyam/pull/759)).

The term "exclude" is now an illegal column for (timeseries) data and meta indicators.
When importing an xlsx file created with pyam < 2.0, which has an "exclude" column in
the "meta" sheet, the column is moved to the new `exclude` attribute with a log message.

PR [#764](https://github.com/IAMconsortium/pyam/pull/764) implemented a more restrictive
approach to exposing pyam-internals at the package level, requiring a more explicit
import of these methods. For example, use `pyam.utils.to_list()`
instead of `pyam.to_list()`.

PR [#773](https://github.com/IAMconsortium/pyam/pull/773) removed the `map_regions()`
method. This feature is now implemented via the **nomenclature.RegionProcessor** class.

## Individual updates

- [#773](https://github.com/IAMconsortium/pyam/pull/773) Remove `map_regions()` and default mappings csv
- [#772](https://github.com/IAMconsortium/pyam/pull/772) Show all missing rows for `require_data()`
- [#771](https://github.com/IAMconsortium/pyam/pull/771) Refactor to start a separate validation module
- [#766](https://github.com/IAMconsortium/pyam/pull/766) Use **ixmp4** for credentials to access a Scenario Explorer database
- [#764](https://github.com/IAMconsortium/pyam/pull/764) Clean-up exposing internal methods and attributes
- [#763](https://github.com/IAMconsortium/pyam/pull/763) Implement a fix against carrying over unused levels when initializing from an indexed pandas object
- [#759](https://github.com/IAMconsortium/pyam/pull/759) Excise "exclude" column from meta and add a own attribute
- [#747](https://github.com/IAMconsortium/pyam/pull/747) Drop support for Python 3.7

# Release v1.9.0

## Highlights

- Ensure compatibility with **pandas v2.0**
- Add **Python 3.11** to supported versions
- Fix an issue when concatenating a list of **pd.DataFrame** objects 

## Individual updates

- [#745](https://github.com/IAMconsortium/pyam/pull/745) Add Python 3.11 to supported versions
- [#742](https://github.com/IAMconsortium/pyam/pull/742) Fix an issue when concatenating only pd.DataFrame objects
- [#739](https://github.com/IAMconsortium/pyam/pull/739) Ensure compatibility with **pandas v2.0**

# Release v1.8.0

## Highlights

- **Performance improvement** at initialization and in internal operations
- Update the **IIASA-Connection API** in preparation of a new database backend
- Improve the **tutorials**

## API changes

In anticipation of a change in the **ixmp** API, the argument `default` in the
IIASA-Connection methods was renamed to `default_only`.

## Individual updates

- [#734](https://github.com/IAMconsortium/pyam/pull/734) Validation for illegal column names in `data`
- [#733](https://github.com/IAMconsortium/pyam/pull/733) Change IIASA-Connection argument to `default_only`
- [#732](https://github.com/IAMconsortium/pyam/pull/732) Fix a few typos in tutorials
- [#731](https://github.com/IAMconsortium/pyam/pull/731) Add fast-path to initialization for sufficient multi-index
- [#730](https://github.com/IAMconsortium/pyam/pull/730) Refactor initialization code
- [#729](https://github.com/IAMconsortium/pyam/pull/729) Improve performance at initialization
- [#723](https://github.com/IAMconsortium/pyam/pull/723) Ensure correct order of `time` attribute

# Release v1.7.0

## Highlights

- Add a feature to compute (weighted) quantiles for scenario data
- Implement a `require_data()` method for scenario validation
- Remove 'xls' as by-default-supported file format to harmonize behavior with **pandas** 

## API changes

The method `compute_bias()` was removed; please use `compute.bias()` instead.

## Dependency changes

Remove **xlrd** as a dependency; please install it explicitly for reading `.xls` files.
Bump minimum version of **pandas** to v1.2.0 to support automatic engine selection.

## Individual updates

- [#715](https://github.com/IAMconsortium/pyam/pull/715) Add a `require_data()` method
- [#713](https://github.com/IAMconsortium/pyam/pull/713) Informative error when using lists for filter by level, `level` now a forbidden column.
- [#709](https://github.com/IAMconsortium/pyam/pull/709) Hotfix ops to support `fillna=0`
- [#708](https://github.com/IAMconsortium/pyam/pull/708) Remove 'xls' as by-default-supported file format
- [#686](https://github.com/IAMconsortium/pyam/pull/686) Add support for (weighted) quantile timeseries as `df.compute.quantiles()` with a [tutorial](https://pyam-iamc.readthedocs.io/en/stable/tutorials/quantiles.html)

# Release v1.6.0

## Highlights

- Improved performance for writing data to *xlsx*
- Support for filtering by model-scenario pairs with an `index` argument
- Better integration with the IIASA Scenario Explorer database API 

## Dependency changes

PR [#701](https://github.com/IAMconsortium/pyam/pull/701) added `xlsxwriter` as a
dependency for better performance.

## Individual updates

- [#702](https://github.com/IAMconsortium/pyam/pull/702) Migrate `compute_bias()` to `compute` module
- [#701](https://github.com/IAMconsortium/pyam/pull/701) Add **xlsxwriter** as dependency to improve `to_excel()` performance
- [#699](https://github.com/IAMconsortium/pyam/pull/699) Add filter options to IIASA API `index()`, `meta()` and `properties()` methods
- [#697](https://github.com/IAMconsortium/pyam/pull/697) Add warning if IIASA API returns empty result
- [#696](https://github.com/IAMconsortium/pyam/pull/696) Added ability to load preferentially from a local cache
- [#695](https://github.com/IAMconsortium/pyam/pull/695) Remove unused meta levels during initialization
- [#688](https://github.com/IAMconsortium/pyam/pull/688) Remove ixmp as optional dependency
- [#684](https://github.com/IAMconsortium/pyam/pull/684) Use new IIASA-manager API with token refresh 
- [#679](https://github.com/IAMconsortium/pyam/pull/679) `set_meta()` now supports pandas.DataFrame as an argument
- [#674](https://github.com/IAMconsortium/pyam/pull/674) Support filtering data by model-scenario pairs with the `index` argument to `filter()` and `slice()`

# Release v1.5.0

## Highlights

This release introduces an [IamSlice](https://pyam-iamc.readthedocs.io/en/latest/api/slice.html)
class that allows faster filtering and inspection of an **IamDataFrame**.

## Individual updates

- [#668](https://github.com/IAMconsortium/pyam/pull/668) Allow renaming of empty IamDataFrame objects
- [#665](https://github.com/IAMconsortium/pyam/pull/665) Provide better support for IamDataFrame objects with non-standard index dimensions
- [#659](https://github.com/IAMconsortium/pyam/pull/659) Add an `offset` method
- [#657](https://github.com/IAMconsortium/pyam/pull/657) Add an `IamSlice` class

# Release v1.4.0

## Highlights

- Add colors used for IPCC AR6 WGIII scenario analysis
- Support scenario data with mixed 'year' and 'datetime' domain (beta) 
- Add explicit support for Python 3.10

## Dependency changes

Following a change of the UNFCCC data inventory API (see
[pik-primap/unfccc_di_api#39](https://github.com/pik-primap/unfccc_di_api/issues/39)),
PR [#647](https://github.com/IAMconsortium/pyam/pull/647) updated the dependencies
to require `unfccc-di-api>=3.0.1`.

## API changes

PR [#598](https://github.com/IAMconsortium/pyam/pull/598) added support for
mixed time-domains, i.e., where the time column has both `integer` and `datetime` items.
As part of the changes, filtering an **IamDataFrame** with yearly data using arguments
that are only relevant for the datetime-domain (e.g., month, hour, time) returns
an empty **IamDataFrame**. Previously, this raised an error.

## Individual updates

- [#651](https://github.com/IAMconsortium/pyam/pull/651) Pin `pint<=0.18` as a quickfix for a regression in the latest release
- [#650](https://github.com/IAMconsortium/pyam/pull/650) Add IPCC AR6 WGIII colors to PYAM_COLORS
- [#647](https://github.com/IAMconsortium/pyam/pull/647) Pin `unfccc-di-api` to latest release
- [#634](https://github.com/IAMconsortium/pyam/pull/634) Better error message when initializing with invisible columns 
- [#598](https://github.com/IAMconsortium/pyam/pull/598) Support mixed 'year' and 'datetime' domain

# Release v1.3.1

This is a patch release to ensure compatibility with
pandas [v1.4.0](https://pandas.pydata.org/docs/whatsnew/v1.4.0.html).

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
