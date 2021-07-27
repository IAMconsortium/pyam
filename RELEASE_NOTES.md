# Next Release

## API changes

PR [#559](https://github.com/IAMconsortium/pyam/pull/559) marked
the attribute `_LONG_IDX` as deprecated. Please use `dimensions` instead. 

## Individual updates

- [#566](https://github.com/IAMconsortium/pyam/pull/566) Updated AR6 default color pallet to final version
used by WG1 
- [#564](https://github.com/IAMconsortium/pyam/pull/564) Add an example with a secondary axis to the plotting gallery 
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
