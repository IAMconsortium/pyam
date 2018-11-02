import numpy as np
import pandas as pd


from pyam.core import _PyamDataFrame, _aggregate_by_variables, _aggregate_by_regions
from pyam.utils import cast_years_to_int, pattern_match, REGION_IDX
from pyam.logger import logger


class IamDataFrame(_PyamDataFrame):
    """This class is a wrapper for dataframes following the IAMC format.
    It provides a number of diagnostic features (including validation of data,
    completeness of variables provided) as well as a number of visualization
    and plotting tools.
    """

    def __init__(self, data, **kwargs):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        data: ixmp.TimeSeries, ixmp.Scenario, pd.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pd.DataFrame or data file with IAMC-format data columns.
            A pd.DataFrame can have the required data as columns or index.

            Special support is provided for data files downloaded directly from
            IIASA SSP and RCP databases. If you run into any problems loading
            data, please make an issue at:
            https://github.com/IAMconsortium/pyam/issues
        """
        super(IamDataFrame, self).__init__(data, **kwargs)
        # cast year column to `int` if necessary
        if not self.data.year.dtype == 'int64':
            self.data.year = cast_years_to_int(self.data.year)

    def check_aggregate(self, variable, components=None, units=None,
                        exclude_on_fail=False, multiplier=1, **kwargs):
        """Check whether the timeseries data match the aggregation
        of components or sub-categories

        Parameters
        ----------
        variable: str
            variable to be checked for matching aggregation of sub-categories
        components: list of str, default None
            list of variables, defaults to all sub-categories of `variable`
        units: str or list of str, default None
            filter variable and components for given unit(s)
        exclude_on_fail: boolean, default False
            flag scenarios failing validation as `exclude: True`
        multiplier: number, default 1
            factor when comparing variable and sum of components
        kwargs: passed to `np.isclose()`
        """
        # default components to all variables one level below `variable`
        if components is None:
            var_list = pd.Series(self.data.variable.unique())
            components = var_list[pattern_match(var_list,
                                                '{}|*'.format(variable), 0)]

        if not len(components):
            msg = 'cannot check aggregate for {} because it has no components'
            logger().info(msg.format(variable))

            return

        # filter and groupby data, use `pd.Series.align` for matching index
        df_variable, df_components = (
            _aggregate_by_variables(self.data, variable, units)
            .align(_aggregate_by_variables(self.data, components, units))
        )

        # use `np.isclose` for checking match
        diff = df_variable[~np.isclose(df_variable, multiplier * df_components,
                                       **kwargs)]

        if len(diff):
            msg = '{} - {} of {} data points are not aggregates of components'
            logger().info(msg.format(variable, len(diff), len(df_variable)))

            if exclude_on_fail:
                self._exclude_on_fail(diff.index.droplevel([2, 3]))

            diff = pd.concat([diff], keys=[variable], names=['variable'])

            return diff.unstack().rename_axis(None, axis=1)

    def check_aggregate_regions(self, variable, region='World',
                                components=None, units=None,
                                exclude_on_fail=False, **kwargs):
        """Check whether the region timeseries data match the aggregation
        of components

        Parameters
        ----------
        variable: str
            variable to be checked for matching aggregation of components data
        region: str
            region to be checked for matching aggregation of components data
        components: list of str, default None
            list of regions, defaults to all regions except region
        units: str or list of str, default None
            filter variable and components for given unit(s)
        exclude_on_fail: boolean, default False
            flag scenarios failing validation as `exclude: True`
        kwargs: passed to `np.isclose()`
        """
        var_df = self.filter(variable=variable, level=0)

        if components is None:
            components = list(set(var_df.data.region) - set([region]))

        if not len(components):
            msg = (
                'cannot check regional aggregate for `{}` because it has no '
                'regional components'
            )
            logger().info(msg.format(variable))

            return None

        # filter and groupby data, use `pd.Series.align` for matching index
        df_region, df_components = (
            _aggregate_by_regions(var_df.data, region, units)
            .align(_aggregate_by_regions(var_df.data, components, units))
        )

        df_components.index = df_components.index.droplevel(
            "variable"
        )

        # Add in variables that are included in region totals but which
        # aren't included in the regional components.
        # For example, if we are looking at World and Emissions|BC, we need
        # to add aviation and shipping to the sum of Emissions|BC for each
        # of World's regional components to do a valid check.
        different_region = components[0]
        var_list = pd.Series(self.data.variable.unique())
        var_components = var_list[pattern_match(var_list,
                                                '{}|*'.format(variable), 0)]
        for var_to_add in var_components:
            var_rows = self.data.variable == var_to_add
            region_rows = self.data.region == different_region
            var_has_regional_info = (var_rows & region_rows).any()
            if not var_has_regional_info:
                df_var_to_add = self.filter(
                    region=region, variable=var_to_add
                ).data.groupby(REGION_IDX).sum()['value']
                df_var_to_add.index = df_var_to_add.index.droplevel("variable")

                if len(df_var_to_add):
                    df_components = df_components.add(df_var_to_add,
                                                      fill_value=0)

        df_components = pd.concat([df_components], keys=[variable],
                                  names=['variable'])

        # use `np.isclose` for checking match
        diff = df_region[~np.isclose(df_region, df_components, **kwargs)]

        if len(diff):
            msg = (
                '{} - {} of {} data points are not aggregates of regional '
                'components'
            )
            logger().info(msg.format(variable, len(diff), len(df_region)))

            if exclude_on_fail:
                self._exclude_on_fail(diff.index.droplevel([2, 3]))

            diff = pd.concat([diff], keys=[region], names=['region'])

            return diff.unstack().rename_axis(None, axis=1)

    def check_internal_consistency(self, **kwargs):
        """Check whether the database is internally consistent

        We check that all variables are equal to the sum of their sectoral
        components and that all the regions add up to the World total. If
        the check is passed, None is returned, otherwise a dictionary of
        inconsistent variables is returned.

        Note: at the moment, this method's regional checking is limited to
        checking that all the regions sum to the World region. We cannot
        make this more automatic unless we start to store how the regions
        relate, see
        [this issue](https://github.com/IAMconsortium/pyam/issues/106).

        Parameters
        ----------
        kwargs: passed to `np.isclose()`
        """
        inconsistent_vars = {}
        for variable in self.variables():
            diff_agg = self.check_aggregate(variable, **kwargs)
            if diff_agg is not None:
                inconsistent_vars[variable + "-aggregate"] = diff_agg

            diff_regional = self.check_aggregate_regions(variable, **kwargs)
            if diff_regional is not None:
                inconsistent_vars[variable + "-regional"] = diff_regional

        return inconsistent_vars if inconsistent_vars else None
