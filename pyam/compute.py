import itertools
import math

import pandas as pd
import wquantiles

import pyam
from pyam._debiasing import _compute_bias
from pyam.index import replace_index_values
from pyam.kaya import kaya_factors, kaya_variables, lmdi
from pyam.timeseries import growth_rate
from pyam.utils import remove_from_list


class IamComputeAccessor:
    """Perform computations on the timeseries data of an IamDataFrame

    An :class:`IamDataFrame` has a module for computation of (advanced) indicators
    from the timeseries data.

    The methods in this module can be accessed via

    .. code-block:: python

        IamDataFrame.compute.<method>(*args, **kwargs)
    """

    def __init__(self, df):
        self._df = df

    def quantiles(
        self, quantiles, weights=None, level=["model", "scenario"], append=False
    ):
        """Compute the optionally weighted quantiles of data grouped by `level`.

        For example, the following will provide the interquartile range and median value
        of CO2 emissions across all models and scenarios in a given dataset:

        .. code-block:: python

            df.filter(variable="Emissions|CO2").compute.quantiles([0.25, 0.5, 0.75])

        Parameters
        ----------
        quantiles : collection
            Group of quantile values to compute
        weights : pd.Series, optional
            Series indexed by `level`
        level : collection, optional
            The index columns to compute quantiles over
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed data or None if `append=True`.

        Raises
        ------
        ValueError
            If more than one variable provided or if argument `weights` is malformed.
        """
        from pyam.core import (
            IamDataFrame,
            concat,
        )

        self_df = self._df
        if len(self_df.variable) > 1:
            raise ValueError(
                "quantiles() currently supports only 1 variable, and this"
                f"dataframe has {len(self_df.variable)}"
            )
        if weights is not None and weights.name != "weight":
            raise ValueError("weights pd.Series must have name 'weight'")

        df = self_df.timeseries()
        model = (
            "Quantiles" if weights is None else "Weighted Quantiles"
        )  # can make this a kwarg

        # get weights aligned with model/scenario in data
        if weights is None:
            df["weight"] = 1.0
        else:
            df = df.join(weights, how="inner")
        w = df["weight"]
        df.drop("weight", axis="columns", inplace=True)

        # prep data for processing
        df = df.reset_index(level=level).drop(columns=level)

        dfs = []
        # indexed over region, variable, and unit
        idxs = df.index.drop_duplicates()
        for idx, q in itertools.product(idxs, quantiles):
            data = pd.Series(
                wquantiles.quantile(df.loc[idx].values.T, w.values, q),
                index=pd.Series(df.columns, name="year"),
                name="value",
            )
            kwargs = {idxs.names[i]: idx[i] for i in range(len(idx))}
            dfs.append(
                IamDataFrame(
                    data,
                    model=model,
                    scenario=str(q),  # can make this a kwarg
                    **kwargs,
                )
            )

        # append to `self` or return as `IamDataFrame`
        return self_df._finalize(concat(dfs), append=append)

    def growth_rate(self, mapping, append=False):
        """Compute the annualized growth rate of a timeseries along the time dimension

        The growth rate parameter in period *t* is computed based on the changes
        to the subsequent period, i.e., from period *t* to period *t+1*.

        Parameters
        ----------
        mapping : dict
            Mapping of *variable* item(s) to the name(s) of the computed data,
            e.g.,

            .. code-block:: python

               {"variable": "name of growth-rate variable", ...}

        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        Raises
        ------
        ValueError
            Math domain error when timeseries crosses 0.

        See Also
        --------
        pyam.timeseries.growth_rate

        """
        value = (
            self._df._data[self._df._apply_filters(variable=mapping)]
            .groupby(remove_from_list(self._df.dimensions, ["year"]), group_keys=False)
            .apply(growth_rate)
        )
        if value.empty:
            value = empty_series(remove_from_list(self._df.dimensions, "unit"))
        else:
            # drop level "unit" and reinsert below, replace "variable"
            value.index = replace_index_values(
                value.index.droplevel("unit"), "variable", mapping
            )

        return self._df._finalize(value, append=append, unit="")

    def learning_rate(self, name, performance, experience, append=False):
        """Compute the implicit learning rate from timeseries data

        Experience curves are based on the concept that a technology's performance
        improves as experience with this technology grows.

        The "learning rate" indicates the performance improvement (e.g., cost reduction)
        for each doubling of the accumulated experience (e.g., cumulative capacity).

        The experience curve parameter *b* is equivalent to the (linear) slope when
        plotting performance and experience timeseries on double-logarithmic scales.
        The learning rate can be computed from the experience curve parameter as
        :math:`1 - 2^{b}`.

        The learning rate parameter in period *t* is computed based on the changes
        to the subsequent period, i.e., from period *t* to period *t+1*.

        Parameters
        ----------
        name : str
            Variable name of the computed timeseries data.
        performance : str
            Variable of the "performance" timeseries (e.g., specific investment costs).
        experience : str
            Variable of the "experience" timeseries (e.g., installed capacity).
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.
        """
        value = (
            self._df._data[self._df._apply_filters(variable=[performance, experience])]
            .groupby(
                remove_from_list(self._df.dimensions, ["variable", "year", "unit"])
            )
            .apply(_compute_learning_rate, performance, experience)
        )

        return self._df._finalize(value, append=append, variable=name, unit="")

    def bias(self, name, method, axis):
        """Compute the bias weights and add to 'meta'

        Parameters
        ----------
        name : str
           Column name in the 'meta' dataframe
        method : str
            Method to compute the bias weights, see the notes
        axis : str
            Index dimensions on which to apply the `method`

        Notes
        -----

        The following methods are implemented:

        - "count": use the inverse of the number of scenarios grouped by `axis` names.

          Using the following method on an IamDataFrame with three scenarios

          .. code-block:: python

              df.compute.bias(name="bias-weight", method="count", axis="scenario")

          results in the following column to be added to *df.meta*:

          .. list-table::
             :header-rows: 1

             * - model
               - scenario
               - bias-weight
             * - model_a
               - scen_a
               - 0.5
             * - model_a
               - scen_b
               - 1
             * - model_b
               - scen_a
               - 0.5

        """
        _compute_bias(self._df, name, method, axis)

    def kaya_variables(self, append=False):
        """Create the set of variables needed to compute Kaya factors
        for the Kaya Decomposition Analysis.

        Parameters
        ----------
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        Notes
        -----

        Example of calling the method:

        .. code-block:: python

            df.compute.kaya_variables(append=True)

        The IamDataFrame must contain the following variables, otherwise the method
        will return None:
        .. list-table::
            - Required Variables
            - Population
            - GDP (MER or PPP)
            - Final Energy
            - Primary Energy
            - Primary Energy|Coal
            - Primary Energy|Oil
            - Primary Energy|Gas
            - Emissions|CO2|Industrial Processes
            - Emissions|CO2|Carbon Capture and Storage
            - Emissions|CO2|Carbon Capture and Storage|Biomass
            - Emissions|CO2|Fossil Fuels and Industry
            - Emissions|CO2|AFOLU
            - Carbon Sequestration|CCS|Fossil|Energy
            - Carbon Sequestration|CCS|Fossil|Industrial Processes
            - Carbon Sequestration|CCS|Biomass|Energy
            - Carbon Sequestration|CCS|Biomass|Industrial Processes

        """

        kaya_variables_frame = kaya_variables.kaya_variables(self._df)
        if kaya_variables_frame is None:
            return None
        if append:
            self._df.append(
                _find_non_duplicate_rows(self._df, kaya_variables_frame), inplace=True
            )
            return None

        return kaya_variables_frame

    def kaya_factors(self, append=False):
        """Compute the Kaya factors needed for the
        Kaya Decomposition Analysis.

        Parameters
        ----------
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        Notes
        -----

        Example of calling the method:

        .. code-block:: python

            df.compute.kaya_factors(append=True)

        The IamDataFrame must contain the following variables, otherwise the method
        will return None:
        .. list-table::
            - Required Variables
            - Population
            - GDP (MER or PPP)
            - Final Energy
            - Primary Energy
            - Primary Energy|Coal
            - Primary Energy|Oil
            - Primary Energy|Gas
            - Emissions|CO2|Industrial Processes
            - Emissions|CO2|Carbon Capture and Storage
            - Emissions|CO2|Carbon Capture and Storage|Biomass
            - Emissions|CO2|Fossil Fuels and Industry
            - Emissions|CO2|AFOLU
            - Carbon Sequestration|CCS|Fossil|Energy
            - Carbon Sequestration|CCS|Fossil|Industrial Processes
            - Carbon Sequestration|CCS|Biomass|Energy
            - Carbon Sequestration|CCS|Biomass|Industrial Processes

        """
        kaya_variables = self.kaya_variables(append=False)
        if kaya_variables is None:
            return None
        kaya_factors_frame = kaya_factors.kaya_factors(kaya_variables)
        if kaya_factors_frame is None:
            return None
        if append:
            self._df.append(
                _find_non_duplicate_rows(self._df, kaya_factors_frame), inplace=True
            )
        return kaya_factors_frame

    def kaya_lmdi(self, ref_scenario, int_scenario, append=False):
        """Calculate the logarithmic mean Divisia index (LMDI) decomposition
        using Kaya factors.

        Parameters
        ----------
        ref_scenario : tuple of strings (model, scenario, region)
            The (model, scenario, region) to be used as the reference scenario
            in the LMDI calculation.
        int_scenario : tuple of strings (model, scenario, region)
            The (model, scenario, region) to be used as the intervention scenario
            in the LMDI calculation.
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        Notes
        -----

        Example of calling the method:

        .. code-block:: python

            df.compute.kaya_lmdi(
                ref_scenario=("model_a", "scenario_a", "region_a"),
                int_scenario=("model_b", "scenario_b", "region_b"),
                append=True,
            )

        The IamDataFrame must contain the following variables, otherwise the method
        will return None:
        .. list-table::
            - Required Variables
            - Population
            - GDP (MER or PPP)
            - Final Energy
            - Primary Energy
            - Primary Energy|Coal
            - Primary Energy|Oil
            - Primary Energy|Gas
            - Emissions|CO2|Industrial Processes
            - Emissions|CO2|Carbon Capture and Storage
            - Emissions|CO2|Carbon Capture and Storage|Biomass
            - Emissions|CO2|Fossil Fuels and Industry
            - Emissions|CO2|AFOLU
            - Carbon Sequestration|CCS|Fossil|Energy
            - Carbon Sequestration|CCS|Fossil|Industrial Processes
            - Carbon Sequestration|CCS|Biomass|Energy
            - Carbon Sequestration|CCS|Biomass|Industrial Processes

        The model, scenario, and region fields for the results dataframe will be
        concatenated values from the reference and intervention scenarios in the
        form reference_scenario_value::intervention_scenario_value.

        Example results data:

            model	            scenario	    region	        variable	        unit	year	value
            model_a::model_a	scen_a::scen_b	World::World	FE/GNP (LMDI)	    unknown	2010	1.321788
            model_a::model_a	scen_a::scen_b	World::World	GNP/P (LMDI)	    unknown	2010	0.000000
            model_a::model_a	scen_a::scen_b	World::World	PEDEq/FE (LMDI)	    unknown	2010	0.816780
            model_a::model_a	scen_a::scen_b	World::World	PEFF/PEDEq (LMDI)	unknown	2010	0.000000
            model_a::model_a	scen_a::scen_b	World::World	Population (LMDI)	unknown	2010	0.000000
            model_a::model_a	scen_a::scen_b	World::World	TFC/PEFF (LMDI)	    unknown	2010	4.853221

        """
        valid_ref_and_int_scenarios = _validate_kaya_scenario_args(
            scenarios=[ref_scenario, int_scenario]
        )
        # we must have two different scenarios to calculate kaya_lmdi
        if (valid_ref_and_int_scenarios is None) or (
            len(valid_ref_and_int_scenarios) != 2
        ):
            return None
        kaya_factors = self.kaya_factors(valid_ref_and_int_scenarios, append=False)
        if kaya_factors is None:
            return None
        kaya_lmdi_frame = lmdi.corrected_lmdi(kaya_factors, ref_scenario, int_scenario)
        if kaya_lmdi_frame is None:
            return None
        if append:
            self._df.append(
                _find_non_duplicate_rows(self._df, kaya_lmdi_frame), inplace=True
            )
        return kaya_lmdi_frame


def _validate_kaya_scenario_args(scenarios):
    validated_scenarios = []
    for scenario in scenarios:
        if (len(scenario) == 3) and _kaya_args_are_strings(scenario):
            validated_scenarios.append(scenario)
    # don't recalculate for identical scenarios
    unique_scenarios = set(scenarios)
    if len(unique_scenarios) == 0:
        return None
    return validated_scenarios


def _kaya_args_are_strings(scenario):
    for arg in scenario:
        if not isinstance(arg, str):
            return False
    return True


def _find_non_duplicate_rows(original_df, variables_to_add):
    variables_for_append = pyam.IamDataFrame(
        variables_to_add.as_pandas(meta_cols=False)
        .merge(original_df.as_pandas(meta_cols=False), how="left", indicator=True)
        .query('_merge=="left_only"')
        .drop(columns="_merge")
    )
    return variables_for_append


def _compute_learning_rate(x, performance, experience):
    """Internal implementation for computing implicit learning rate from timeseries data

    Parameters
    ----------
    x : :class:`pandas.Series`
        Timeseries data of the *performance* and *experience* variables
        indexed over the time domain.
    performance : str
        Variable of the "performance" timeseries (e.g., specific investment costs).
    experience : str
        Variable of the "experience" timeseries (e.g., cumulative installed capacity).

    Returns
    -------
    Indexed :class:`pandas.Series` of implicit learning rates
    """
    # drop all index dimensions other than "variable" and "year"
    x.index = x.index.droplevel(
        [i for i in x.index.names if i not in ["variable", "year"]]
    )

    # apply log, dropping all values that are zero or negative
    x = x[x > 0].apply(math.log10)

    # return empty pd.Series if not all relevant variables exist
    if not all([v in x.index for v in [performance, experience]]):
        return empty_series(remove_from_list(x.index.names, "variable"))

    # compute the "experience parameter" (slope of experience curve on double-log scale)
    b = (x[performance] - x[performance].shift(periods=-1)) / (
        x[experience] - x[experience].shift(periods=-1)
    )

    # translate to "learning rate" (e.g., cost reduction per doubling of capacity)
    return b.apply(lambda y: 1 - math.pow(2, y))


def empty_series(names):
    """Return an empty pd.Series with correct index names"""
    empty_list = [[]] * len(names)
    return pd.Series(
        index=pd.MultiIndex(levels=empty_list, codes=empty_list, names=names),
        dtype="float64",
    )
