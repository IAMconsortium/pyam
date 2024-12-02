import logging
import warnings

import pyam
from pyam.kaya import input_variable_names, kaya_variable_names

logger = logging.getLogger(__name__)

required_input_variables = [
    vars(input_variable_names)[variable_name]
    for variable_name in dir(input_variable_names)
    if not variable_name.startswith("__")
]


def kaya_variables(input_data):
    if _is_input_data_incomplete(input_data):
        return None

    kaya_variables = pyam.concat(
        [
            _calc_pop(input_data),
            _calc_gdp(input_data),
            _calc_fe(input_data),
            _calc_pe(input_data),
            _calc_pe_ff(input_data),
            _calc_tfc(input_data),
            _calc_nfc(input_data),
        ]
    )
    return kaya_variables


def _is_input_data_incomplete(input_data):
    # copy data so we don't create side effects
    # in particular, require_data will change the "exclude" series
    input_data = input_data.copy()
    # Get all unique model/scenario/region combinations
    scenario_model_region = input_data.data[
        ["model", "scenario", "region"]
    ].drop_duplicates()

    # Check each combination
    for _, row in scenario_model_region.iterrows():
        single_combination = input_data.filter(
            model=row["model"], scenario=row["scenario"], region=row["region"]
        )

        # Get variables present for this combination
        single_combination_variables = set(single_combination.data["variable"].unique())
        # special case for GDP: either form is acceptable, so don't check for either
        # as long as one is present
        required_variables_set = make_required_variables_set(
            single_combination_variables
        )
        # Check if any required variables are missing
        missing_variables = set(required_variables_set) - single_combination_variables

        if missing_variables is not None:
            logger.info(
                f"Variables missing for model: {row['model']}, scenario: {row['scenario']}, region: {row['region']}:"
                f"\n{missing_variables}"
            )

    # special case for GDP: either form is acceptable, so don't check for either
    # as long as one is present
    required_variables_set = make_required_variables_set(
        set(input_data.data["variable"].unique())
    )
    # exclude model/scenario combinations that have missing variables, disregarding region
    # even if all variables are not present for a region, arithmetic operations
    # will return an empty dataframe, not throw an error, so it is safe to proceed
    input_data.require_data(variable=required_input_variables, exclude_on_fail=True)
    # supress warning about empty dataframe if filtering excludes all scenarios
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return input_data.filter(exclude=False).empty


def make_required_variables_set(input_variables):
    required_variables_set = set(required_input_variables)
    if not _missing_gdp(input_variables):
        # either form of GDP is acceptable, so don't check for both
        # as long as one is present
        return required_variables_set - set(
            [input_variable_names.GDP_PPP, input_variable_names.GDP_MER]
        )
    return required_variables_set


def _missing_gdp(input_variables):
    return (
        input_variable_names.GDP_PPP in input_variables
        or input_variable_names.GDP_MER in input_variables
    )


def _calc_pop(input_data):
    return input_data.filter(variable=input_variable_names.POPULATION)


def _calc_gdp(input_data):
    variable = input_variable_names.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variable_names.GDP_MER
    return input_data.filter(variable=variable)


def _calc_fe(input_data):
    return input_data.filter(variable=input_variable_names.FINAL_ENERGY)


def _calc_pe(input_data):
    return input_data.filter(variable=input_variable_names.PRIMARY_ENERGY)


def _calc_pe_ff(input_data):
    input_data = input_data.copy()
    input_data.add(
        input_variable_names.PRIMARY_ENERGY_COAL,
        input_variable_names.PRIMARY_ENERGY_OIL,
        "pe_coal_oil",
        append=True,
    )
    return input_data.add(
        input_variable_names.PRIMARY_ENERGY_GAS,
        "pe_coal_oil",
        kaya_variable_names.PRIMARY_ENERGY_FF,
    )


def _calc_nfc(input_data):
    input_data = input_data.copy()
    input_data.subtract(
        input_variable_names.EMISSIONS_CO2_FOSSIL_FUELS_AND_INDUSTRY,
        input_variable_names.EMISSIONS_CO2_INDUSTRIAL_PROCESSES,
        "net_energy_emissions_with_biomass_ccs",
        ignore_units="Mt CO2/yr",
        append=True,
    )
    return input_data.add(
        input_variable_names.EMISSIONS_CO2_CCS_BIOMASS,
        "net_energy_emissions_with_biomass_ccs",
        kaya_variable_names.NFC,
        ignore_units="Mt CO2/yr",
        append=False,
    )


def _calc_tfc(input_data):
    input_data = input_data.copy()
    ccs_fossil_energy = _calc_ccs_fossil_energy(input_data)
    nfc = _calc_nfc(input_data)
    nfc_with_ccs_fossil_energy = nfc.append(ccs_fossil_energy)
    return nfc_with_ccs_fossil_energy.add(
        "ccs_fossil_energy",
        kaya_variable_names.NFC,
        kaya_variable_names.TFC,
        ignore_units="Mt CO2/yr",
    )


def _calc_ccs_fossil_energy(input_data):
    input_data = input_data.copy()
    input_data.subtract(
        input_variable_names.EMISSIONS_CO2_CCS,
        input_variable_names.EMISSIONS_CO2_CCS_BIOMASS,
        "ccs_fossil",
        ignore_units="Mt CO2/yr",
        append=True,
    )
    return input_data.subtract(
        "ccs_fossil",
        input_variable_names.CCS_FOSSIL_INDUSTRY,
        "ccs_fossil_energy",
        ignore_units="Mt CO2/yr",
    )
