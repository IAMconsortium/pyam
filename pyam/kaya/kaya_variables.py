import logging
from functools import reduce

from pyam.kaya import input_variable_names, kaya_variable_names

logger = logging.getLogger(__name__)

input_variable_list = [
    vars(input_variable_names)[variable_name]
    for variable_name in dir(input_variable_names)
    if not variable_name.startswith("__")
]


def kaya_variables(input_data, scenarios):
    # copy data so we don't create side effects
    # in particular, require_data will change the "exclude" series
    input_data = input_data.copy()
    validated_input_data = _validate_input_data(input_data)
    if validated_input_data.empty:
        return None
    kaya_variable_frames = []
    for scenario in scenarios:
        input = validated_input_data.filter(
            model=scenario[0], scenario=scenario[1], region=scenario[2]
        )
        if input.empty:
            break
        kaya_variable_frames.append(_calc_pop(input))
        kaya_variable_frames.append(_calc_gdp(input))
        kaya_variable_frames.append(_calc_fe(input))
        kaya_variable_frames.append(_calc_pe(input))
        kaya_variable_frames.append(_calc_pe_ff(input))
        kaya_variable_frames.append(_calc_tfc(input))
        kaya_variable_frames.append(_calc_nfc(input))

    if len(kaya_variable_frames) == 0:
        return None
    # append all the IamDataFrames into one
    return reduce(lambda x, y: x.append(y), kaya_variable_frames)


def _validate_input_data(input_data):
    missing_variables = input_data.require_data(
        variable=input_variable_list, exclude_on_fail=True
    )
    if missing_variables is not None:
        logger.info(
            f"These variables are missing from the \
            scenarios in input_data:\n{missing_variables}"
        )
    return input_data.filter(exclude=False)


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
