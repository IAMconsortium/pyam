import pyam
from pyam.kaya import input_variable_names, kaya_factor_names, kaya_variable_names


def kaya_factors(kaya_variables_frame):
    kaya_factors = pyam.concat(
        [
            _calc_gnp_per_p(kaya_variables_frame),
            _calc_fe_per_gnp(kaya_variables_frame),
            _calc_pedeq_per_fe(kaya_variables_frame),
            _calc_peff_per_pedeq(kaya_variables_frame),
            _calc_tfc_per_peff(kaya_variables_frame),
            _calc_nfc_per_tfc(kaya_variables_frame),
            kaya_variables_frame.filter(
                variable=[kaya_variable_names.TFC, input_variable_names.POPULATION]
            ),
        ]
    )
    return kaya_factors


def _calc_gnp_per_p(input_data):
    variable = input_variable_names.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variable_names.GDP_MER
    return input_data.divide(
        variable,
        input_variable_names.POPULATION,
        kaya_factor_names.GNP_per_P,
        append=False,
    )


def _calc_fe_per_gnp(input_data):
    variable = input_variable_names.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variable_names.GDP_MER
    return input_data.divide(
        input_variable_names.FINAL_ENERGY,
        variable,
        kaya_factor_names.FE_per_GNP,
        append=False,
    )


def _calc_pedeq_per_fe(input_data):
    return input_data.divide(
        input_variable_names.PRIMARY_ENERGY,
        input_variable_names.FINAL_ENERGY,
        kaya_factor_names.PEdeq_per_FE,
        append=False,
    )


def _calc_peff_per_pedeq(input_data):
    return input_data.divide(
        kaya_variable_names.PRIMARY_ENERGY_FF,
        input_variable_names.PRIMARY_ENERGY,
        kaya_factor_names.PEFF_per_PEDEq,
        append=False,
    )


def _calc_tfc_per_peff(input_data):
    return input_data.divide(
        kaya_variable_names.TFC,
        kaya_variable_names.PRIMARY_ENERGY_FF,
        kaya_factor_names.TFC_per_PEFF,
        ignore_units="Mt CO2/EJ",
        append=False,
    )


def _calc_nfc_per_tfc(input_data):
    return input_data.divide(
        kaya_variable_names.NFC,
        kaya_variable_names.TFC,
        kaya_factor_names.NFC_per_TFC,
        ignore_units="",
        append=False,
    )  # .rename(unit={"unknown": ""})
