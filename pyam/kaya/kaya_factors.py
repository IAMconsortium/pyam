from functools import reduce

from pyam.kaya import input_variable_names, kaya_factor_names, kaya_variable_names


def kaya_factors(kaya_variables_frame, scenarios):
    kaya_factors_frames = []
    for scenario in scenarios:
        input = kaya_variables_frame.filter(
            model=scenario[0], scenario=scenario[1], region=scenario[2]
        )
        if input.empty:
            break
        kaya_factors_frames.append(_calc_gnp_per_p(input))
        kaya_factors_frames.append(_calc_fe_per_gnp(input))
        kaya_factors_frames.append(_calc_pedeq_per_fe(input))
        kaya_factors_frames.append(_calc_peff_per_pedeq(input))
        kaya_factors_frames.append(_calc_tfc_per_peff(input))
        kaya_factors_frames.append(_calc_nfc_per_tfc(input))
        kaya_factors_frames.append(
            input.filter(
                variable=[kaya_variable_names.TFC, input_variable_names.POPULATION]
            )
        )
    if len(kaya_factors_frames) == 0:
        return None
    return reduce(lambda x, y: x.append(y), kaya_factors_frames)


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
        ignore_units=True,
        append=False,
    ).rename(unit={"unknown": ""})
