from functools import reduce

import numpy as np
import pandas as pd

import pyam
from pyam.kaya import (
    input_variable_names,
    kaya_factor_names,
    kaya_variable_names,
    lmdi_names,
)


def corrected_lmdi(kaya_factors, ref_scenario, int_scenario):
    ref_input = (
        kaya_factors.filter(
            model=ref_scenario[0], scenario=ref_scenario[1], region=ref_scenario[2]
        )
        .as_pandas()
        .assign(scenario_class="reference")
    )
    int_input = (
        kaya_factors.filter(
            model=int_scenario[0], scenario=int_scenario[1], region=int_scenario[2]
        )
        .as_pandas()
        .assign(scenario_class="intervention")
    )
    input = pyam.IamDataFrame(pd.concat([ref_input, int_input]))

    uncorrected = _uncorrected_lmdi(input)
    non_neg = _lmdi_non_neg(uncorrected)
    total_non_neg = _sum_lmdi_non_neg(non_neg)
    total_w_neg = _tfc_diff(input)
    difference = total_non_neg.append(total_w_neg).subtract(
        "total_no_neg", "tfc_diff", "difference", append=False, ignore_units=True
    )

    lmdi_frames = []
    p_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.Pop_LMDI, total_non_neg
    )
    p_correction = p_percent.append(difference).multiply(
        lmdi_names.Pop_LMDI, "difference", "correction", ignore_units=True
    )
    p_corrected = p_correction.append(non_neg).add(
        lmdi_names.Pop_LMDI, "correction", lmdi_names.Pop_LMDI, ignore_units=True
    )
    lmdi_frames.append(p_corrected)

    gnp_per_p_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.GNP_per_P_LMDI, total_non_neg
    )
    gnp_per_p_correction = gnp_per_p_percent.append(difference).multiply(
        lmdi_names.GNP_per_P_LMDI, "difference", "correction", ignore_units=True
    )
    gnp_per_p_corrected = gnp_per_p_correction.append(non_neg).add(
        lmdi_names.GNP_per_P_LMDI,
        "correction",
        lmdi_names.GNP_per_P_LMDI,
        ignore_units=True,
    )
    lmdi_frames.append(gnp_per_p_corrected)

    fe_per_gnp_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.FE_per_GNP_LMDI, total_non_neg
    )
    fe_per_gnp_correction = fe_per_gnp_percent.append(difference).multiply(
        lmdi_names.FE_per_GNP_LMDI, "difference", "correction", ignore_units=True
    )
    fe_per_gnp_corrected = fe_per_gnp_correction.append(non_neg).add(
        lmdi_names.FE_per_GNP_LMDI,
        "correction",
        lmdi_names.FE_per_GNP_LMDI,
        ignore_units=True,
    )
    lmdi_frames.append(fe_per_gnp_corrected)

    pedeq_per_fe_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.PEdeq_per_FE_LMDI, total_non_neg
    )
    pedeq_per_fe_correction = pedeq_per_fe_percent.append(difference).multiply(
        lmdi_names.PEdeq_per_FE_LMDI, "difference", "correction", ignore_units=True
    )
    pedeq_per_fe_corrected = pedeq_per_fe_correction.append(non_neg).add(
        lmdi_names.PEdeq_per_FE_LMDI,
        "correction",
        lmdi_names.PEdeq_per_FE_LMDI,
        ignore_units=True,
    )
    lmdi_frames.append(pedeq_per_fe_corrected)

    peff_per_pedeq_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.PEFF_per_PEDEq_LMDI, total_non_neg
    )
    peff_per_pedeq_correction = peff_per_pedeq_percent.append(difference).multiply(
        lmdi_names.PEFF_per_PEDEq_LMDI, "difference", "correction", ignore_units=True
    )
    peff_per_pedeq_corrected = peff_per_pedeq_correction.append(non_neg).add(
        lmdi_names.PEFF_per_PEDEq_LMDI,
        "correction",
        lmdi_names.PEFF_per_PEDEq_LMDI,
        ignore_units=True,
    )
    lmdi_frames.append(peff_per_pedeq_corrected)

    tfc_per_peff_percent = _calc_percent_of_total_for_one_term(
        non_neg, lmdi_names.TFC_per_PEFF_LMDI, total_non_neg
    )
    tfc_per_peff_correction = tfc_per_peff_percent.append(difference).multiply(
        lmdi_names.TFC_per_PEFF_LMDI, "difference", "correction", ignore_units=True
    )
    tfc_per_peff_corrected = tfc_per_peff_correction.append(non_neg).add(
        lmdi_names.TFC_per_PEFF_LMDI,
        "correction",
        lmdi_names.TFC_per_PEFF_LMDI,
        ignore_units=True,
    )
    lmdi_frames.append(tfc_per_peff_corrected)

    full_lmdi = reduce(lambda x, y: x.append(y), lmdi_frames)
    full_lmdi_no_scenario_class_column = pyam.IamDataFrame(
        full_lmdi.as_pandas().drop(columns="scenario_class")
    )
    return full_lmdi_no_scenario_class_column


def _lmdi_non_neg(uncorrected):
    p_non_neg = _calc_one_non_negative_term(uncorrected, lmdi_names.Pop_LMDI)
    gnp_per_p_non_neg = _calc_one_non_negative_term(
        uncorrected, lmdi_names.GNP_per_P_LMDI
    )
    fe_per_gnp_non_neg = _calc_one_non_negative_term(
        uncorrected, lmdi_names.FE_per_GNP_LMDI
    )
    pedeq_per_fe_non_neg = _calc_one_non_negative_term(
        uncorrected, lmdi_names.PEdeq_per_FE_LMDI
    )
    peff_per_pedeq_non_neg = _calc_one_non_negative_term(
        uncorrected, lmdi_names.PEFF_per_PEDEq_LMDI
    )
    tfc_per_peff_non_neg = _calc_one_non_negative_term(
        uncorrected, lmdi_names.TFC_per_PEFF_LMDI
    )

    return (
        p_non_neg.append(gnp_per_p_non_neg)
        .append(fe_per_gnp_non_neg)
        .append(pedeq_per_fe_non_neg)
        .append(peff_per_pedeq_non_neg)
        .append(tfc_per_peff_non_neg)
    )


def _sum_lmdi_non_neg(lmdi_non_neg):
    lmdi_non_neg.add(
        lmdi_names.Pop_LMDI,
        lmdi_names.GNP_per_P_LMDI,
        "sum_to_GNP_per_P_LMDI",
        append=True,
        ignore_units=True,
    )
    lmdi_non_neg.add(
        "sum_to_GNP_per_P_LMDI",
        lmdi_names.FE_per_GNP_LMDI,
        "sum_to_FE_per_GNP_LMDI",
        append=True,
        ignore_units=True,
    )
    lmdi_non_neg.add(
        "sum_to_FE_per_GNP_LMDI",
        lmdi_names.PEdeq_per_FE_LMDI,
        "sum_to_PEdeq_per_FE_LMDI",
        append=True,
        ignore_units=True,
    )
    lmdi_non_neg.add(
        "sum_to_PEdeq_per_FE_LMDI",
        lmdi_names.PEFF_per_PEDEq_LMDI,
        "sum_to_PEFF_per_PEDEq_LMDI",
        append=True,
        ignore_units=True,
    )
    return lmdi_non_neg.add(
        "sum_to_PEFF_per_PEDEq_LMDI",
        lmdi_names.TFC_per_PEFF_LMDI,
        "total_no_neg",
        append=False,
        ignore_units=True,
    )


def _calc_percent_of_total_for_one_term(non_neg, lmdi_term_name, tfc_diff):
    return non_neg.append(tfc_diff).divide(
        lmdi_term_name, "total_no_neg", lmdi_term_name, ignore_units=True
    )


def _tfc_diff(kaya_factors):

    (combined_model_name, combined_scenario_name, combined_region_name) = (
        _make_combined_scenario_name(kaya_factors.as_pandas())
    )
    tfc = (
        kaya_factors.filter(
            variable=kaya_variable_names.TFC, scenario_class="reference"
        )
        .rename(variable={kaya_variable_names.TFC: "tfc_ref"})
        .append(
            kaya_factors.filter(
                variable=kaya_variable_names.TFC, scenario_class="intervention"
            )
        )
    )
    tfc = pyam.IamDataFrame(
        tfc.as_pandas()
        .assign(scenario_class="LMDI")
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
    )
    return tfc.subtract(
        "tfc_ref", kaya_variable_names.TFC, "tfc_diff", ignore_units=True
    )


def _calc_one_non_negative_term(uncorrected_lmdi, lmdi_term_name):
    return uncorrected_lmdi.apply(
        _remove_negative, lmdi_term_name, args=[lmdi_term_name], ignore_units=True
    )


def _remove_negative(lmdi_term):
    return lmdi_term.clip(lower=0)


def _uncorrected_lmdi(kaya_factors):

    p = _calc_one_lmdi_term(
        kaya_factors, input_variable_names.POPULATION, lmdi_names.Pop_LMDI
    )
    gnp_per_p = _calc_one_lmdi_term(
        kaya_factors, kaya_factor_names.GNP_per_P, lmdi_names.GNP_per_P_LMDI
    )
    fe_per_gnp = _calc_one_lmdi_term(
        kaya_factors, kaya_factor_names.FE_per_GNP, lmdi_names.FE_per_GNP_LMDI
    )
    pe_deq_per_fe = _calc_one_lmdi_term(
        kaya_factors, kaya_factor_names.PEdeq_per_FE, lmdi_names.PEdeq_per_FE_LMDI
    )
    peff_per_pe_deq = _calc_one_lmdi_term(
        kaya_factors, kaya_factor_names.PEFF_per_PEDEq, lmdi_names.PEFF_per_PEDEq_LMDI
    )
    tfc_per_peff = _calc_one_lmdi_term(
        kaya_factors, kaya_factor_names.TFC_per_PEFF, lmdi_names.TFC_per_PEFF_LMDI
    )
    return (
        p.append(gnp_per_p)
        .append(fe_per_gnp)
        .append(pe_deq_per_fe)
        .append(peff_per_pe_deq)
        .append(tfc_per_peff)
    )


def _calc_one_lmdi_term(
    input_data,
    kaya_factor_name,
    lmdi_term_name,
    kaya_product_name=kaya_variable_names.TFC,
):
    return input_data.apply(
        _lmdi,
        lmdi_term_name,
        axis="variable",
        args=[kaya_factor_name, kaya_product_name],
        ignore_units=True,
    )


def _lmdi(kaya_factor, kaya_product):

    (combined_model_name, combined_scenario_name, combined_region_name) = (
        _make_combined_scenario_name(kaya_factor)
    )

    factor_ref = (
        kaya_factor.reset_index()
        .query('scenario_class == "reference"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )

    factor_int = (
        kaya_factor.reset_index()
        .query('scenario_class == "intervention"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )
    tfc_ref = (
        kaya_product.reset_index()
        .query('scenario_class == "reference"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )
    tfc_int = (
        kaya_product.reset_index()
        .query('scenario_class == "intervention"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )
    return (
        ((tfc_ref - tfc_int) / (np.log(tfc_ref) - np.log(tfc_int)))
        * (np.log(factor_ref / factor_int))
    ).squeeze(axis=1)


def _make_combined_scenario_name(kaya_factor):
    ref = kaya_factor.reset_index().query('scenario_class == "reference"')
    int = kaya_factor.reset_index().query('scenario_class == "intervention"')

    ref_model_name = ref.model.values[0]
    int_model_name = int.model.values[0]

    ref_scenario_name = ref.scenario.values[0]
    int_scenario_name = int.scenario.values[0]

    ref_region_name = ref.region.values[0]
    int_region_name = int.region.values[0]

    return (
        ref_model_name + "::" + int_model_name,
        ref_scenario_name + "::" + int_scenario_name,
        ref_region_name + "::" + int_region_name,
    )
