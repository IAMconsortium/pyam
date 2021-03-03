import pandas as pd
import numpy.testing as npt
import pint
import pytest
from pyam import IamDataFrame


def get_units_test_df(test_df):
    # modify units in standard test dataframe
    df = test_df.data
    df.loc[0:1, "unit"] = "custom_unit"
    return IamDataFrame(df)


def assert_converted_units(df, current, to, exp, **kwargs):
    # testing for `inplace=False` - converted values and expected unit
    _df = df.convert_unit(current, to, **kwargs, inplace=False)

    npt.assert_allclose(_df._data.values, exp.values, rtol=1e-4)
    # When *to* is only a species symbol (e.g. 'co2e'), units are added and a
    # non-aliased symbol is returned (e.g. 'Mt CO2e'). Compare using 'in' and
    # lower().
    assert to.lower() in _df._data.index.get_level_values("unit")[5].lower()

    # testing for `inplace=True` - converted values and expected unit
    df.convert_unit(current, to, **kwargs, inplace=True)
    npt.assert_allclose(df._data, exp.values, rtol=1e-4)
    assert to.lower() in df.data.unit[5].lower()


@pytest.mark.parametrize("current,to", [("EJ/yr", "TWh/yr"), ("EJ", "TWh")])
def test_convert_unit_with_pint(test_df, current, to):
    """Convert unit with default UnitRegistry (i.e, application_registry)"""
    df = get_units_test_df(test_df)

    # replace EJ/yr by EJ to test pint with single unit
    if current == "EJ":
        df.rename(unit={"EJ/yr": "EJ"}, inplace=True)

    exp = pd.Series([1.0, 6.0, 138.88, 833.33, 555.55, 1944.44], name="value")
    assert_converted_units(df, current, to, exp)


def test_convert_unit_from_repo(test_df):
    """Convert unit with definition loaded from `IAMconsortium/units` repo"""
    df = get_units_test_df(test_df)
    exp = pd.Series([1.0, 6.0, 17.06, 102.361, 68.241, 238.843], name="value")
    assert_converted_units(df, "EJ/yr", "Mtce/yr", exp)


def test_convert_unit_with_custom_registry(test_df):
    """Convert unit conversion with custom UnitRegistry"""
    df = get_units_test_df(test_df).rename(unit={"EJ/yr": "foo"})

    # check that conversion fails with application registry
    with pytest.raises(pint.UndefinedUnitError):
        df.convert_unit("foo", "baz")

    # define a custom unit registry
    ureg = pint.UnitRegistry()
    ureg.define("baz = [custom]")
    ureg.define("foo =  3 * baz")

    exp = pd.Series([1.0, 6.0, 1.5, 9, 6, 21], name="value")
    assert_converted_units(df, "foo", "baz", exp, registry=ureg)


# This test is parametrized as the product of three sets:
# 1. The test_df fixture.
# 2. Current species, context, and expected output magnitude.
# 3. Input and output expressions, and any factor on the output magnitude due
#    to differences in units between these.
@pytest.mark.parametrize(
    "context, current_species, exp",
    [
        ("AR5GWP100", "CH4", 28),
        ("AR4GWP100", "CH4", 25),
        ("SARGWP100", "CH4", 21),
        # Without context, CO2e → CO2e works
        (None, "CO2e", 1.0),
        # Lower-case symbol, handled as alias for CH4
        ("AR5GWP100", "ch4", 28),
        # Lower-case alias for CO2_eq and 'co2-equiv' handled *and* convertible to
        # 'CO2e' without a context/metric
        (None, "co2_eq", 1.0),
        (None, "co2-equiv", 1.0),
        # Using "-equiv" after a unit should make no difference
        ("AR4GWP100", "co2-equiv", 1.0),
        ("AR5GWP100", "ch4-equiv", 28),
        # Converting C -> CO2e should work
        # TODO remove context once IAMconsortium/units#23 is resolved
        ("AR4GWP100", "C", 11 / 3),
    ],
)
@pytest.mark.parametrize(
    "current_expr, to_expr, exp_factor",
    [
        # exp_factor is used when the conversion includes both a species *and* unit
        # change.
        # Conversions where the *to* argument contains a mass unit
        ("g {}", "g {}", 1),
        ("Mt {}", "Mt {}", 1),
        ("Mt {} / yr", "Mt {} / yr", 1),
        ("g {} / sec", "g {} / sec", 1),
        # Only a species symbol as the *to* argument
        ("Mt {}", "{}", 1),
        # *to* contains units, but no mass units. UndefinedUnitError when no
        # context is given, otherwise DimensionalityError.
        pytest.param(
            "Mt {} / yr",
            "{} / yr",
            1,
            marks=pytest.mark.xfail(
                raises=(pint.UndefinedUnitError, pint.DimensionalityError)
            ),
        ),
        # *to* contains both species *and* mass units that are different than
        # *current*
        ("t {} / year", "kt {} / year", 1e-3),
    ],
)
def test_convert_gwp(
    test_df, context, current_species, current_expr, to_expr, exp, exp_factor
):
    """Units and GHG species can be converted."""
    # Handle parameters
    current = current_expr.format(current_species)
    to = to_expr.format("CO2e")

    # Expected values
    exp_values = test_df._data.copy()
    exp_values[[False, False, True, True, True, True]] *= exp * exp_factor

    # Prepare test data and assert cenverted units
    df = get_units_test_df(test_df).rename(unit={"EJ/yr": current})
    assert_converted_units(df.copy(), current, to, exp_values, context=context)


def test_convert_unit_bad_args(test_pd_df):
    """Unit conversion with bad arguments raises errors."""
    idf = IamDataFrame(test_pd_df).rename(unit={"EJ/yr": "Mt CH4"})

    # Conversion fails with both *factor* and *registry*
    with pytest.raises(ValueError, match="Use either `factor` or `registry`!"):
        idf.convert_unit("Mt CH4", "CO2e", factor=1.0, registry=object())

    # Conversion fails with an invalid registry
    with pytest.raises(TypeError, match="must be `pint.UnitRegistry`"):
        idf.convert_unit("Mt CH4", "CO2e", registry=object())

    # Conversion fails without context; exception provides a usage hint
    match = r"GWP conversion with IamDataFrame.convert_unit\(\) requires..."
    with pytest.raises(pint.UndefinedUnitError, match=match):
        idf.convert_unit("Mt CH4", "CO2e")


def test_convert_unit_with_custom_factor(test_df):
    """Convert units with custom factors."""
    # unit conversion with custom factor
    df = get_units_test_df(test_df)
    exp = pd.Series([1.0, 6.0, 1.0, 6.0, 4.0, 14.0], name="value")
    assert_converted_units(df, "EJ/yr", "foo", exp, factor=2)
