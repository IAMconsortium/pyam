import operator

import pandas as pd
import pint
import pytest
from iam_units import registry

from pyam import IamDataFrame
from pyam.operations import apply_ops
from pyam.testing import assert_iamframe_equal
from pyam.utils import IAMC_IDX

DF_INDEX = ["scenario", 2005, 2010]
# dictionary with common IamDataFrame args for all tests operating on variable
DF_ARGS = dict(model="model_a", region="World")


def df_ops_variable(func, variable, unit, meta):
    """Return IamDataFrame when performing operation on test_df without default"""
    _data = pd.DataFrame(["scen_a", func(1, 0.5), func(6, 3)], index=DF_INDEX)
    return IamDataFrame(_data.T, **DF_ARGS, variable=variable, unit=unit, meta=meta)


def df_ops_variable_default(func, variable, unit, meta):
    """Return IamDataFrame when performing operation on test_df with default (5)"""
    _data = pd.DataFrame(
        [["scen_a", "scen_b"], [func(1, 0.5), func(2, 5)], [func(6, 3), func(7, 5)]],
        index=DF_INDEX,
    )
    return IamDataFrame(_data.T, **DF_ARGS, variable=variable, unit=unit, meta=meta)


def df_ops_variable_number(func, variable, unit, meta):
    """Return IamDataFrame when performing operation on test_df with a number (2)"""
    _data = pd.DataFrame(
        [["scen_a", "scen_b"], [func(1, 2), func(2, 2)], [func(6, 2), func(7, 2)]],
        index=DF_INDEX,
    )
    return IamDataFrame(_data.T, **DF_ARGS, variable=variable, unit=unit, meta=meta)


def df_ops_fillna_0(func, variable, unit, meta):
    """Return IamDataFrame when performing operation on test_df with default (5)"""
    _data = pd.DataFrame(
        [["scen_a", "scen_b"], [func(1, 0.5), func(2, 0)], [func(6, 3), func(7, 0)]],
        index=DF_INDEX,
    )
    return IamDataFrame(_data.T, **DF_ARGS, variable=variable, unit=unit, meta=meta)


def test_add_raises(test_df_year):
    """Calling an operation with args that don't return an IamDataFrame raises"""
    match = "Value returned by `add` cannot be cast to an IamDataFrame: 5"
    with pytest.raises(ValueError, match=match):
        test_df_year.add(2, 3, "foo")


@pytest.mark.parametrize(
    "variable, df_func, expected_unit",
    (
        ("Primary Energy|Coal", df_ops_variable, "EJ/yr"),
        # pint changes the unit into its standard format
        (registry.Quantity(2, "EJ/yr"), df_ops_variable_number, "EJ / a"),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_add_variable(test_df_year, variable, df_func, expected_unit, append):
    """Check that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.add, "Sum", unit=expected_unit, meta=test_df_year.meta)

    if append:
        obs = test_df_year.copy()
        obs.add("Primary Energy", variable, "Sum", append=True)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.add("Primary Energy", variable, "Sum")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_fillna_0, 0),
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_add_variable_ignore_units(test_df_year, arg, df_func, fillna, append):
    """Check that in-dataframe addition works with ignore_units"""

    # change one unit to make ignore_units strictly necessary
    test_df_year.rename(
        variable={"Primary Energy": "Primary Energy"},
        unit={"EJ/yr": "custom_unit"},
        inplace=True,
    )

    exp = df_func(operator.add, "Sum", unit="foo", meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Sum")
    if append:
        obs = test_df_year.copy()
        obs.add(*args, ignore_units="foo", fillna=fillna, append=True)
        exp = test_df_year.append(exp)
    else:
        # check that incompatible units raise the expected error
        with pytest.raises(pint.UndefinedUnitError):
            test_df_year.add(*args, fillna=fillna)

        # using ignore_units works as expected
        obs = test_df_year.add(*args, ignore_units="foo", fillna=fillna)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_add_variable_non_si_unit(test_df_year, append):
    """Check that in-dataframe addition works with non-SI-units"""
    df = test_df_year.rename(unit={"EJ/yr": "foo"})

    exp = df_ops_variable(operator.add, "Sum", unit="foo", meta=test_df_year.meta)

    if append:
        obs = df.copy()
        obs.add("Primary Energy", "Primary Energy|Coal", "Sum", append=True)
        exp = df.append(exp)
    else:
        obs = df.add("Primary Energy", "Primary Energy|Coal", "Sum")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_add_scenario(test_df_year, append):
    """Check that in-dataframe addition works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_sum")
    exp = IamDataFrame(
        pd.DataFrame([1 + 2, 6 + 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )

    if append:
        obs = test_df_year.copy()
        obs.add(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.add(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, expected_unit",
    (
        ("Primary Energy|Coal", df_ops_variable, "EJ/yr"),
        # pint changes the unit into its standard format
        (registry.Quantity(2, "EJ/yr"), df_ops_variable_number, "EJ / a"),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_subtract_variable(test_df_year, arg, df_func, expected_unit, append):
    """Check that in-dataframe subtraction works on the default `variable` axis"""

    exp = df_func(operator.sub, "Diff", unit=expected_unit, meta=test_df_year.meta)

    if append:
        obs = test_df_year.copy()
        obs.subtract("Primary Energy", arg, "Diff", append=True)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.subtract("Primary Energy", arg, "Diff")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_subtract_variable_ignore_units(test_df_year, arg, df_func, fillna, append):
    """Check that in-dataframe subtraction works witgh ignore_units"""

    # change one unit to make ignore-units strictly necessary
    test_df_year.rename(
        variable={"Primary Energy": "Primary Energy"},
        unit={"EJ/yr": "custom_unit"},
        inplace=True,
    )

    exp = df_func(operator.sub, "Diff", unit="foo", meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Diff")
    if append:
        obs = test_df_year.copy()
        obs.subtract(*args, ignore_units="foo", fillna=fillna, append=True)
        exp = test_df_year.append(exp)
    else:
        # check that incompatible units raise the expected error
        with pytest.raises(pint.UndefinedUnitError):
            test_df_year.add(*args, fillna=fillna)

        # using ignore_units works as expected
        obs = test_df_year.subtract(*args, ignore_units="foo", fillna=fillna)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_subtract_variable_non_si_unit(test_df_year, append):
    """Check that in-dataframe addition works with non-SI units"""
    df = test_df_year.rename(unit={"EJ/yr": "foo"})

    exp = df_ops_variable(operator.sub, "Diff", unit="foo", meta=test_df_year.meta)

    if append:
        obs = df.copy()
        obs.subtract("Primary Energy", "Primary Energy|Coal", "Diff", append=True)
        exp = df.append(exp)
    else:
        obs = df.subtract("Primary Energy", "Primary Energy|Coal", "Diff")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_subtract_scenario(test_df_year, append):
    """Check that in-dataframe subtraction works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_diff")
    exp = IamDataFrame(
        pd.DataFrame([1 - 2, 6 - 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )

    if append:
        obs = test_df_year.copy()
        obs.subtract(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.subtract(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, expected_unit",
    (
        ("Primary Energy|Coal", df_ops_variable, "EJ ** 2 / a ** 2"),
        (2, df_ops_variable_number, "EJ / a"),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_multiply_variable(test_df_year, arg, df_func, expected_unit, append):
    """Check that in-dataframe multiplication works on the default `variable` axis"""

    exp = df_func(operator.mul, "Prod", unit=expected_unit, meta=test_df_year.meta)

    if append:
        obs = test_df_year.copy()
        obs.multiply("Primary Energy", arg, "Prod", append=True)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.multiply("Primary Energy", arg, "Prod")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_multiply_variable_ignore_units(test_df_year, arg, df_func, fillna, append):
    """Check that in-dataframe addition works with ignore_units"""

    # change one unit to make ignore_units strictly necessary
    test_df_year.rename(
        variable={"Primary Energy": "Primary Energy"},
        unit={"EJ/yr": "custom_unit"},
        inplace=True,
    )

    exp = df_func(operator.mul, "Prod", unit="foo", meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Prod")
    if append:
        obs = test_df_year.copy()
        obs.multiply(*args, ignore_units="foo", fillna=fillna, append=True)
        exp = test_df_year.append(exp)
    else:
        # check that incompatible units raise the expected error
        with pytest.raises(pint.UndefinedUnitError):
            test_df_year.add(*args, fillna=fillna)

        obs = test_df_year.multiply(*args, ignore_units="foo", fillna=fillna)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_multiply_scenario(test_df_year, append):
    """Check that in-dataframe multiplication works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_product")
    exp = IamDataFrame(
        pd.DataFrame([1 * 2, 6 * 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit="EJ ** 2 / a ** 2",
    )

    if append:
        obs = test_df_year.copy()
        obs.multiply(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.multiply(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, expected_unit",
    (
        ("Primary Energy|Coal", df_ops_variable, ""),
        (registry.Quantity(2, "EJ/yr"), df_ops_variable_number, ""),
        (2, df_ops_variable_number, "EJ / a"),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_divide_variable(test_df_year, arg, df_func, expected_unit, append):
    """Check that in-dataframe division works on the default `variable` axis"""

    exp = df_func(operator.truediv, "Ratio", unit=expected_unit, meta=test_df_year.meta)

    if append:
        obs = test_df_year.copy()
        obs.divide("Primary Energy", arg, "Ratio", append=True)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.divide("Primary Energy", arg, "Ratio")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("value", (0, 0.0, registry.Quantity(0, "EJ/yr")))
def test_divide_by_zero_raises(test_df_year, value):
    """Check that division by zero (as single value) raises an error"""
    with pytest.raises(ZeroDivisionError):
        test_df_year.divide("Primary Energy", value, "Ratio")


@pytest.mark.parametrize("append", (False, True))
def test_divide_by_zero_drop_zero(test_df_year, append, caplog):
    """Check that division by zero in a series removes zeroes and writes to log"""

    exp = df_ops_variable(operator.truediv, "Ratio", unit="", meta=test_df_year.meta)
    exp.filter(year=2005, inplace=True)

    test_df_year._data.loc[
        "model_a", "scen_a", "World", "Primary Energy|Coal", "EJ/yr", 2010
    ] = 0

    if append:
        obs = test_df_year.copy()
        obs.divide("Primary Energy", "Primary Energy|Coal", "Ratio", append=True)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.divide("Primary Energy", "Primary Energy|Coal", "Ratio")

    assert_iamframe_equal(exp, obs)

    msg = (
        "Dropped 1 datapoints to avoid division by zero:\n"
        "     model scenario region  year\n"
        "0  model_a   scen_a  World  2010"
    )
    idx = caplog.messages.index(msg)
    assert caplog.records[idx].levelname == "WARNING"


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_divide_variable_ignore_units(test_df_year, arg, df_func, fillna, append):
    """Check that in-dataframe division works with ignore_units"""

    # change one unit to make ignore_units strictly necessary
    test_df_year.rename(
        variable={"Primary Energy": "Primary Energy"},
        unit={"EJ/yr": "custom_unit"},
        inplace=True,
    )

    exp = df_func(operator.truediv, "Ratio", unit="foo", meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Ratio")
    if append:
        obs = test_df_year.copy()
        obs.divide(*args, ignore_units="foo", fillna=fillna, append=True)
        exp = test_df_year.append(exp)
    else:
        # check that incompatible units raise the expected error
        with pytest.raises(pint.UndefinedUnitError):
            test_df_year.add(*args, fillna=fillna)

        # using ignore_units works as expected
        obs = test_df_year.divide(*args, ignore_units="foo", fillna=fillna)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_divide_variable_non_si_unit_unit(test_df_year, append):
    """Check that in-dataframe addition works with non-SI units"""
    df = test_df_year.rename(unit={"EJ/yr": "foo"})

    exp = df_ops_variable(operator.truediv, "Ratio", unit="", meta=test_df_year.meta)

    if append:
        obs = df.copy()
        obs.divide("Primary Energy", "Primary Energy|Coal", "Ratio", append=True)
        exp = df.append(exp)
    else:
        obs = df.divide("Primary Energy", "Primary Energy|Coal", "Ratio")

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_divide_scenario(test_df_year, append):
    """Check that in-dataframe division works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_ratio")
    exp = IamDataFrame(
        pd.DataFrame([1 / 2, 6 / 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit="",
    )

    if append:
        obs = test_df_year.copy()
        obs.divide(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.divide(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_apply_variable(test_df_year, append):
    """Check that in-dataframe `apply` works on the default `variable` axis"""

    def custom_func(a, b, c, d):
        return a * b + c * d

    v = "new variable"

    exp = IamDataFrame(
        pd.DataFrame(
            [custom_func(1, 2, 0.5, 3), custom_func(6, 2, 3, 3)], index=[2005, 2010]
        ).T,
        **DF_ARGS,
        scenario="scen_a",
        variable=v,
        unit="EJ / a",  # applying operations with pint reformats the unit
        meta=test_df_year.meta,
    )

    args = ["Primary Energy", 2]
    kwds = dict(d=3, c="Primary Energy|Coal")

    if append:
        obs = test_df_year.copy()
        obs.apply(custom_func, name="new variable", append=True, args=args, **kwds)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.apply(custom_func, name=v, args=args, **kwds)
        assert_iamframe_equal(exp, obs)


def test_ops_unknown_axis(test_df_year):
    """Using an unknown axis raises an error"""
    with pytest.raises(ValueError, match="Unknown axis: foo"):
        apply_ops(test_df_year, "_", "_", "foo")


def test_ops_unknown_method(test_df_year):
    """Using an unknown method raises an error"""
    with pytest.raises(ValueError, match="Unknown method: foo"):
        apply_ops(test_df_year, "_", "foo", "variable")


@pytest.mark.parametrize("periods, year", (({}, 2010), ({"periods": -1}, 2005)))
@pytest.mark.parametrize("append", (False, True))
def test_diff(test_df_year, periods, year, append):
    """Test `diff` method including non-default periods argument"""

    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "World", "foo", "EJ/yr", 5],
                ["model_a", "scen_a", "World", "bar", "EJ/yr", 2.5],
                ["model_a", "scen_b", "World", "foo", "EJ/yr", 5],
            ],
            columns=IAMC_IDX + [year],
        ),
        meta=test_df_year.meta,
    )
    # values are negative if computing diff in a negative direction
    if year == 2005:
        exp._data = -exp._data

    mapping = {"Primary Energy": "foo", "Primary Energy|Coal": "bar"}

    if append:
        obs = test_df_year.copy()
        obs.diff(mapping=mapping, append=True, **periods)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.diff(mapping=mapping, **periods)
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_diff_empty(test_df_year, append):
    """Assert that `diff` with only one time period returns empty"""

    df = test_df_year.filter(year=2005)
    mapping = {"Primary Energy": "foo", "Primary Energy|Coal": "bar"}

    if append:
        obs = df.copy()
        obs.diff(mapping=mapping, append=True)
        assert_iamframe_equal(df, obs)  # assert that no data was added
    else:
        obs = df.diff(mapping=mapping)
        assert obs.empty
