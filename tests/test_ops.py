import pandas as pd
import pytest
import operator

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal
from pyam._ops import _op_data


DF_INDEX = ["scenario", 2005, 2010]
# dictionary with common IamDataFrame args for all tests operating on variable
DF_ARGS = dict(model="model_a", region="World")


UNIT_EJ = "exajoule / year"
UNIT_EJ_SQ = "exajoule ** 2 / year ** 2"


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


def test_add_raises(test_df_year):
    """Calling an operation with args that don't return an IamDataFrame raises"""
    match = "Value returned by `add` cannot be cast to an IamDataFrame: 5"
    with pytest.raises(ValueError, match=match):
        test_df_year.add(2, 3, "foo")


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        # ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        # ("Primary Energy|Coal", df_ops_variable_default, 5),
        # (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_add_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.add, "Sum", unit=UNIT_EJ, meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Sum")
    if append:
        obs = test_df_year.copy()
        obs.add(*args, fillna=fillna, append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        assert_iamframe_equal(exp, test_df_year.add(*args, fillna=fillna))


@pytest.mark.parametrize("append", (False, True))
def test_add_scenario(test_df_year, append):
    """Verify that in-dataframe addition works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_sum")
    exp = IamDataFrame(
        pd.DataFrame([1 + 2, 6 + 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit=UNIT_EJ,
    )

    if append:
        obs = test_df_year.copy()
        obs.add(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.add(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        # ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        # ("Primary Energy|Coal", df_ops_variable_default, 5),
        # (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_subtract_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe subtraction works on the default `variable` axis"""

    exp = df_func(operator.sub, "Diff", unit=UNIT_EJ, meta=test_df_year.meta)

    if append:
        obs = test_df_year.copy()
        obs.subtract("Primary Energy", arg, "Diff", fillna=fillna, append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.subtract("Primary Energy", arg, "Diff", fillna=fillna)
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_subtract_scenario(test_df_year, append):
    """Verify that in-dataframe subtraction works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_diff")
    exp = IamDataFrame(
        pd.DataFrame([1 - 2, 6 - 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit=UNIT_EJ,
    )

    if append:
        obs = test_df_year.copy()
        obs.subtract(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.subtract(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna, unit",
    (
        ("Primary Energy|Coal", df_ops_variable, None, UNIT_EJ_SQ),
        # ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}, UNIT_EJ),
        # ("Primary Energy|Coal", df_ops_variable_default, 5, UNIT_EJ),
        # (2, df_ops_variable_number, None, UNIT_EJ),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_multiply_variable(test_df_year, arg, df_func, fillna, append, unit):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.mul, "Prod", unit=unit, meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Prod")
    if append:
        obs = test_df_year.copy()
        obs.multiply(*args, fillna=fillna, append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        assert_iamframe_equal(exp, test_df_year.multiply(*args, fillna=fillna))


@pytest.mark.parametrize("append", (False, True))
def test_multiply_scenario(test_df_year, append):
    """Verify that in-dataframe addition works on a custom axis (`scenario`)"""

    v = ("scen_a", "scen_b", "scen_product")
    exp = IamDataFrame(
        pd.DataFrame([1 * 2, 6 * 7], index=[2005, 2010]).T,
        model="model_a",
        scenario=v[2],
        region="World",
        variable="Primary Energy",
        unit=UNIT_EJ_SQ,
    )

    if append:
        obs = test_df_year.copy()
        obs.multiply(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.multiply(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        # ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        # ("Primary Energy|Coal", df_ops_variable_default, 5),
        # (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_divide_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.truediv, "Ratio", unit="", meta=test_df_year.meta)

    args = ("Primary Energy", arg, "Ratio")
    if append:
        obs = test_df_year.copy()
        obs.divide(*args, fillna=fillna, append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        assert_iamframe_equal(exp, test_df_year.divide(*args, fillna=fillna))


@pytest.mark.parametrize("append", (False, True))
def test_divide_scenario(test_df_year, append):
    """Verify that in-dataframe addition works on a custom axis (`scenario`)"""

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
    """Verify that in-dataframe apply works on the default `variable` axis"""

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
        unit=UNIT_EJ,
        meta=test_df_year.meta
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
        _op_data(test_df_year, "_", "_", "foo")


def test_ops_unknown_method(test_df_year):
    """Using an unknown method raises an error"""
    with pytest.raises(ValueError, match="Unknown method: foo"):
        _op_data(test_df_year, "_", "foo", "variable")
