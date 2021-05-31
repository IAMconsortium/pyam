import pandas as pd
import pytest
import operator

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal
from pyam._ops import _op_data


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


def test_add_raises(test_df_year):
    """Calling an operation with args that don't return an IamDataFrame raises"""
    match = "Value returned by `add` cannot be cast to an IamDataFrame: 5"
    with pytest.raises(ValueError, match=match):
        test_df_year.add(2, 3, "foo")


@pytest.mark.parametrize(
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_add_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.add, "Sum", unit="EJ/yr", meta=test_df_year.meta)

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
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_subtract_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe subtraction works on the default `variable` axis"""

    exp = df_func(operator.sub, "Diff", unit="EJ/yr", meta=test_df_year.meta)

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
    "arg, df_func, fillna",
    (
        ("Primary Energy|Coal", df_ops_variable, None),
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_multiply_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.mul, "Prod", unit="EJ/yr", meta=test_df_year.meta)

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
        unit="EJ/yr",
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
        ("Primary Energy|Coal", df_ops_variable_default, {"c": 7, "b": 5}),
        ("Primary Energy|Coal", df_ops_variable_default, 5),
        (2, df_ops_variable_number, None),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_divide_variable(test_df_year, arg, df_func, fillna, append):
    """Verify that in-dataframe addition works on the default `variable` axis"""

    exp = df_func(operator.truediv, "Ratio", unit="EJ/yr", meta=test_df_year.meta)

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
        unit="EJ/yr",
    )

    if append:
        obs = test_df_year.copy()
        obs.divide(*v, axis="scenario", append=True)
        assert_iamframe_equal(test_df_year.append(exp), obs)
    else:
        obs = test_df_year.divide(*v, axis="scenario")
        assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_apply_variable(plot_stackplot_df, append):
    """Verify that in-dataframe apply works on the default `variable` axis"""

    def custom_func(a, b, c, d, e):
        return a / c + b / d + e

    args = ["Emissions|CO2|Tar", "Emissions|CO2|Cars", "Emissions|CO2|LUC"]
    kwds = {"d": "Emissions|CO2|Agg", "e": 5}
    exp = IamDataFrame(
        pd.DataFrame(
            [
                0.3 / (-0.3) + 1.6 / 0.5 + 5,
                0.35 / (-0.6) + 3.8 / (-0.1) + 5,
                0.35 / (-1.2) + 3.0 / (-0.5) + 5,
                0.33 / (-1.0) + 2.5 / (-0.7) + 5,
            ],
            index=[2005, 2010, 2015, 2020],
        ).T,
        model="IMG",
        scenario="a_scen",
        region="World",
        variable="new variable",
        unit="Mt CO2/yr",
    )

    if append:
        obs = plot_stackplot_df.copy()
        obs.apply(custom_func, name="new variable", append=True, args=args, **kwds)
        assert_iamframe_equal(plot_stackplot_df.append(exp), obs)
    else:
        obs = plot_stackplot_df.apply(
            custom_func, name="new variable", args=args, **kwds
        )
        assert_iamframe_equal(exp, obs)


def test_ops_unknown_axis(test_df_year):
    """Using an unknown axis raises an error"""
    with pytest.raises(ValueError, match="Unknown axis: foo"):
        _op_data(test_df_year, "_", "_", "foo")


def test_ops_unknown_method(test_df_year):
    """Using an unknown method raises an error"""
    with pytest.raises(ValueError, match="Unknown method: foo"):
        _op_data(test_df_year, "_", "foo", "variable")
