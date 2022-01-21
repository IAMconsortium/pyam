from . import compare
import pandas.testing as pdt


def assert_iamframe_equal(a, b, **assert_kwargs):
    diff = compare(a, b, **assert_kwargs)
    if not diff.empty:
        msg = "IamDataFrame.data are different: \n {}"
        raise AssertionError(msg.format(diff.head()))

    pdt.assert_frame_equal(
        _drop_nan_col(a.meta), _drop_nan_col(b.meta), check_dtype=False, check_like=True
    )


def _drop_nan_col(df):
    return df.dropna(axis="columns", how="all")
