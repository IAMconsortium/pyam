from . import compare
import pandas.testing as pdt


def assert_iamframe_equal(left, right, **kwargs):
    """Check that left and right IamDataFrame instances are equal.

    Parameters
    ----------
    left, right : :class:`IamDataFrame`
        Two IamDataFrame instances to be compared.
    kwargs
        Passed to :meth:`IamDataFrame.compare`, comparing the `data` objects.

    Raises
    ------
    AssertionError if *left* and *right* are different.

    Notes
    -----
    Columns of the *meta* attribute where all values are *nan* are ignored.
    """
    diff = compare(left, right, **kwargs)
    if not diff.empty:
        msg = "IamDataFrame.data are different: \n {}"
        raise AssertionError(msg.format(diff.head()))

    pdt.assert_frame_equal(
        _drop_nan_col(left.meta),
        _drop_nan_col(right.meta),
        check_dtype=False,
        check_like=True,
    )


def _drop_nan_col(df):
    return df.dropna(axis="columns", how="all")
