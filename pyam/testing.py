import pandas.testing as pdt
from pytest import raises

from pyam import IamDataFrame

from . import compare


def assert_iamframe_equal(
    left: IamDataFrame,
    right: IamDataFrame,
    check_meta: bool = True,
    **kwargs,
) -> None:
    """Check that left and right IamDataFrame instances are equal.

    Parameters
    ----------
    left, right : :class:`IamDataFrame`
        Two IamDataFrame instances to be compared.
    check_meta: bool
        If set to true, only the timeseries values will be compared, default False
    **kwargs
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

    if check_meta:
        pdt.assert_frame_equal(
            left.meta.dropna(axis="columns", how="all"),
            right.meta.dropna(axis="columns", how="all"),
            check_column_type=False,
            check_dtype=False,
            check_like=True,
        )

    pdt.assert_series_equal(
        left.exclude,
        right.exclude,
    )


def assert_iamframe_not_equal(
    left: IamDataFrame,
    right: IamDataFrame,
    check_meta: bool = True,
    **kwargs,
) -> None:
    with raises(AssertionError):
        assert_iamframe_equal(left, right, check_meta=check_meta, **kwargs)
