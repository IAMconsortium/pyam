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
        left.meta.dropna(axis="columns", how="all"),
        right.meta.dropna(axis="columns", how="all"),
        check_dtype=False,
        check_like=True,
    )
