from . import compare
import pandas.testing as pdt


def assert_iamframe_equal(a, b, **assert_kwargs):
    diff = compare(a, b, **assert_kwargs)
    if not diff.empty:
        msg = "IamDataFrame.data are different: \n {}"
        raise AssertionError(msg.format(diff.head()))

    pdt.assert_frame_equal(a.meta, b.meta, check_dtype=False, check_like=True)
