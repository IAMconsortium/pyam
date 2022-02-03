import os
import pytest

from pyam import IamDataFrame, run_control

from .conftest import TEST_DATA_DIR, TEST_DF


def test_exec():
    rc = {
        "exec": [
            {"file": os.path.join(TEST_DATA_DIR, "exec.py"), "functions": ["do_exec"]},
        ]
    }

    run_control().update(rc)
    df = IamDataFrame(TEST_DF)

    exp = ["bar"] * len(TEST_DF["scenario"].unique())
    obs = df["foo"].values
    assert (exp == obs).all()


def test_no_file():
    rc = run_control()
    pytest.raises(IOError, rc.update, "no_such_file.yaml")
