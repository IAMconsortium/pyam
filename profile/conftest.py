from pathlib import Path
import pandas as pd
import pytest

from pyam import IAMC_IDX

DATA_PATH = Path("data")
TEST_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 1, 6.0],
        ["model_a", "scen_a", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 3],
        ["model_a", "scen_b", "World", "Primary Energy", "EJ/yr", 2, 7],
    ],
    columns=IAMC_IDX + [2005, 2010],
)

TEST_FRAMES = [TEST_DF] + [
    pd.read_excel(f, sheet_name="data") for f in DATA_PATH.glob("*.xlsx")
]


@pytest.fixture(scope="function", params=TEST_FRAMES)
def data(request):
    yield request.param
