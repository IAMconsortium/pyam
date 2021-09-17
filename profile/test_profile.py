from pyam import IamDataFrame, IAMC_IDX
import pandas as pd


TEST_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 1, 6.0],
        ["model_a", "scen_a", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 3],
        ["model_a", "scen_b", "World", "Primary Energy", "EJ/yr", 2, 7],
    ],
    columns=IAMC_IDX + [2005, 2010],
)


def test_init():
    IamDataFrame(TEST_DF)
