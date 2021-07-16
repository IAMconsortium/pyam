# has to go first for environment setup reasons
import matplotlib

matplotlib.use("agg")

from pathlib import Path
import os
from requests.exceptions import ConnectionError
import pytest
import numpy as np
import pandas as pd

from datetime import datetime
from pyam import IamDataFrame, META_IDX, IAMC_IDX, iiasa


# verify whether IIASA database API can be reached, skip tests otherwise
try:
    iiasa.Connection()
    IIASA_UNAVAILABLE = False
except ConnectionError:  # pragma: no cover
    IIASA_UNAVAILABLE = True

TEST_API = "integration-test"
TEST_API_NAME = "IXSE_INTEGRATION_TEST"


here = Path(__file__).parent
IMAGE_BASELINE_DIR = here / "expected_figs"
TEST_DATA_DIR = here / "data"


TEST_YEARS = [2005, 2010]
TEST_DTS = [datetime(2005, 6, 17), datetime(2010, 7, 21)]
TEST_TIME_STR = ["2005-06-17", "2010-07-21"]
TEST_TIME_STR_HR = ["2005-06-17 00:00:00", "2010-07-21 12:00:00"]

DTS_MAPPING = {2005: TEST_DTS[0], 2010: TEST_DTS[1]}


TEST_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 1, 6.0],
        ["model_a", "scen_a", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 3],
        ["model_a", "scen_b", "World", "Primary Energy", "EJ/yr", 2, 7],
    ],
    columns=IAMC_IDX + TEST_YEARS,
)

META_COLS = ["number", "string"]
META_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", 1, "foo"],
        ["model_a", "scen_b", 2, np.nan],
    ],
    columns=META_IDX + META_COLS,
).set_index(META_IDX)


FULL_FEATURE_DF = pd.DataFrame(
    [
        ["World", "Primary Energy", "EJ/yr", 12, 15],
        ["reg_a", "Primary Energy", "EJ/yr", 8, 9],
        ["reg_b", "Primary Energy", "EJ/yr", 4, 6],
        ["World", "Primary Energy|Coal", "EJ/yr", 9, 10],
        ["reg_a", "Primary Energy|Coal", "EJ/yr", 6, 6],
        ["reg_b", "Primary Energy|Coal", "EJ/yr", 3, 4],
        ["World", "Primary Energy|Wind", "EJ/yr", 3, 5],
        ["reg_a", "Primary Energy|Wind", "EJ/yr", 2, 3],
        ["reg_b", "Primary Energy|Wind", "EJ/yr", 1, 2],
        ["World", "Emissions|CO2", "EJ/yr", 10, 14],
        ["World", "Emissions|CO2|Energy", "EJ/yr", 6, 8],
        ["World", "Emissions|CO2|AFOLU", "EJ/yr", 3, 4],
        ["World", "Emissions|CO2|Bunkers", "EJ/yr", 1, 2],
        ["reg_a", "Emissions|CO2", "EJ/yr", 6, 8],
        ["reg_a", "Emissions|CO2|Energy", "EJ/yr", 4, 5],
        ["reg_a", "Emissions|CO2|AFOLU", "EJ/yr", 2, 3],
        ["reg_b", "Emissions|CO2", "EJ/yr", 3, 4],
        ["reg_b", "Emissions|CO2|Energy", "EJ/yr", 2, 3],
        ["reg_b", "Emissions|CO2|AFOLU", "EJ/yr", 1, 1],
        ["World", "Price|Carbon", "USD/tCO2", 4, 27],
        ["reg_a", "Price|Carbon", "USD/tCO2", 1, 30],
        ["reg_b", "Price|Carbon", "USD/tCO2", 10, 21],
        ["World", "Population", "m", 3, 5],
        ["reg_a", "Population", "m", 2, 3],
        ["reg_b", "Population", "m", 1, 2],
    ],
    columns=["region", "variable", "unit"] + TEST_YEARS,
)


img = ["IMAGE", "a_scenario"]
msg = ["MESSAGE-GLOBIOM", "a_scenario"]

REG_DF = pd.DataFrame(
    [
        img + ["NAF", "Primary Energy", "EJ/yr", 1, 6],
        img + ["ME", "Primary Energy", "EJ/yr", 2, 7],
        img + ["World", "Primary Energy", "EJ/yr", 3, 13],
        msg + ["MEA", "Primary Energy", "EJ/yr", 1, 6],
        msg + ["AFR", "Primary Energy", "EJ/yr", 2, 7],
        msg + ["World", "Primary Energy", "EJ/yr", 3, 13],
    ],
    columns=IAMC_IDX + TEST_YEARS,
)


RECURSIVE_DF = pd.DataFrame(
    [
        ["Secondary Energy|Electricity", "EJ/yr", 5, 19.0],
        ["Secondary Energy|Electricity|Wind", "EJ/yr", 5, 17],
        ["Secondary Energy|Electricity|Wind|Offshore", "EJ/yr", 1, 5],
        ["Secondary Energy|Electricity|Wind|Onshore", "EJ/yr", 4, 12],
        ["Secondary Energy|Electricity|Solar", "EJ/yr", np.nan, 2],
    ],
    columns=["variable", "unit"] + TEST_YEARS,
)


TEST_STACKPLOT_DF = pd.DataFrame(
    [
        ["World", "Emissions|CO2|Energy|Oil", "Mt CO2/yr", 2, 3.2, 2.0, 1.8],
        ["World", "Emissions|CO2|Energy|Gas", "Mt CO2/yr", 1.3, 1.6, 1.0, 0.7],
        ["World", "Emissions|CO2|Energy|BECCS", "Mt CO2/yr", 0.0, 0.4, -0.4, 0.3],
        ["World", "Emissions|CO2|Cars", "Mt CO2/yr", 1.6, 3.8, 3.0, 2.5],
        ["World", "Emissions|CO2|Tar", "Mt CO2/yr", 0.3, 0.35, 0.35, 0.33],
        ["World", "Emissions|CO2|Agg", "Mt CO2/yr", 0.5, -0.1, -0.5, -0.7],
        ["World", "Emissions|CO2|LUC", "Mt CO2/yr", -0.3, -0.6, -1.2, -1.0],
    ],
    columns=["region", "variable", "unit", 2005, 2010, 2015, 2020],
)
# appease stickler
TEST_STACKPLOT_DF["model"] = "IMG"
TEST_STACKPLOT_DF["scenario"] = "a_scen"


# minimal IamDataFrame with four different time formats
@pytest.fixture(
    scope="function",
    params=[
        # standard IAMC format
        {},
        # testing several versions of datetime format
        dict([(i, j) for i, j in zip(TEST_YEARS, TEST_DTS)]),
        dict([(i, j) for i, j in zip(TEST_YEARS, TEST_TIME_STR)]),
        dict([(i, j) for i, j in zip(TEST_YEARS, TEST_TIME_STR_HR)]),
    ],
)
def test_df(request):
    df = IamDataFrame(data=TEST_DF.rename(request.param, axis="columns"))
    for i in META_COLS:
        df.set_meta(META_DF[i])
    yield df


# minimal IamDataFrame for specifically testing 'year'-column features
@pytest.fixture(scope="function")
def test_df_year():
    df = IamDataFrame(data=TEST_DF)
    for i in META_COLS:
        df.set_meta(META_DF[i])
    yield df


# minimal IamDataFrame for specifically testing 'time'-column features
@pytest.fixture(scope="function")
def test_df_time():
    df = IamDataFrame(
        data=TEST_DF.rename({2005: TEST_DTS[0], 2010: TEST_DTS[1]}, axis="columns")
    )
    for i in META_COLS:
        df.set_meta(META_DF[i])
    yield df


# minimal test data as pandas.DataFrame (only 'year' time format)
@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF.copy()


# minimal test data as pandas.DataFrame with index (only 'year' time format)
@pytest.fixture(scope="function")
def test_df_index():
    yield TEST_DF.set_index(IAMC_IDX)


# IamDataFrame with variable-and-region-structure for testing aggregation tools
@pytest.fixture(scope="function", params=[None, "datetime"])
def simple_df(request):
    _df = FULL_FEATURE_DF.copy()
    if request.param == "datetime":
        _df.rename(DTS_MAPPING, axis="columns", inplace=True)
    df = IamDataFrame(model="model_a", scenario="scen_a", data=_df)
    df.set_meta("foo", "string")
    yield df


# IamDataFrame with subannual time resolution
@pytest.fixture(scope="function")
def subannual_df():
    _df = FULL_FEATURE_DF.iloc[0:6].copy()

    def add_subannual(_data, name, value):
        _data["subannual"] = name
        _data[TEST_YEARS] = _data[TEST_YEARS] * value
        return _data

    # primary energy is a direct sum across sub-annual timeslices
    mapping = [("year", 1), ("winter", 0.7), ("summer", 0.3)]
    lst = [add_subannual(_df.copy(), name, value) for name, value in mapping]

    df = IamDataFrame(model="model_a", scenario="scen_a", data=pd.concat(lst))
    df.set_meta("foo", "string")
    yield df


@pytest.fixture(scope="function")
def reg_df():
    df = IamDataFrame(data=REG_DF)
    yield df


@pytest.fixture(scope="session")
def plot_df():
    df = IamDataFrame(data=os.path.join(TEST_DATA_DIR, "plot_data.csv"))
    yield df


# IamDataFrame with two scenarios and structure for recursive aggregation
@pytest.fixture(scope="function", params=["year", "datetime"])
def recursive_df(request):

    data = (
        RECURSIVE_DF
        if request.param == "year"
        else RECURSIVE_DF.rename(DTS_MAPPING, axis="columns")
    )

    df = IamDataFrame(data, model="model_a", scenario="scen_a", region="World")
    df2 = df.rename(scenario={"scen_a": "scen_b"})
    df2._data *= 2
    df.append(df2, inplace=True)

    yield df


@pytest.fixture(scope="session")
def plot_stackplot_df():
    df = IamDataFrame(TEST_STACKPLOT_DF)
    yield df


@pytest.fixture(scope="session")
def conn():
    if not IIASA_UNAVAILABLE:
        return iiasa.Connection(TEST_API)
