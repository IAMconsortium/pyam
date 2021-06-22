import pandas as pd
from pyam import IamDataFrame, read_unfccc
from pyam.testing import assert_iamframe_equal


UNFCCC_DF = pd.DataFrame(
    [[1990, 1609.25345], [1991, 1434.21149], [1992, 1398.38269]],
    columns=["year", "value"],
)

INDEX_ARGS = dict(model="UNFCCC", scenario="Data Inventory")


def test_unfccc_tier1():
    # test that UNFCCC API returns expected data and units
    exp = IamDataFrame(
        UNFCCC_DF,
        **INDEX_ARGS,
        region="DEU",
        variable="Emissions|CH4|Agriculture",
        unit="kt CH4",
    )

    obs = read_unfccc(party_code="DEU", gases=["CH4"], tier=1)

    # assert that the data is similar
    horizon = [1990, 1991, 1992]
    assert_iamframe_equal(obs.filter(year=horizon, variable="*Agri*"), exp)

    # assert that variables are similar
    types = [
        "Agriculture",
        "Energy",
        "Industrial Processes and Product Use",
        "Land Use, Land-Use Change and Forestry",
        "Waste",
    ]
    print([f"Emissions|CH4|{i}" for i in types])
    print(obs.variable)
    assert obs.variable == [f"Emissions|CH4|{i}" for i in types]

    # assert that the unit is merged as expected
    assert obs.unit == ["kt CH4"]
