from pyam import IamDataFrame


def test_init(data):
    IamDataFrame(data)


def test_filter(df):
    df.filter(year=2010)
