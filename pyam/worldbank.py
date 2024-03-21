import pandas as pd

from pyam import IamDataFrame


def read_worldbank(model="World Bank", scenario="WDI", **kwargs) -> IamDataFrame:
    """Read data from the World Bank Data Catalogue and return as IamDataFrame

    This function is a simple wrapper for the function
    :func:`wbdata.get_dataframe`. Import the module to retrieve/search
    the list of indicators (and their id's), countries, etc.

    .. code-block:: python

        import wbdata as wb

    Parameters
    ----------
    model : str, optional
        The `model` name to be used for the returned timeseries data.
    scenario : str, optional
        The `scenario` name to be used for the returned timeseries data.
    **kwargs
        passed to :func:`wbdata.get_dataframe`

    Notes
    -----
    The function :func:`wbdata.get_dataframe` takes an `indicators`
    argument, which is a dictionary where the keys are desired indicators and the values
    are the desired column names. If the `indicators` passed to :func:`read_worldbank`
    is a single indicator code string, we should instead use :func:`wbdata.get_series`.

    The function :func:`wbdata.get_dataframe` does not return a unit,
    but it can be collected for some indicators using the function
    :func:`wbdata.get_indicators`.
    In the current implementation, unit is defined as `n/a` for all data;
    this can be enhanced later (if there is interest from users).

    Returns
    -------
    :class:`IamDataFrame`
    """
    # import packages for functions with low-frequency usage only when needed
    # also, there seems to be an issue with wbdata on Mac OS
    # see https://github.com/OliverSherouse/wbdata/issues/74
    import wbdata  # noqa: F401

    data: pd.DataFrame = wbdata.get_dataframe(**kwargs)
    value = data.columns
    data.reset_index(inplace=True)
    data.rename(columns={"date": "year"}, inplace=True)
    df = IamDataFrame(
        data,
        model=model,
        scenario=scenario,
        value=value,
        unit="n/a",
        region="country",
    )
    # TODO use wb.get_indicators to retrieve correct units (where available)

    # if `indicators` is a mapping, use it for renaming
    if "indicators" in kwargs and isinstance(kwargs["indicators"], dict):
        df.rename(variable=kwargs["indicators"], inplace=True)

    return df
