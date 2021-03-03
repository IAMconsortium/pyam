from pyam import IamDataFrame

try:
    import pandas_datareader.wb as wb

    HAS_DATAREADER = True
except ImportError:  # pragma: no cover
    wb = None
    HAS_DATAREADER = False


def read_worldbank(model="World Bank", scenario="WDI", **kwargs):
    """Read data from the World Bank Data Catalogue and return as IamDataFrame

    This function is a simple wrapper for the class
    :class:`pandas_datareader.wb.WorldBankReader` and the function
    :func:`pandas_datareader.wb.download`. Import the module to retrieve/search
    the list of indicators (and their id's), countries, etc.

    .. code-block:: python

        from pandas_datareader import wb

    Parameters
    ----------
    model : str, optional
        The `model` name to be used for the returned timeseries data.
    scenario : str, optional
        The `scenario` name to be used for the returned timeseries data.
    kwargs
        passed to :func:`pandas_datareader.wb.download`

    Notes
    -----
    The function :func:`pandas_datareader.wb.download` takes an `indicator`
    argument, which can be a string or list of strings. If the `indicator`
    passed to :func:`read_worldbank` is a dictionary of a World Bank id mapped
    to a string, the variables in the returned IamDataFrame will be renamed.

    The function :func:`pandas_datareader.wb.download` does not return a unit,
    but it can be collected for some indicators using the function
    :func:`pandas_datareader.wb.get_indicators`.
    In the current implementation, unit is defined as `n/a` for all data;
    this can be enhanced later (if there is interest from users).

    Returns
    -------
    :class:`IamDataFrame`
    """
    if not HAS_DATAREADER:  # pragma: no cover
        raise ImportError("Required package `pandas-datareader` not found!")

    data = wb.download(**kwargs)
    df = IamDataFrame(
        data.reset_index(),
        model=model,
        scenario=scenario,
        value=data.columns,
        unit="n/a",
        region="country",
    )
    # TODO use wb.get_indicators to retrieve corrent units (where available)

    # if `indicator` is a mapping, use it for renaming
    if "indicator" in kwargs and isinstance(kwargs["indicator"], dict):
        df.rename(variable=kwargs["indicator"], inplace=True)

    return df
