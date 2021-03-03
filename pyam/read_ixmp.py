try:
    import ixmp
except ImportError:
    pass


def read_ix(ix, **kwargs):
    """Read timeseries data from an ixmp object

    Parameters
    ----------
    ix: ixmp.TimeSeries or ixmp.Scenario
        this option requires the ixmp package as a dependency
    kwargs: arguments passed to ixmp.TimeSeries.timeseries()
    """
    if not isinstance(ix, ixmp.TimeSeries):
        error = "not recognized as valid ixmp class: {}".format(ix)
        raise ValueError(error)

    df = ix.timeseries(iamc=False, **kwargs)
    df["model"] = ix.model
    df["scenario"] = ix.scenario
    return df, "year", []
