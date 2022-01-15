from pyam.utils import remove_from_list
from pyam.timeseries import compute_learning_rate


class IamComputeAccessor:
    """Perform computations on the timeseries data of an IamDataFrame

    An :class:`IamDataFrame` has a module for computation of (advanced) indicators
    from the timeseries data.

    The methods in this module can be accessed via

    .. code-block:: python

        IamDataFrame.compute.<method>(*args, **kwargs)
    """

    def __init__(self, df):
        self._df = df

    def learning_rate(self, name, performance, experience, append=False):
        """Compute the implicit learning rate from timeseries data

        Refer to :func:`pyam.timeseries.compute_learning_rate` for more information.

        Parameters
        ----------
        name : str
            Variable name of the computed timeseries data.
        performance : str
            Variable of the "performance" timeseries (e.g., specific investment costs).
        experience : str
            Variable of the "experience" timeseries (e.g., installed capacity).
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.
        """
        _data = self._df._data[
            self._df._apply_filters(variable=[performance, experience])
        ].groupby(remove_from_list(self._df.dimensions, ["variable", "year", "unit"]))
        _value = _data.apply(compute_learning_rate, performance, experience)

        args = dict(variable=name, unit="")
        if append:
            self._df.append(_value, **args, inplace=True)
        else:
            return self._df.__class__(_value, meta=self._df.meta, **args)