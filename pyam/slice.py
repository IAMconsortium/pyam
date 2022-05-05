import pandas as pd


class IamSlice(pd.Series):
    """A slice object of the IamDataFrame timeseries data index"""
    @property
    def _constructor(self):
        return IamSlice

    _internal_names = pd.Series._internal_names + ["_iamcache"]
    _internal_names_set = set(_internal_names)

    def __init__(self, data=None, index=None, **kwargs):
        super().__init__(data, index, **kwargs)
        self._iamcache = dict()
        self.time_col = "year" if "year" in self.index.names else "time"
        self._time = None

    def __dir__(self):
        return self.dimensions + super().__dir__()

    def __getattr__(self, attr):
        ret = object.__getattribute__(self, "_iamcache").get(attr)
        if ret is not None:
            return ret.tolist() if attr != "time" else ret

        if attr in self.dimensions:
            ret = self._iamcache[attr] = self.index[self].unique(level=attr)
            return ret.tolist() if attr != "time" else ret

        return super().__getattr__(attr)

    def __len__(self):
        return self.sum()

    @property
    def dimensions(self):
        """Return the list of index names & data coordinates"""
        return self.index.names

    @property
    def time(self):
        """The time index, i.e., axis labels related to the time domain.

        Returns
        -------
        - A :class:`pandas.Int64Index` if the time_domain is 'year'
        - A :class:`pandas.DatetimeIndex` if the time domain is 'datetime'
        - A :class:`pandas.Index` if the time domain is 'mixed'
        """
        if self._time is None:
            self._time = pd.Index(
                self._data.index.unique(level=self.time_col).values, name="time"
            )

        return self._time

    def __repr__(self):
        return self.info() + "\n\n" + super().__repr__()

    def info(self, n=80):
        """Print a summary of the represented index dimensions

        Parameters
        ----------
        n : int
            The maximum line length
        """
        # concatenate list of index dimensions and levels
        info = f"{type(self)}\nIndex dimensions:\n"
        c1 = max([len(i) for i in self.dimensions]) + 1
        c2 = n - c1 - 5
        info += "\n".join(
            [
                f" * {i:{c1}}: {print_list(getattr(self, i), c2)}"
                for i in self.dimensions
            ]
        )

        return info