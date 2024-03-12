import pandas as pd

from pyam.utils import print_list


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

    def __dir__(self):
        return self.dimensions + super().__dir__()

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            cache = object.__getattribute__(self, "_iamcache")
            ret = cache.get(attr)
            if ret is not None:
                return ret.tolist()

            if attr in self.dimensions:
                ret = cache[attr] = self.index[self].unique(level=attr)
                return ret.tolist()

            raise

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
        - A :class:`pandas.Index` (dtype 'int64') if the :attr:`time_domain` is 'year'
        - A :class:`pandas.DatetimeIndex` if the time-domain is 'datetime'
        - A :class:`pandas.Index` if the time-domain is 'mixed'
        """
        ret = self._iamcache.get("time")
        if ret is None:
            ret = self._iamcache["time"] = (
                self.index[self].unique(level=self.time_col).rename("time")
            )
        return ret

    @property
    def time_col(self):
        return "year" if "year" in self.dimensions else "time"

    def __repr__(self):
        return self.info()

    def info(self, n=80):
        """Print a summary of the represented index dimensions and data coordinates

        Parameters
        ----------
        n : int
            The maximum line length
        """
        # concatenate list of index dimensions and levels
        info = f"{type(self)}\nIndex dimensions and data coordinates:\n"
        c1 = max([len(i) for i in self.dimensions]) + 1
        c2 = n - c1 - 5
        info += "\n".join(
            [
                f"   {i:{c1}}: {print_list(getattr(self, i), c2)}"
                for i in self.dimensions
            ]
        )

        return info
