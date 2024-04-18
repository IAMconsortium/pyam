import copy
import os
from collections.abc import Mapping

import yaml

from pyam.str import is_str

# user-defined defaults for various plot settings
_RUN_CONTROL = None


# path to regional mapping files
def _REG_MAP_PATH(x) -> str:
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "region_mappings", x
    )


# defaults for run control
_RC_DEFAULTS = {
    "color": {},
    "marker": {},
    "linestyle": {},
    "order": {},
}


def reset_rc_defaults():
    """Reset run control object to original defaults"""
    global _RUN_CONTROL
    _RUN_CONTROL = RunControl()


def run_control():
    """Global run control for user-defined plotting style defaults"""
    global _RUN_CONTROL
    if _RUN_CONTROL is None:
        _RUN_CONTROL = RunControl()
    return _RUN_CONTROL


def _recursive_update(d, u):
    """recursively update a dictionary d with a dictionary u"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = _recursive_update(d.get(k, {}), v)
            d[k] = r
        elif isinstance(v, list):  # values for `order` are lists
            if k in d:
                d[k] += [i for i in v if i not in d[k]]
            else:
                d[k] = v
        else:
            d[k] = u[k]
    return d


class RunControl(Mapping):
    """A thin wrapper around a Python dictionary for plotting style defaults

    Input can be provided as dictionaries or YAML files.
    """

    def __init__(self, rc=None, defaults=None):
        """
        Parameters
        ----------
        rc : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a
            dictionary describing run control configuration
        defaults : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a
            dictionary describing **default** run control configuration
        """
        rc = rc or {}
        defaults = defaults or copy.deepcopy(_RC_DEFAULTS)

        rc = self._load_yaml(rc)
        defaults = self._load_yaml(defaults)
        self.store = _recursive_update(defaults, rc)

    def update(self, rc):
        """Add additional run control parameters

        Parameters
        ----------
        rc : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a
            dictionary describing run control configuration
        """
        rc = self._load_yaml(rc)
        self.store = _recursive_update(self.store, rc)

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()

    def _get_path(self, key, fyaml, fname):
        if os.path.exists(fname):
            return fname

        _fname = os.path.join(os.path.dirname(fyaml), fname)
        if not os.path.exists(_fname):
            msg = (
                "YAML key '{}' in {}: {} is not a valid relative " + "or absolute path"
            )
            raise OSError(msg.format(key, fyaml, fname))
        return _fname

    def _load_yaml(self, obj):
        if hasattr(obj, "read"):  # it's a file
            obj = obj.read()
        if is_str(obj) and not os.path.exists(obj):
            raise OSError(f"File {obj} does not exist")
        if is_str(obj) and os.path.exists(obj):
            fname = obj
            with open(fname) as f:
                obj = f.read()
        if not isinstance(obj, dict):
            obj = yaml.load(obj, Loader=yaml.FullLoader)
        return obj

    def recursive_update(self, k, d):
        """Recursively update a top-level option in the run control

        Parameters
        ----------
        k : string
            the top-level key
        d : dictionary or similar
            the dictionary to use for updating
        """
        u = self.__getitem__(k)
        self.store[k] = _recursive_update(u, d)
