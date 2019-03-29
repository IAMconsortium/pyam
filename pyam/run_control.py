import copy
import collections
import os
import yaml

from pyam.utils import isstr

# user-defined defaults for various plot settings
_RUN_CONTROL = None

# path to regional mapping files
_REG_MAP_PATH = lambda x: os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'region_mappings', x)

# defaults for run control
_RC_DEFAULTS = {
    'color': {},
    'marker': {},
    'linestyle': {},
    'region_mapping': {
        'default': _REG_MAP_PATH('default_mapping.csv'),
    }
}


def reset_rc_defaults():
    """Reset run control object to original defaults"""
    global _RUN_CONTROL
    _RUN_CONTROL = RunControl()


def run_control():
    """Global run control for determining user-defined defaults for plotting style"""
    global _RUN_CONTROL
    if _RUN_CONTROL is None:
        _RUN_CONTROL = RunControl()
    return _RUN_CONTROL


def _recursive_update(d, u):
    """recursively update a dictionary d with a dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = _recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class RunControl(collections.Mapping):
    """A thin wrapper around a Python Dictionary to support configuration of
    harmonization execution. Input can be provided as dictionaries or YAML
    files.
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
            msg = "YAML key '{}' in {}: {} is not a valid relative " + \
                "or absolute path"
            raise IOError(msg.format(key, fyaml, fname))
        return _fname

    def _load_yaml(self, obj):
        check_rel_paths = False
        if hasattr(obj, 'read'):  # it's a file
            obj = obj.read()
        if isstr(obj) and not os.path.exists(obj):
            raise IOError('File {} does not exist'.format(obj))
        if isstr(obj) and os.path.exists(obj):
            check_rel_paths = True
            fname = obj
            with open(fname) as f:
                obj = f.read()
        if not isinstance(obj, dict):
            obj = yaml.load(obj)
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
