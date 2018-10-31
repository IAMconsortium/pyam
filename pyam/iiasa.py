import requests
import json

import numpy as np
import pandas as pd

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

from pyam.core import IamDataFrame
from pyam.utils import LONG_IDX, isstr, pattern_match

_URL_TEMPLATE = 'https://db1.ene.iiasa.ac.at/{}-api/rest/v2.1/'
_AUTH_URL = 'https://db1.ene.iiasa.ac.at/EneAuth/config/v1/anonym'


class Connection(object):
    """A class to facilitate querying an IIASA scenario explorer database"""

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            A valid database name. Available options include:
                - sr15
        """
        valid = ['sr15']
        if name not in valid:
            raise ValueError('{} is not a valid name. Choose one of {}'.format(
                name, valid))

        self.base_url = _URL_TEMPLATE.format(name)

    @lru_cache()
    def auth(self):
        r = requests.get(_AUTH_URL)
        return r.json()

    @lru_cache()
    def variables(self):
        url = self.base_url + 'ts'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        df = pd.read_json(r.content, orient='records')
        return pd.Series(df['variable'].unique(), name='variable')

    @lru_cache()
    def regions(self):
        url = self.base_url + 'nodes?hierarchy=%2A'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        df = pd.read_json(r.content, orient='records')
        return pd.Series(df['name'].unique(), name='region')

    @lru_cache()
    def metadata(self):
        url = self.base_url + 'runs?getOnlyDefaultRuns=false'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        return pd.read_json(r.content, orient='records')

    def _query_post_data(self, **kwargs):
        def _get_kwarg(k):
            x = kwargs.pop(k, [])
            return [x] if isstr(x) else x

        m_pattern = _get_kwarg('model')
        s_pattern = _get_kwarg('scenario')
        v_pattern = _get_kwarg('variable')
        r_pattern = _get_kwarg('region')

        def _match(data, patterns):
            # this is empty, return empty list which means "everything"
            if not patterns:
                return []
            # otherwise match everything
            matches = np.array([False] * len(data))
            for p in patterns:
                matches |= pattern_match(data, p)
            return data[matches].unique()

        # get unique run ids
        meta = self.metadata()
        meta = meta[meta.is_default]
        models = _match(meta['model'], m_pattern)
        scenarios = _match(meta['scenario'], s_pattern)
        if len(models) == 0 and len(scenarios) == 0:
            runs = []
        else:
            where = np.array([True] * len(meta))
            if len(models) > 0:
                where &= meta.model.isin(models)
            if len(scenarios) >= 0:
                where &= meta.scenario.isin(scenarios)
            runs = meta.run_id[where].unique().tolist()

        # get unique other values
        variables = _match(self.variables(), v_pattern)
        regions = _match(self.regions(), r_pattern)

        data = {
            "filters": {
                "regions": list(regions),
                "variables": list(variables),
                "runs": list(runs),
                "years": [],
                "units": [],
                "times": []
            }
        }
        return data

    @lru_cache()
    def query(self, **kwargs):
        headers = {
            'Authorization': 'Bearer {}'.format(self.auth()),
            'Content-Type': 'application/json',
        }
        data = json.dumps(self._query_post_data(**kwargs))
        url = self.base_url + 'runs/bulk/ts'
        r = requests.post(url, headers=headers, data=data)
        return pd.read_json(r.content, orient='records')


def read_iiasa(name, **kwargs):
    conn = Connection(name)
    df = conn.query(**kwargs)
    return IamDataFrame(df[LONG_IDX + ['value']])


def read_iiasa_sr15(**kwargs):
    return read_iiasa('sr15', **kwargs)
