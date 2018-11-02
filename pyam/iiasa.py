import json
import logging
import requests

import numpy as np
import pandas as pd

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

from pyam.core import IamDataFrame
from pyam.logger import logger
from pyam.utils import LONG_IDX, isstr, pattern_match

# quiet this fool
logging.getLogger('requests').setLevel(logging.WARNING)

_URL_TEMPLATE = 'https://db1.ene.iiasa.ac.at/{}-api/rest/v2.1/'
_AUTH_URL = 'https://db1.ene.iiasa.ac.at/EneAuth/config/v1/anonym'

_CITATIONS = {
    'iamc15': 'D. Huppmann, E. Kriegler, V. Krey, K. Riahi, '
    'J. Rogelj, S.K. Rose, J. Weyant, et al., '
    'IAMC 1.5C Scenario Explorer and Data hosted by IIASA. '
    'IIASA & IAMC, 2018. '
    'doi: 10.22022/SR15/08-2018.15429, '
    'url: data.ene.iiasa.ac.at/iamc-1.5c-explorer'
}


def valid_connection_names():
    return list(_CITATIONS.keys())


class Connection(object):
    """A class to facilitate querying an IIASA scenario explorer database"""

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            A valid database name. For available options, see
            valid_connection_names().
        """
        valid = valid_connection_names()
        if name not in valid:
            raise ValueError('{} is not a valid name. Choose one of {}'.format(
                name, valid))

        logger().info(
            'You are connected to the {} {}. Please cite as:\n\n{}'
            .format(name, 'scenario explorer', _CITATIONS[name])
        )

        self.base_url = _URL_TEMPLATE.format(name)

    @lru_cache()
    def auth(self):
        """Anonymous user authentication token"""
        r = requests.get(_AUTH_URL)
        return r.json()

    @lru_cache()
    def metadata(self):
        """
        Metadata (e.g., models, scenarios, etc.) in the connected data
        source
        """
        url = self.base_url + 'runs?getOnlyDefaultRuns=false'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        return pd.read_json(r.content, orient='records')

    def models(self):
        """All models in the connected data source"""
        return pd.Series(self.metadata()['model'].unique(), name='model')

    def scenarios(self):
        """All scenarios in the connected data source"""
        return pd.Series(self.metadata()['scenario'].unique(), name='scenario')

    @lru_cache()
    def variables(self):
        """All variables in the connected data source"""
        url = self.base_url + 'ts'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        df = pd.read_json(r.content, orient='records')
        return pd.Series(df['variable'].unique(), name='variable')

    @lru_cache()
    def regions(self):
        """All regions in the connected data source"""
        url = self.base_url + 'nodes?hierarchy=%2A'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        df = pd.read_json(r.content, orient='records')
        return pd.Series(df['name'].unique(), name='region')

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
            if len(scenarios) > 0:
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

    def query(self, **kwargs):
        """
        Query the data source, subselecting data. Available keyword arguments
        include

        - model
        - scenario
        - region
        - variable

        Example
        -------

        ```
        Connection.query(model='MESSAGE', scenario='SSP2*',
                         variable=['Emissions|CO2', 'Primary Energy'])
        ```
        """
        headers = {
            'Authorization': 'Bearer {}'.format(self.auth()),
            'Content-Type': 'application/json',
        }
        data = json.dumps(self._query_post_data(**kwargs))
        url = self.base_url + 'runs/bulk/ts'
        r = requests.post(url, headers=headers, data=data)
        return pd.read_json(r.content, orient='records')


def read_iiasa(name, **kwargs):
    """
    Query an IIASA database. See Connection.query() for more documentation
    """
    conn = Connection(name)
    df = conn.query(**kwargs)
    return IamDataFrame(df[LONG_IDX + ['value']])


def read_iiasa_iamc15(**kwargs):
    """
    Query the IAMC 1.5C Scenario Explorer.
    See Connection.query() for more documentation
    """
    return read_iiasa('iamc15', **kwargs)
