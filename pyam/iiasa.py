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
from pyam.utils import META_IDX, islistable, isstr, pattern_match

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
    def scenario_list(self, default=True):
        """
        Metadata regarding the list of scenarios (e.g., models, scenarios,
        run identifier, etc.) in the connected data source.

        Parameter
        ---------
        default : bool, optional, default True
            Return *only* the default version of each Scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If :obj:`False`, return all versions.
        """
        default = 'true' if default else 'false'
        add_url = 'runs?getOnlyDefaultRuns={}'
        url = self.base_url + add_url.format(default)
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        return pd.read_json(r.content, orient='records')

    @lru_cache()
    def available_metadata(self):
        """
        List all scenario metadata indicators available in the connected
        data source
        """
        url = self.base_url + 'metadata/types'
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        return pd.read_json(r.content, orient='records')['name']

    @lru_cache()
    def metadata(self, default=True):
        """
        Metadata of scenarios in the connected data source

        Parameter
        ---------
        default : bool, optional, default True
            Return *only* the default version of each Scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If :obj:`False`, return all versions.
        """
        # at present this reads in all data for all scenarios, it could be sped
        # up in the future to try to query a subset
        default = 'true' if default else 'false'
        add_url = 'runs?getOnlyDefaultRuns={}&includeMetadata=true'
        url = self.base_url + add_url.format(default)
        headers = {'Authorization': 'Bearer {}'.format(self.auth())}
        r = requests.get(url, headers=headers)
        df = pd.read_json(r.content, orient='records')

        def extract(row):
            return (
                pd.concat([row[['model', 'scenario']],
                           pd.Series(row.metadata)])
                .to_frame()
                .T
                .set_index(['model', 'scenario'])
            )

        return pd.concat([extract(row) for idx, row in df.iterrows()],
                         sort=False).reset_index()

    def models(self):
        """All models in the connected data source"""
        return pd.Series(self.scenario_list()['model'].unique(),
                         name='model')

    def scenarios(self):
        """All scenarios in the connected data source"""
        return pd.Series(self.scenario_list()['scenario'].unique(),
                         name='scenario')

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
        meta = self.scenario_list()
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
        # refactor returned json object to be castable to an IamDataFrame
        df = (
            pd.read_json(r.content, orient='records')
            .drop(columns='runId')
            .rename(columns={'time': 'subannual'})
        )
        # check if returned dataframe has subannual disaggregation, drop if not
        if pd.Series([i in [-1, 'year'] for i in df.subannual]).all():
            df.drop(columns='subannual', inplace=True)
        # check if there are multiple version for any model/scenario
        lst = (
            df[META_IDX + ['version']].drop_duplicates()
            .groupby(META_IDX).count().version
        )
        if max(lst) > 1:
            raise ValueError('multiple versions for {}'.format(
                lst[lst > 1].index.to_list()))
        df.drop(columns='version', inplace=True)

        return df


def read_iiasa(name, meta=False, **kwargs):
    """
    Query an IIASA database. See Connection.query() for more documentation

    Parameters
    ----------
    name : str
        A valid IIASA database name, see pyam.iiasa.valid_connection_names()
    meta : bool or list of strings
        If not False, also include metadata indicators (or subset if provided).
    kwargs :
        Arguments for pyam.iiasa.Connection.query()
    """
    conn = Connection(name)
    # data
    df = conn.query(**kwargs)
    df = IamDataFrame(df)
    # metadata
    if meta:
        mdf = conn.metadata()
        # only data for models/scenarios in df
        mdf = mdf[mdf.model.isin(df['model'].unique()) &
                  mdf.scenario.isin(df['scenario'].unique())]
        # get subset of data if meta is a list
        if islistable(meta):
            mdf = mdf[['model', 'scenario'] + meta]
        mdf = mdf.set_index(['model', 'scenario'])
        # we have to loop here because `set_meta()` can only take series
        for col in mdf:
            df.set_meta(mdf[col])
    return df


def read_iiasa_iamc15(**kwargs):
    """
    Query the IAMC 1.5C Scenario Explorer.
    See Connection.query() for more documentation
    """
    return read_iiasa('iamc15', **kwargs)
