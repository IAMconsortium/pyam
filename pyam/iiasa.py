import collections
import json
import logging
import os
import requests
import warnings
import yaml

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

_BASE_URL = 'https://db1.ene.iiasa.ac.at/EneAuth/config/v1'
_CITE_MSG = """
You are connected to the {} scenario explorer hosted by IIASA.
 If you use this data in any published format, please cite the
 data as provided in the explorer guidelines: {}.
""".replace('\n', '')


def _check_response(r, msg='Trouble with request', error=RuntimeError):
    if not r.ok:
        raise error('{}: {}'.format(msg, str(r.text)))


def _get_token(creds):
    if creds is None:  # get anonymous auth
        url = '/'.join([_BASE_URL, 'anonym'])
        r = requests.get(url)
        _check_response(r, 'Could not get anonymous token')
        return r.json()

    # otherwise read creds and try to login
    filecreds = False
    try:
        if isinstance(creds, collections.Mapping):
            user, pw = creds['username'], creds['password']
        elif os.path.exists(str(creds)):
            with open(str(creds), 'r') as stream:
                creds = yaml.safe_load(stream)
            user, pw = creds['username'], creds['password']
            filecreds = True
        else:
            user, pw = creds
    except Exception as e:
        msg = 'Could not read credentials: {}\n{}'.format(
            creds, str(e))
        raise type(e)(msg)
    if not filecreds:
        warnings.warn('You provided credentials in plain text. DO NOT save ' +
                      'these in a repository or otherwise post them online')

    headers = {'Accept': 'application/json',
               'Content-Type': 'application/json'}
    data = {'username': user, 'password': pw}
    url = '/'.join([_BASE_URL, 'login'])
    r = requests.post(url, headers=headers, data=json.dumps(data))
    _check_response(r, 'Login failed for user: {}'.format(user))
    return r.json()


class Connection(object):
    """A class to facilitate querying an IIASA scenario explorer database"""

    def __init__(self, name=None, creds=None):
        """
        Parameters
        ----------
        name : str, optional
            A valid database name. For available options, see
            valid_connection_names().
        creds : str, list-like, or dict, optional
            Either:
              - a yaml filename/path with entries of  'username' and 'password'
                (preferred)
              - an ordered container (tuple, list, etc.) with the same values
              - a dictionary with the same keys
        """
        self._token = _get_token(creds)

        # connect if provided a name
        self._connected = None
        if name:
            self.connect(name)

    @property
    @lru_cache()
    def valid_connections(self):
        url = '/'.join([_BASE_URL, 'applications'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, 'Could not get valid connection list')
        valid = [x['name'] for x in r.json()]
        return valid

    def connect(self, name):
        # TODO: deprecate in next release
        if name == 'iamc15':
            warnings.warn(
                'The name `iamc15` is deprecated and will be removed in the ' +
                'next release. Please use `IXSE_SR15`.'
            )
            name = 'IXSE_SR15'

        valid = self.valid_connections
        if len(valid) == 0:
            raise RuntimeError(
                'No valid connections found for the provided credentials.'
            )

        if name not in valid:
            msg = """
            {} not recognized as a valid connection name.
            Choose from one of the supported connections for your user: {}.
            """
            raise ValueError(msg.format(name, valid))

        url = '/'.join([_BASE_URL, 'applications', name, 'config'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, 'Could not get application information')
        response = r.json()
        idxs = {x['path']: i for i, x in enumerate(response)}

        self._base_url = response[idxs['baseUrl']]['value']
        # TODO: request the full citation to be added to this metadata intead
        # of linking to the about page
        about = '/'.join([response[idxs['uiUrl']]['value'], '#', 'about'])
        logger().info(_CITE_MSG.format(name, about))

        self._connected = name

    @property
    def current_connection(self):
        return self._connected

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
        url = '/'.join([self._base_url, add_url.format(default)])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, 'Could not get scenario list')
        return pd.read_json(r.content, orient='records')

    @lru_cache()
    def available_metadata(self):
        """
        List all scenario metadata indicators available in the connected
        data source
        """
        url = '/'.join([self._base_url, 'metadata/types'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
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
        url = '/'.join([self._base_url, add_url.format(default)])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
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
        url = '/'.join([self._base_url, 'ts'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
        df = pd.read_json(r.content, orient='records')
        return pd.Series(df['variable'].unique(), name='variable')

    @lru_cache()
    def regions(self):
        """All regions in the connected data source"""
        url = '/'.join([self._base_url, 'nodes?hierarchy=%2A'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
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
            'Authorization': 'Bearer {}'.format(self._token),
            'Content-Type': 'application/json',
        }
        data = json.dumps(self._query_post_data(**kwargs))
        url = '/'.join([self._base_url, 'runs/bulk/ts'])
        r = requests.post(url, headers=headers, data=data)
        _check_response(r)
        # refactor returned json object to be castable to an IamDataFrame
        df = pd.read_json(r.content, orient='records')
        columns = ['model', 'scenario', 'variable', 'unit',
                   'region', 'year', 'value', 'time', 'meta',
                   'runId', 'version']
        df = pd.DataFrame(data=df, columns=columns)
        # replace missing meta (for backward compatibility)
        df.fillna({'meta': 0}, inplace=True)
        df.drop(columns='runId', inplace=True)
        df.rename(columns={'time': 'subannual'}, inplace=True)
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


def read_iiasa(name, meta=False, creds=None, **kwargs):
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
    conn = Connection(name, creds)
    # data
    df = conn.query(creds=creds, **kwargs)
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
