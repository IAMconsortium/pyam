import collections
import json
import logging
import os
import requests
import yaml
from functools import lru_cache

import numpy as np
import pandas as pd

from pyam.core import IamDataFrame
from pyam.utils import META_IDX, islistable, isstr, pattern_match

logger = logging.getLogger(__name__)
# quiet this fool
logging.getLogger('requests').setLevel(logging.WARNING)

_BASE_URL = 'https://db1.ene.iiasa.ac.at/EneAuth/config/v1'
_CITE_MSG = """
You are connected to the {} scenario explorer hosted by IIASA.
 If you use this data in any published format, please cite the
 data as provided in the explorer guidelines: {}
""".replace('\n', '')


def _check_response(r, msg='Trouble with request', error=RuntimeError):
    if not r.ok:
        raise error('{}: {}'.format(msg, str(r.text)))


def _get_token(creds, base_url):
    if creds is None:  # get anonymous auth
        url = '/'.join([base_url, 'anonym'])
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
        logger.warning('You provided credentials in plain text. DO NOT save '
                       'these in a repository or otherwise post them online')

    headers = {'Accept': 'application/json',
               'Content-Type': 'application/json'}
    data = {'username': user, 'password': pw}
    url = '/'.join([base_url, 'login'])
    r = requests.post(url, headers=headers, data=json.dumps(data))
    _check_response(r, 'Login failed for user: {}'.format(user))
    return r.json()


class Connection(object):
    """A class to facilitate querying an IIASA scenario explorer database"""

    def __init__(self, name=None, creds=None, base_url=_BASE_URL):
        """
        Parameters
        ----------
        name : str, optional
            A valid database name. For available options, see
            valid_connections().
        creds : str, list-like, or dict, optional
            Either:
              - a yaml filename/path with entries of  'username' and 'password'
                (preferred)
              - an ordered container (tuple, list, etc.) with the same values
              - a dictionary with the same keys
        base_url: str, custom authentication server URL
        """
        self._base_url = base_url
        self._token = _get_token(creds, base_url=self._base_url)

        # connect if provided a name
        self._connected = None
        if name:
            self.connect(name)

    @property
    @lru_cache()
    def _connection_map(self):
        url = '/'.join([self._base_url, 'applications'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, 'Could not get valid connection list')
        aliases = set()
        conn_map = {}
        for x in r.json():
            if 'config' in x:
                env = next((r['value'] for r in x['config']
                            if r['path'] == 'env'), None)
                name = x['name']
                if env is not None:
                    if env in aliases:
                        logger.warning('Duplicate instance alias {}'
                                       .format(env))
                        conn_map[name] = name
                        first_duplicate = conn_map.pop(env)
                        conn_map[first_duplicate] = first_duplicate
                    else:
                        conn_map[env] = name
                    aliases.add(env)
                else:
                    conn_map[name] = name
        return conn_map

    @property
    @lru_cache()
    def valid_connections(self):
        """ Show a list of valid connection names (application aliases or
            names when alias is not available or duplicated)

        :return: list of str
        """
        return list(self._connection_map.keys())

    def connect(self, name):
        if name in self._connection_map:
            name = self._connection_map[name]

        valid = self._connection_map.values()
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

        url = '/'.join([self._base_url, 'applications', name, 'config'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, 'Could not get application information')
        response = r.json()
        idxs = {x['path']: i for i, x in enumerate(response)}

        self._base_url = response[idxs['baseUrl']]['value']
        # TODO: request the full citation to be added to this metadata instead
        #       of linking to the about page
        about = '/'.join([response[idxs['uiUrl']]['value'], '#', 'about'])
        logger.info(_CITE_MSG.format(name, about))

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
    def regions(self, include_synonyms=False):
        """All regions in the connected data source

        :param include_synonyms: whether to include synonyms
               (possibly leading to duplicate region names for
               regions with more than one synonym)
        """
        url = '/'.join([self._base_url, 'nodes?hierarchy=%2A'])
        headers = {'Authorization': 'Bearer {}'.format(self._token)}
        params = {'includeSynonyms': include_synonyms}
        r = requests.get(url, headers=headers, params=params)
        _check_response(r)
        return self.convert_regions_payload(r.content, include_synonyms)

    @staticmethod
    def convert_regions_payload(response, include_synonyms):
        df = pd.read_json(response, orient='records')
        if df.empty:
            return df
        if 'synonyms' not in df.columns:
            df['synonyms'] = [list()] * len(df)
        df = df.astype({
            'id': str,
            'name': str,
            'hierarchy': str,
            'parent': str,
            'synonyms': object
        })
        if include_synonyms:
            df = df[['name', 'synonyms']].explode('synonyms')
            return df.rename(columns={'name': 'region', 'synonyms': 'synonym'})
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
        # pass empty list to API if all variables selected
        if len(variables) == len(self.variables()):
            variables = []
        regions = _match(self.regions(), r_pattern)
        # pass empty list to API if all regions selected
        if len(regions) == len(self.regions()):
            regions = []

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
        logger.debug('Querying timeseries data '
                       'from {} with filter {}'.format(url, data))
        r = requests.post(url, headers=headers, data=data)
        _check_response(r)
        # refactor returned json object to be castable to an IamDataFrame
        dtype = dict(model=str, scenario=str, variable=str, unit=str,
                     region=str, year=int, value=float, version=int)
        df = pd.read_json(r.content, orient='records', dtype=dtype)
        logger.debug('Response size is {0} bytes, '
                       '{1} records'.format(len(r.content), len(df)))
        columns = ['model', 'scenario', 'variable', 'unit',
                   'region', 'year', 'value', 'time', 'meta',
                   'version']
        # keep only known columns or init empty df
        df = pd.DataFrame(data=df, columns=columns)
        # replace missing meta (for backward compatibility)
        df.fillna({'meta': 0}, inplace=True)
        df.fillna({'time': 0}, inplace=True)
        df.rename(columns={'time': 'subannual'}, inplace=True)
        # check if returned dataframe has subannual disaggregation, drop if not
        if pd.Series([i in [-1, 'year'] for i in df.subannual]).all():
            df.drop(columns='subannual', inplace=True)
        # check if there are multiple version for any model/scenario
        lst = (
            df[META_IDX + ['version']].drop_duplicates()
            .groupby(META_IDX).count().version
        )
        # checking if there are multiple versions
        # for every model/scenario combination
        if len(lst) > 1 and max(lst) > 1:
            raise ValueError('multiple versions for {}'.format(
                lst[lst > 1].index.to_list()))
        df.drop(columns='version', inplace=True)

        return df


def read_iiasa(name, meta=False, creds=None, base_url=_BASE_URL, **kwargs):
    """
    Query an IIASA database. See Connection.query() for more documentation

    Parameters
    ----------
    name : str
        A valid IIASA database name, see pyam.iiasa.valid_connections()
    meta : bool or list of strings
        If not False, also include metadata indicators (or subset if provided).
    creds : dict
        Credentials to access IXMP and authentication service APIs
        (username/password)
    base_url: str
        Authentication server URL
    kwargs :
        Arguments for pyam.iiasa.Connection.query()
    """
    conn = Connection(name, creds, base_url)
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
