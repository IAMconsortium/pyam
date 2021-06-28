from pathlib import Path
import json
import logging
import requests
import yaml
from functools import lru_cache


import numpy as np
import pandas as pd

from collections.abc import Mapping
from pyam.core import IamDataFrame
from pyam.utils import (
    META_IDX,
    IAMC_IDX,
    isstr,
    pattern_match,
    DEFAULT_META_INDEX,
    islistable,
)

logger = logging.getLogger(__name__)
# set requests-logger to WARNING only
logging.getLogger("requests").setLevel(logging.WARNING)

_AUTH_URL = "https://db1.ene.iiasa.ac.at/EneAuth/config/v1"
_CITE_MSG = """
You are connected to the {} scenario explorer hosted by IIASA.
 If you use this data in any published format, please cite the
 data as provided in the explorer guidelines: {}
""".replace(
    "\n", ""
)

# path to local configuration settings
DEFAULT_IIASA_CREDS = Path("~").expanduser() / ".local" / "pyam" / "iiasa.yaml"


def set_config(user, password, file=None):
    """Save username and password for the IIASA API connection to a file"""
    file = Path(file) if file is not None else DEFAULT_IIASA_CREDS
    if not file.parent.exists():
        file.parent.mkdir(parents=True)

    with open(file, mode="w") as f:
        logger.info(f"Setting IIASA-connection configuration file: {file}")
        yaml.dump(dict(username=user, password=password), f)


def _get_config(file=None):
    """Read username and password for IIASA API connection from file"""
    file = Path(file) if file is not None else DEFAULT_IIASA_CREDS
    if file.exists():
        with open(file, "r") as stream:
            return yaml.safe_load(stream)


def _check_response(r, msg="Trouble with request", error=RuntimeError):
    if not r.ok:
        raise error("{}: {}".format(msg, str(r.text)))


def _get_token(creds, base_url):
    """Parse credentials and get token from IIASA authentication service"""

    # try reading default config or parse file
    if creds is None:
        creds = _get_config()
    elif isinstance(creds, Path) or isstr(creds):
        _creds = _get_config(creds)
        if _creds is None:
            logger.error(f"Could not read credentials from `{creds}`")
        creds = _creds
    else:
        msg = (
            "Passing credentials as clear-text is not allowed. "
            "Please use `pyam.iiasa.set_config(<user>, <password>)` instead!"
        )
        raise DeprecationWarning(msg)

    # if (still) no creds, get anonymous auth and return
    if creds is None:
        url = "/".join([base_url, "anonym"])
        r = requests.get(url)
        _check_response(r, "Could not get anonymous token")
        return r.json(), None

    # parse creds, write warning
    if isinstance(creds, Mapping):
        user, pw = creds["username"], creds["password"]
    else:
        user, pw = creds

    # get user token
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    data = {"username": user, "password": pw}
    url = "/".join([base_url, "login"])
    r = requests.post(url, headers=headers, data=json.dumps(data))
    _check_response(r, "Login failed for user: {}".format(user))
    return r.json(), user


class Connection(object):
    """A class to facilitate querying an IIASA Scenario Explorer database API

    Parameters
    ----------
    name : str, optional
        The name of a database API.
        See :attr:`pyam.iiasa.Connection.valid_connections` for a list
        of available APIs.
    creds : str or :class:`pathlib.Path`, optional
        By default, the class will search for user credentials which
        were set using :meth:`pyam.iiasa.set_config`.
        Alternatively, you can provide a path to a yaml file
        with entries of 'username' and 'password'.
    base_url : str, optional
        custom authentication server URL

    Notes
    -----
    Credentials (username & password) are not required to access any public
    Scenario Explorer instances (i.e., with Guest login).
    """

    def __init__(self, name=None, creds=None, auth_url=_AUTH_URL):
        self._auth_url = auth_url
        self._token, self._user = _get_token(creds, base_url=self._auth_url)

        # connect if provided a name
        self._connected = None
        if name:
            self.connect(name)

        if self._user:
            logger.info(f"You are connected as user `{self._user}`")
        else:
            logger.info("You are connected as an anonymous user")

    @property
    @lru_cache()
    def _connection_map(self):
        url = "/".join([self._auth_url, "applications"])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, "Could not get valid connection list")
        aliases = set()
        conn_map = {}
        for x in r.json():
            if "config" in x:
                env = next(
                    (r["value"] for r in x["config"] if r["path"] == "env"), None
                )
                name = x["name"]
                if env is not None:
                    if env in aliases:
                        logger.warning("Duplicate instance alias {}".format(env))
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
        """Return available resources (database API connections)"""
        return list(self._connection_map.keys())

    def connect(self, name):
        """Connect to a specific resource (database API)"""
        if name in self._connection_map:
            name = self._connection_map[name]

        valid = self._connection_map.values()
        if len(valid) == 0:
            raise RuntimeError(
                "No valid connections found for the provided credentials."
            )

        if name not in valid:
            msg = """
            {} not recognized as a valid connection name.
            Choose from one of the supported connections for your user: {}.
            """
            raise ValueError(msg.format(name, self._connection_map.keys()))

        url = "/".join([self._auth_url, "applications", name, "config"])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r, "Could not get application information")
        response = r.json()
        idxs = {x["path"]: i for i, x in enumerate(response)}

        self._auth_url = response[idxs["baseUrl"]]["value"]
        # TODO: proper citation (as metadata) instead of link to the about page
        if "uiUrl" in idxs:
            about = "/".join([response[idxs["uiUrl"]]["value"], "#", "about"])
            logger.info(_CITE_MSG.format(name, about))

        # TODO: use API "nice-name"
        self._connected = name

    @property
    def current_connection(self):
        """Currently connected resource (database API connection)"""
        return self._connected

    def index(self, default=True):
        """Return the index of models and scenarios in the connected resource

        Parameters
        ----------
        default : bool, optional
            If `True`, return *only* the default version of a model/scenario.
            Any model/scenario without a default version is omitted.
            If `False`, returns all versions.
        """
        cols = ["version"] if default else ["version", "is_default"]
        return self._query_index(default)[META_IDX + cols].set_index(META_IDX)

    @lru_cache()
    def _query_index(self, default=True, meta=False):
        # TODO: at present this reads in all data for all scenarios,
        #  it could be sped up in the future to try to query a subset
        _default = "true" if default else "false"
        _meta = "true" if meta else "false"
        add_url = f"runs?getOnlyDefaultRuns={_default}&includeMetadata={_meta}"
        url = "/".join([self._auth_url, add_url])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)

        # cast response to dataframe and return
        return pd.read_json(r.content, orient="records")

    @property
    @lru_cache()
    def meta_columns(self):
        """Return the list of meta indicators in the connected resource"""
        url = "/".join([self._auth_url, "metadata/types"])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
        return pd.read_json(r.content, orient="records")["name"]

    def meta(self, default=True, **kwargs):
        """Return categories and indicators (meta) of scenarios

        Parameters
        ----------
        default : bool, optional
            Return *only* the default version of each scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If `False`, return all versions.
        """
        df = self._query_index(default, meta=True)
        cols = ["version"] if default else ["version", "is_default"]
        if kwargs:
            if kwargs.pop("run_id", False):
                cols += ["run_id"]
        index = DEFAULT_META_INDEX + ([] if default else ["version"])

        def extract(row):
            return (
                pd.concat([row[META_IDX + cols], pd.Series(row.metadata)])
                .to_frame()
                .T.set_index(index)
            )

        return pd.concat([extract(row) for i, row in df.iterrows()], sort=False)

    def properties(self, default=True):
        """Return the audit properties of scenarios

        Parameters
        ----------
        default : bool, optional
            Return *only* the default version of each scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If :obj:`False`, return all versions.
        """
        _df = self._query_index(default, meta=True)
        audit_cols = ["cre_user", "cre_date", "upd_user", "upd_date"]
        audit_mapping = dict([(i, i.replace("_", "ate_")) for i in audit_cols])
        other_cols = ["version"] if default else ["version", "is_default"]

        return (
            _df[META_IDX + other_cols + audit_cols]
            .set_index(META_IDX)
            .rename(columns=audit_mapping)
        )

    def models(self):
        """List all models in the connected resource"""
        return pd.Series(self._query_index()["model"].unique(), name="model")

    def scenarios(self):
        """List all scenarios in the connected resource"""
        return pd.Series(self._query_index()["scenario"].unique(), name="scenario")

    @lru_cache()
    def variables(self):
        """List all variables in the connected resource"""
        url = "/".join([self._auth_url, "ts"])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        r = requests.get(url, headers=headers)
        _check_response(r)
        df = pd.read_json(r.content, orient="records")
        return pd.Series(df["variable"].unique(), name="variable")

    @lru_cache()
    def regions(self, include_synonyms=False):
        """List all regions in the connected resource

        Parameters
        ----------
        include_synonyms : bool
            whether to include synonyms
            (possibly leading to duplicate region names for
            regions with more than one synonym)
        """
        url = "/".join([self._auth_url, "nodes?hierarchy=%2A"])
        headers = {"Authorization": "Bearer {}".format(self._token)}
        params = {"includeSynonyms": include_synonyms}
        r = requests.get(url, headers=headers, params=params)
        _check_response(r)
        return self.convert_regions_payload(r.content, include_synonyms)

    @staticmethod
    def convert_regions_payload(response, include_synonyms):
        df = pd.read_json(response, orient="records")
        if df.empty:
            return df
        if "synonyms" not in df.columns:
            df["synonyms"] = [list()] * len(df)
        df = df.astype(
            {
                "id": str,
                "name": str,
                "hierarchy": str,
                "parent": str,
                "synonyms": object,
            }
        )
        if include_synonyms:
            df = df[["name", "synonyms"]].explode("synonyms")
            return df.rename(columns={"name": "region", "synonyms": "synonym"})
        return pd.Series(df["name"].unique(), name="region")

    def _query_post(self, meta, default=True, **kwargs):
        def _get_kwarg(k):
            # TODO refactor API to return all models if model-list is empty
            x = kwargs.pop(k, "*" if k == "model" else [])
            return [x] if isstr(x) else x

        m_pattern = _get_kwarg("model")
        s_pattern = _get_kwarg("scenario")
        v_pattern = _get_kwarg("variable")
        r_pattern = _get_kwarg("region")

        def _match(data, patterns):
            # this is empty, return empty list which means "everything"
            if not patterns:
                return []
            # otherwise match everything
            matches = np.array([False] * len(data))
            for p in patterns:
                matches |= pattern_match(data, p)
            return data[matches].unique()

        # drop non-default runs if only default is requested
        if default and hasattr(meta, "is_default"):
            meta = meta[meta.is_default]

        # determine relevant run id's
        meta = meta.reset_index()
        models = _match(meta["model"], m_pattern)
        scenarios = _match(meta["scenario"], s_pattern)
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
        logger.debug(
            f"Prepared filter for {len(regions)} region(s), "
            f"{len(variables)} variables and {len(runs)} runs"
        )
        data = {
            "filters": {
                "regions": list(regions),
                "variables": list(variables),
                "runs": list(runs),
                "years": [],
                "units": [],
                "timeslices": [],
            }
        }
        return data

    def query(self, default=True, meta=True, **kwargs):
        """Query the connected resource for timeseries data (with filters)

        Parameters
        ----------
        default : bool, optional
            Return *only* the default version of each scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If :obj:`False`, return all versions.
        meta : bool or list, optional
            If :obj:`True`, merge all meta columns indicators
            (or subset if list is given).
        kwargs
            Available keyword arguments include

            - model
            - scenario
            - region
            - variable

        Returns
        -------
        IamDataFrame

        Examples
        --------
        You can read from a :class:`pyam.iiasa.Connection` instance using
        keyword arguments similar to filtering an :class:`IamDataFrame`:

        .. code-block:: python

            Connection.query(model='MESSAGE*', scenario='SSP2*',
                             variable=['Emissions|CO2', 'Primary Energy'])

        """
        headers = {
            "Authorization": "Bearer {}".format(self._token),
            "Content-Type": "application/json",
        }

        # retrieve meta (with run ids) or only index
        if meta:
            _meta = self.meta(default=default, run_id=True)
            # downselect to subset of meta columns if given as list
            if islistable(meta):
                # always merge 'version' (even if not requested explicitly)
                # 'run_id' is required to determine `_args`, dropped later
                _meta = _meta[set(meta).union(["version", "run_id"])]
        else:
            _meta = self._query_index(default=default).set_index(DEFAULT_META_INDEX)

        # retrieve data
        _args = json.dumps(self._query_post(_meta, default=default, **kwargs))
        url = "/".join([self._auth_url, "runs/bulk/ts"])
        logger.debug(f"Query timeseries data from {url} with data {_args}")
        r = requests.post(url, headers=headers, data=_args)
        _check_response(r)
        # refactor returned json object to be castable to an IamDataFrame
        dtype = dict(
            model=str,
            scenario=str,
            variable=str,
            unit=str,
            region=str,
            year=int,
            value=float,
            version=int,
        )
        data = pd.read_json(r.content, orient="records", dtype=dtype)
        logger.debug(f"Response: {len(r.content)} bytes, {len(data)} records")
        cols = IAMC_IDX + ["year", "value", "subannual", "version"]
        # keep only known columns or init empty df
        data = pd.DataFrame(data=data, columns=cols)

        # check if timeseries data has subannual disaggregation, drop if not
        if "subannual" in data:
            timeslices = data.subannual.dropna().unique()
            if all([i in [-1, "Year"] for i in timeslices]):
                data.drop(columns="subannual", inplace=True)

        # define the index for the IamDataFrame
        if default:
            index = DEFAULT_META_INDEX
            data.drop(columns="version", inplace=True)
        else:
            index = DEFAULT_META_INDEX + ["version"]
            logger.info(
                "Initializing an `IamDataFrame` " f"with non-default index {index}"
            )

        # merge meta indicators (if requested) and cast to IamDataFrame
        if meta:
            # 'run_id' is necessary to retrieve data, not returned by default
            if not (islistable(meta) and "run_id" in meta):
                _meta.drop(columns="run_id", inplace=True)
            return IamDataFrame(data, meta=_meta, index=index)
        else:
            return IamDataFrame(data, index=index)


def read_iiasa(name, default=True, meta=True, creds=None, base_url=_AUTH_URL, **kwargs):
    """Query an IIASA Scenario Explorer database API and return as IamDataFrame

    Parameters
    ----------
    name : str
        A valid name of an IIASA scenario explorer instance,
        see :attr:`pyam.iiasa.Connection.valid_connections`
    default : bool, optional
        Return *only* the default version of each scenario.
        Any (`model`, `scenario`) without a default version is omitted.
        If :obj:`False`, return all versions.
    meta : bool or list of strings, optional
        If `True`, include all meta categories & quantitative indicators
        (or subset if list is given).
    creds : str, :class:`pathlib.Path`, list-like, or dict, optional
        | Credentials (username & password) are not required to access
          any public Scenario Explorer instances (i.e., with Guest login).
        | See :class:`pyam.iiasa.Connection` for details.
        | Use :meth:`pyam.iiasa.set_config` to set credentials
          for accessing private/restricted Scenario Explorer instances.
    base_url : str
        Authentication server URL
    kwargs
        Arguments for :meth:`pyam.iiasa.Connection.query`
    """
    return Connection(name, creds, base_url).query(default=default, meta=meta, **kwargs)
