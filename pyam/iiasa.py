from pathlib import Path
import json
import logging
import os.path
import requests

import httpx
import jwt
from requests.auth import AuthBase

import yaml
from functools import lru_cache

import numpy as np
import pandas as pd

from pyam.core import IamDataFrame
from pyam.utils import (
    META_IDX,
    IAMC_IDX,
    isstr,
    pattern_match,
    islistable,
)

logger = logging.getLogger(__name__)
# set requests-logger to WARNING only
logging.getLogger("requests").setLevel(logging.WARNING)

_AUTH_URL = "https://api.manager.ece.iiasa.ac.at"
_CITE_MSG = """
You are connected to the {} scenario explorer hosted by IIASA.
 If you use this data in any published format, please cite the
 data as provided in the explorer guidelines: {}
""".replace(
    "\n", ""
)

# path to local configuration settings
DEFAULT_IIASA_CREDS = Path("~").expanduser() / ".local" / "pyam" / "iiasa.yaml"

JWT_DECODE_ARGS = {"verify_signature": False, "verify_exp": True}


def set_config(user, password, file=None):
    """Save username and password for the IIASA API connection to a file"""
    file = Path(file) if file is not None else DEFAULT_IIASA_CREDS
    if not file.parent.exists():
        file.parent.mkdir(parents=True)

    with open(file, mode="w") as f:
        logger.info(f"Setting IIASA-connection configuration file: {file}")
        yaml.dump(dict(username=user, password=password), f, sort_keys=False)


def _read_config(file):
    """Read username and password for IIASA API connection from file"""
    with open(file, "r") as stream:
        return yaml.safe_load(stream)


def _check_response(r, msg="Error connecting to IIASA database", error=RuntimeError):
    if not r.ok:
        raise error(f"{msg}: {r.text}")


class SceSeAuth(AuthBase):
    def __init__(self, creds: str = None, auth_url: str = _AUTH_URL):
        """Connection to the Scenario Services Manager AAC service.

        Parameters
        ----------
        creds : pathlib.Path or str, optional
            Path to a file with authentication credentials
        auth_url : str, optionl
            Url of the authentication service
        """
        self.client = httpx.Client(base_url=auth_url, timeout=10.0, http2=True)
        self.access_token, self.refresh_token = None, None

        if creds is None:
            if DEFAULT_IIASA_CREDS.exists():
                self.creds = _read_config(DEFAULT_IIASA_CREDS)
            else:
                self.creds = None
        elif isinstance(creds, Path) or isstr(creds):
            self.creds = _read_config(creds)
        else:
            raise DeprecationWarning(
                "Passing credentials as clear-text is not allowed. "
                "Please use `pyam.iiasa.set_config(<user>, <password>)` instead!"
            )

        # if no creds, get anonymous token
        # TODO: explicit token for anonymous login will not be necessary for ixmp-server
        if self.creds is None:
            r = self.client.get("/legacy/anonym/")
            if r.status_code >= 400:
                raise ValueError("Unknown API error: " + r.text)
            self.user = None
            self.access_token = r.json()

        # else get user-token
        else:
            self.user = self.creds["username"]
            self.obtain_jwt()

    def __call__(self):
        try:
            # raises jwt.ExpiredSignatureError if token is expired
            jwt.decode(self.access_token, options=JWT_DECODE_ARGS)

        except jwt.ExpiredSignatureError:
            self.refresh_jwt()

        return {"Authorization": "Bearer " + self.access_token}

    def obtain_jwt(self):
        r = self.client.post("/v1/token/obtain/", json=self.creds)
        if r.status_code == 401:
            raise ValueError(
                "Credentials not valid to connect to https://manager.ece.iiasa.ac.at."
            )
        elif r.status_code >= 400:
            raise ValueError("Unknown API error: " + r.text)

        _json = r.json()
        self.access_token = _json["access"]
        self.refresh_token = _json["refresh"]

    def refresh_jwt(self):
        try:
            # raises jwt.ExpiredSignatureError if token is expired
            jwt.decode(self.refresh_token, options=JWT_DECODE_ARGS)
            r = self.client.post(
                "/v1/token/refresh/", json={"refresh": self.refresh_token}
            )
            if r.status_code >= 400:
                raise ValueError("Unknown API error: " + r.text)
            self.access_token = r.json()["access"]
        except jwt.ExpiredSignatureError:
            self.obtain_jwt()


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
    auth_url : str, optional
        custom authentication server URL

    Notes
    -----
    Credentials (username & password) are not required to access any public
    Scenario Explorer instances (i.e., with Guest login).
    """

    def __init__(self, name=None, creds=None, auth_url=_AUTH_URL):
        self._auth_url = auth_url  # scenario services manager API
        self._base_url = None  # database connection API

        self.auth = SceSeAuth(creds=creds, auth_url=self._auth_url)

        # connect if provided a name
        self._connected = None
        if name:
            self.connect(name)

        if self.auth.user is not None:
            logger.info(f"You are connected as user `{self.auth.user}`")
        else:
            logger.info("You are connected as an anonymous user")

    @property
    @lru_cache()
    def _connection_map(self):
        # TODO: application-list will be reimplemented in conjunction with ixmp-server
        r = self.auth.client.get("legacy/applications", headers=self.auth())
        if r.status_code >= 400:
            raise ValueError("Unknown API error: " + r.text)

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
        if name not in valid:
            raise ValueError(
                f"You do not have access to instance '{name}' or it does not exist. "
                "Use `Connection.valid_connections` for a list of accessible services."
            )

        # TODO: config will be reimplemented in conjunction with ixmp-server
        r = self.auth.client.get(
            f"legacy/applications/{name}/config", headers=self.auth()
        )
        if r.status_code >= 400:
            raise ValueError("Unknown API error: " + r.text)

        response = r.json()
        idxs = {x["path"]: i for i, x in enumerate(response)}

        self._base_url = response[idxs["baseUrl"]]["value"]
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

    @property
    def meta_columns(self):
        """Return the list of meta indicators in the connected resource"""
        url = "/".join([self._base_url, "metadata/types"])
        r = requests.get(url, headers=self.auth())
        _check_response(r)
        return pd.read_json(r.text, orient="records")["name"]

    def _query_index(self, default=True, meta=False, cols=[], **kwargs):
        # TODO: at present this reads in all data for all scenarios,
        #  it could be sped up in the future to try to query a subset
        _default = "true" if default else "false"
        _meta = "true" if meta else "false"
        add_url = f"runs?getOnlyDefaultRuns={_default}&includeMetadata={_meta}"
        url = "/".join([self._base_url, add_url])
        r = requests.get(url, headers=self.auth())
        _check_response(r)

        # cast response to dataframe, apply filter by kwargs, and return
        runs = pd.read_json(r.text, orient="records")
        if runs.empty:
            logger.warning("No permission to view model(s) or no scenarios exist.")
            return pd.DataFrame([], columns=META_IDX + ["version", "run_id"] + cols)

        if kwargs:
            keep = np.ones(len(runs), dtype=bool)
            for key, values in kwargs.items():
                if key not in META_IDX + ["version"]:
                    raise ValueError(f"Invalid filter: '{key}'")
                keep_col = pattern_match(pd.Series(runs[key].values), values)
                keep = np.logical_and(keep, keep_col)
            return runs[keep]
        else:
            return runs

    def index(self, default=True, **kwargs):
        """Return the index of models and scenarios

        Parameters
        ----------
        default : bool, optional
            If `True`, return *only* the default version of a model/scenario.
            Any model/scenario without a default version is omitted.
            If `False`, returns all versions.
        kwargs
            Arguments to filer by *model* and *scenario*, `*` can be used as wildcard.
        """
        cols = ["version"] if default else ["version", "is_default"]
        return self._query_index(default, **kwargs)[META_IDX + cols].set_index(META_IDX)

    def meta(self, default=True, run_id=False, **kwargs):
        """Return categories and indicators (meta) of scenarios

        Parameters
        ----------
        default : bool, optional
            Return *only* the default version of each scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If `False`, return all versions.
        run_id : bool, optional
            Include "run id" column
        kwargs
            Arguments to filer by *model* and *scenario*, `*` can be used as wildcard
        """
        df = self._query_index(default, meta=True, **kwargs)

        cols = ["version"] if default else ["version", "is_default"]
        if run_id:
            cols.append("run_id")

        meta = df[META_IDX + cols]
        if not meta.empty and df.metadata.any():
            extra_meta = pd.DataFrame.from_records(df.metadata)
            meta = pd.concat([meta, extra_meta], axis=1)

        return meta.set_index(META_IDX + ([] if default else ["version"]))

    def properties(self, default=True, **kwargs):
        """Return the audit properties of scenarios

        Parameters
        ----------
        default : bool, optional
            Return *only* the default version of each scenario.
            Any (`model`, `scenario`) without a default version is omitted.
            If :obj:`False`, return all versions.
        kwargs
            Arguments to filer by *model* and *scenario*, `*` can be used as wildcard
        """
        audit_cols = ["cre_user", "cre_date", "upd_user", "upd_date"]
        other_cols = ["version"] if default else ["version", "is_default"]
        cols = audit_cols + other_cols

        _df = self._query_index(default, meta=True, cols=cols, **kwargs)
        audit_mapping = dict([(i, i.replace("_", "ate_")) for i in audit_cols])

        return _df.set_index(META_IDX).rename(columns=audit_mapping)

    def models(self):
        """List all models in the connected resource"""
        return pd.Series(self._query_index()["model"].unique(), name="model")

    def scenarios(self):
        """List all scenarios in the connected resource"""
        return pd.Series(self._query_index()["scenario"].unique(), name="scenario")

    @lru_cache()
    def variables(self):
        """List all variables in the connected resource"""
        url = "/".join([self._base_url, "ts"])
        r = requests.get(url, headers=self.auth())
        _check_response(r)
        df = pd.read_json(r.text, orient="records")
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
        url = "/".join([self._base_url, "nodes?hierarchy=%2A"])
        params = {"includeSynonyms": include_synonyms}
        r = requests.get(url, headers=self.auth(), params=params)
        _check_response(r)
        return self.convert_regions_payload(r.text, include_synonyms)

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
        headers = self.auth().copy()
        headers["Content-Type"] = "application/json"

        # retrieve meta (with run ids) or only index
        if meta:
            _meta = self.meta(default=default, run_id=True)
            # downselect to subset of meta columns if given as list
            if islistable(meta):
                # always merge 'version' (even if not requested explicitly)
                # 'run_id' is required to determine `_args`, dropped later
                _meta = _meta[set(meta).union(["version", "run_id"])]
        else:
            _meta = self._query_index(default=default).set_index(META_IDX)

        # return nothing if no data exists at all
        if _meta.empty:
            return

        # retrieve data
        _args = json.dumps(self._query_post(_meta, default=default, **kwargs))
        url = "/".join([self._base_url, "runs/bulk/ts"])
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
        data = pd.read_json(r.text, orient="records", dtype=dtype)
        logger.debug(f"Response: {len(r.text)} bytes, {len(data)} records")
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
            index = META_IDX
            data.drop(columns="version", inplace=True)
        else:
            index = META_IDX + ["version"]
            logger.info(f"Initializing `IamDataFrame` with non-default index {index}")

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
        | Name of an IIASA Scenario Explorer database instance.
        | See :attr:`pyam.iiasa.Connection.valid_connections`.
    default : bool, optional
        Return *only* the default version of each scenario.
        Any (`model`, `scenario`) without a default version is omitted.
        If :obj:`False`, return all versions.
    meta : bool or list of strings, optional
        If `True`, include all meta categories & quantitative indicators
        (or subset if list is given).
    creds : str or :class:`pathlib.Path`, optional
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


def lazy_read_iiasa(
    file, name, default=True, meta=True, creds=None, base_url=_AUTH_URL, **kwargs
):
    """
    Try to load data from a local cache, failing that, loads it from the internet.

    Check if the file in a given location is an up-to-date version of an IIASA
    database. If so, load it. If not, load  data from the IIASA scenario explorer
    database API and save to that location. Does not check that the previously read
    version is a complete instance of the database, so if the initial load applies a
    filter, you will read only data that passes the same filter as well as any
    additional filter you apply.

    Parameters
    ----------
    file : str or :class:`pathlib.Path`
        The location to test for valid data and save the data if not up-to-date. Must be
        either xls, xlsx or csv.
    name : str
        | Name of an IIASA Scenario Explorer database instance.
        | See :attr:`pyam.iiasa.Connection.valid_connections`.
    default : bool, optional
        Return *only* the default version of each scenario.
        Any (`model`, `scenario`) without a default version is omitted.
        If :obj:`False`, return all versions.
    meta : bool or list of strings, optional
        If `True`, include all meta categories & quantitative indicators
        (or subset if list is given).
    creds : str or :class:`pathlib.Path`, optional
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

    file = Path(file)
    assert file.suffix in [
        ".csv",
        ".xlsx",
        ".xls",
    ], "We will only read and write to csv, xls and xlsx format."
    if os.path.exists(file):
        date_set = pd.to_datetime(os.path.getmtime(file), unit="s")
        version_info = Connection(name, creds, base_url).properties()
        latest_new = np.nanmax(pd.to_datetime(version_info["create_date"]))
        latest_update = np.nanmax(pd.to_datetime(version_info["update_date"]))
        latest = pd.Series([latest_new, latest_update]).max()
        if latest < date_set:
            old_read = IamDataFrame(file)
            if kwargs:
                old_read = old_read.filter(**kwargs)
            logger.info("Database read from file")
            return old_read
        else:
            logger.info("Database out of date and will be re-downloaded")
    # If we get here, we need to redownload the database
    new_read = read_iiasa(
        name, meta=meta, default=default, creds=creds, base_url=base_url, **kwargs
    )
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    if file.suffix == ".csv":
        new_read.to_csv(file)
    else:
        new_read.to_excel(file)
    return new_read
