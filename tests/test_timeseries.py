#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
import pandas as pd
from pyam import fill_series, cumulative, cross_threshold, to_int
import pytest


def test_fill_series():
    # note that the series is not order and the index is defined as float
    y = pd.Series(data=[np.nan, 1, 4, 1], index=[2002.0, 2008.0, 2005.0, 2013.0])
    assert fill_series(y, 2006) == 3.0


def test_fill_series_datetime():
    # note that the series is not order and the index is defined as float
    y = pd.Series(
        data=[3, 1],
        index=[datetime.datetime(2001, 1, 1), datetime.datetime(2003, 1, 1)],
    )
    assert fill_series(y, datetime.datetime(2002, 1, 1)) == 2.0


def test_fill_series_out_of_range():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002.0, 2005.0, 2007.0, 2013.0])
    assert fill_series(y, 2001) is np.nan


def test_cols_to_int():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002.0, 2007.5, 2003.0, 2013.0])
    pytest.raises(ValueError, to_int, x=y)


def test_cumulative():
    # note that the series is not order and the index is defined as float
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002.0, 2007.0, 2003.0, 2013.0])
    assert cumulative(y, 2008, 2013) == 6


def test_cumulative_out_of_range():
    # set logger level to exclude warnings in unit test output
    logging.getLogger("pyam").setLevel("ERROR")
    # note that the series is not order and the index is defined as float
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002.0, 2005.0, 2007.0, 2013.0])
    assert cumulative(y, 2008, 2015) is np.nan
    logging.getLogger("pyam").setLevel("NOTSET")


def test_cross_treshold():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = cross_threshold(y, 2)
    assert obs == [2007, 2011]


def test_cross_treshold_empty():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = cross_threshold(y, 4)
    assert obs == []


def test_cross_treshold_from_below():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = cross_threshold(y, 2, direction="from below")
    assert obs == [2007]


def test_cross_treshold_from_above():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = cross_threshold(y, 2, direction="from above")
    assert obs == [2011]


def test_cross_treshold_direction_error():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    pytest.raises(ValueError, cross_threshold, x=y, threshold=2, direction="up")
