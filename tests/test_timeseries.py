#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pyam import cross_threshold
import pytest


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
    obs = cross_threshold(y, 2, direction='from below')
    assert obs == [2007]


def test_cross_treshold_from_above():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = cross_threshold(y, 2, direction='from above')
    assert obs == [2011]


def test_cross_treshold_direction_error():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    pytest.raises(ValueError, cross_threshold, x=y, threshold=2,
                  direction='up')
