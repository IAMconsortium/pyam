#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pyam


def test_cross_treshold():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = pyam.cross_threshold(y, 2)
    assert obs == [2007, 2011]


def test_cross_treshold_empty():
    y = pd.Series(data=[np.nan, 1, 3, 1], index=[2002, 2005, 2007, 2013])
    obs = pyam.cross_threshold(y, 4)
    assert obs == []
