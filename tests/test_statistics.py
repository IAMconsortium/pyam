# -*- coding: utf-8 -*-

import pandas as pd
from pyam import Statistics


def test_statistics(plot_df):
    plot_df.set_meta(meta=['a', 'b', 'b', 'a'], name='category')
    stats = Statistics(df=plot_df, groupby={'category': ['b', 'a']})

    # test describing as pd.DataFrame
    primary = plot_df.filter(variable='Primary Energy', year=2005).timeseries()
    stats.describe(data=primary, header='primary')
    # test describing as unamed pd.Series with `subheader` arg
    coal = plot_df.filter(variable='Primary Energy|Coal').timeseries()[2005]
    coal.name = None
    stats.describe(data=coal, header='coal', subheader=2005)

    obs = stats.summarize()

    idx = pd.MultiIndex(levels=[['category'], ['b', 'a']],
                        labels=[[0, 0], [0, 1]], names=['', ''])
    cols = pd.MultiIndex(levels=[['count', 'primary', 'coal'],
                                 ['', 2005]],
                         labels=[[0, 1, 2], [0, 1, 1]], names=[None, None])
    exp = pd.DataFrame(data=[['2', '1.35 (2.00, 0.70)', '0.42 (0.50, 0.35)'],
                             ['2', '1.20 (1.40, 1.00)', '0.42 (0.50, 0.35)']],
                       index=idx, columns=cols)
    pd.testing.assert_frame_equal(obs, exp)
