import logging

import pandas as pd
import plotly.graph_objects as go
from pyam.index import get_index_levels

logger = logging.getLogger(__name__)


def sankey(df, mapping):
    """Plot sankey diagram of existing data using plotly.

        Currently only for one year possible.
        Parameters
        ----------
        df : pd.DataFrame
            data to plot as wide format
        mapping : dict
            assigns the source and target component of a variable
            {variable: (source, target)}.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
    """
    if len(get_index_levels(df, 'region')) != 1:
        msg = 'Can only plot one region. Filter or aggregate before!'
        raise ValueError(msg)
    if len(df.columns) != 1:
        msg = 'Can only plot one year. Filter before!'
        raise ValueError(msg)

    mapping_df = pd.DataFrame.from_dict(mapping, orient='index',
                                        columns=['source', 'target'])
    mapping_df_merged = mapping_df.merge(df, how='left', left_index=True,
                                         right_on='variable')
    label_set = set(mapping_df_merged['source'].
                    append(mapping_df_merged['target']))
    label_series = pd.Series(list(label_set))
    for ind, val in label_series.items():
        mapping_df_merged.replace(val, ind, inplace=True)
    region = get_index_levels(df, 'region')[0]
    unit = get_index_levels(df, 'unit')[0]
    year = df.columns[0]
    fig = go.Figure(data=[go.Sankey(
        valuesuffix=unit,
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=label_series,
            hovertemplate='%{label}: %{value}<extra></extra>',
            color="blue"
        ),
        link=dict(
            source=mapping_df_merged.source,
            target=mapping_df_merged.target,
            value=mapping_df_merged[year],
            hovertemplate='"%{source.label}" to "%{target.label}": \
                %{value}<extra></extra>'
    ))])
    fig.update_layout(title_text="%s %s" % (region, year), font_size=10)
    return fig
