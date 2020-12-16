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
            Assigns the source and target component of a variable

            .. code-block:: python

            {
                variable: (source, target),
                }

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

    _df = (
        pd.DataFrame.from_dict(mapping, orient='index',
                               columns=['source', 'target'])
        .merge(df, how='left', left_index=True, right_on='variable')
    )
    label_mapping = dict([(label, i) for i, label
                          in enumerate(set(_df['source']
                                           .append(_df['target'])))])
    _df.replace(label_mapping, inplace=True)
    region = get_index_levels(df, 'region')[0]
    unit = get_index_levels(df, 'unit')[0]
    year = df.columns[0]
    fig = go.Figure(data=[go.Sankey(
        valuesuffix=unit,
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=pd.Series(list(label_mapping)),
            hovertemplate='%{label}: %{value}<extra></extra>',
            color="blue"
        ),
        link=dict(
            source=_df.source,
            target=_df.target,
            value=_df[year],
            hovertemplate='"%{source.label}" to "%{target.label}": \
                %{value}<extra></extra>'
        )
    )])
    fig.update_layout(title_text=f'region: {region}, year: {year}',
                      font_size=10)
    return fig
