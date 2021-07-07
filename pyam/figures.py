import logging

import pandas as pd
from pyam.index import get_index_levels

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    go = None
    HAS_PLOTLY = False


def sankey(df, mapping):
    """Plot a sankey diagram

    It is currently only possible to create this diagram for single years.

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`
        Data to be plotted
    mapping : dict
        Assigns the source and target component of a variable

        .. code-block:: python

            {
                variable: (source, target),
            }

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure`
    """
    if not HAS_PLOTLY:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency `plotly`, use pip or conda to install"
        )
    # Check for duplicates
    for col in [name for name in df.dimensions if name != "variable"]:
        levels = get_index_levels(df._data, col)
        if len(levels) > 1:
            raise ValueError(f"Non-unique values in column {col}: {levels}")

    # Concatenate the data with source and target columns
    _df = pd.DataFrame.from_dict(
        mapping, orient="index", columns=["source", "target"]
    ).merge(df._data, how="left", left_index=True, right_on="variable")
    label_mapping = dict(
        [(label, i) for i, label in enumerate(set(_df["source"].append(_df["target"])))]
    )
    _df.replace(label_mapping, inplace=True)
    region = get_index_levels(_df, "region")[0]
    unit = get_index_levels(_df, "unit")[0]
    year = get_index_levels(_df, "year")[0]
    fig = go.Figure(
        data=[
            go.Sankey(
                valuesuffix=unit,
                node=dict(
                    pad=15,
                    thickness=10,
                    line=dict(color="black", width=0.5),
                    label=pd.Series(list(label_mapping)),
                    hovertemplate="%{label}: %{value}<extra></extra>",
                    color="blue",
                ),
                link=dict(
                    source=_df.source,
                    target=_df.target,
                    value=_df.value,
                    hovertemplate='"%{source.label}" to "%{target.label}": \
                %{value}<extra></extra>',
                ),
            )
        ]
    )
    fig.update_layout(title_text=f"region: {region}, year: {year}", font_size=10)
    return fig
