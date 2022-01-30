import numpy as np


def filter_by_time_domain(values, levels, codes):
    """Internal implementation to filter by time domain"""

    if values == "year":
        matches = [i for (i, label) in enumerate(levels) if isinstance(label, int)]
    elif values == "datetime":
        matches = [i for (i, label) in enumerate(levels) if not isinstance(label, int)]
    else:
        raise ValueError(f"Filter by `datetime='{values}'` not supported!")

    return np.isin(codes, matches)
