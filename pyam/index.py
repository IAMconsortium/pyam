def get_index_levels(df, level):
    """Return the levels for a specific level"""
    for i, n in enumerate(df.index.names):
        if n == level:
            return list(df.index.levels[i])

    raise ValueError(f'This object does not have an index level `{level}`!')


def replace_index_values(df, level, mapping, verify_integrity=True):
    """Replace one or several values by mapping at a particular index level"""
    return df.index.set_levels([mapping[i] if i in mapping else i
                                for i in get_index_levels(df, level)], level)
