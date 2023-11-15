import ixmp4


def write_to_ixmp4(df, platform: ixmp4.Platform):
    """Save all scenarios as new default runs in an ixmp4 platform database instance

    Parameters
    ----------
    df : pyam.IamDataFrame
        The IamDataFrame instance with scenario data
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    """
    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    for model, scenario in df.index:
        _df = df.filter(model=model, scenario=scenario)

        run = platform.Run(model=model, scenario=scenario, version="new")
        run.iamc.add(_df.data)
        for key, value in dict(_df.meta.iloc[0]).items():
            run.meta[key] = value
        run.set_as_default()