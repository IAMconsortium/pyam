from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from pyam import IamDataFrame, read_datapackage
from pyam.testing import assert_iamframe_equal

from conftest import TEST_DATA_DIR

FILTER_ARGS = dict(scenario='scen_a')


def test_io_csv(test_df, tmpdir):
    # write to csv
    file = tmpdir / 'testing_io_write_read.csv'
    test_df.to_csv(file)

    # read from csv and assert that `data` tables are equal
    import_df = IamDataFrame(file)
    pd.testing.assert_frame_equal(test_df.data, import_df.data)


@pytest.mark.parametrize("meta_args", [
    [{}, {}],
    [dict(include_meta='foo'), dict(meta_sheet_name='foo')]
])
def test_io_xlsx(test_df, meta_args, tmpdir):
    # add column to `meta`
    test_df.set_meta(['a', 'b'], 'string')

    # write to xlsx (direct file name and ExcelWriter, see #300)
    file = tmpdir / 'testing_io_write_read.xlsx'
    for f in [file, pd.ExcelWriter(file)]:
        test_df.to_excel(f, **meta_args[0])
        if isinstance(f, pd.ExcelWriter):
            f.close()

        # read from xlsx
        import_df = IamDataFrame(file, **meta_args[1])

        # assert that IamDataFrame instances are equal and delete file
        assert_iamframe_equal(test_df, import_df)


def test_init_df_with_na_unit(test_pd_df, tmpdir):
    # missing values in the unit column are replaced by an empty string
    test_pd_df.loc[1, 'unit'] = np.nan
    df = IamDataFrame(test_pd_df)
    assert df.unit == ['', 'EJ/yr']

    # writing to file and importing as pandas returns `nan`, not empty string
    file = tmpdir / 'na_unit.csv'
    df.to_csv(file)
    df_csv = pd.read_csv(file)
    assert np.isnan(df_csv.loc[1, 'Unit'])
    IamDataFrame(file)  # reading from file as IamDataFrame works

    file = tmpdir / 'na_unit.xlsx'
    df.to_excel(file)
    df_excel = pd.read_excel(file)
    assert np.isnan(df_excel.loc[1, 'Unit'])
    IamDataFrame(file)  # reading from file as IamDataFrame works


@pytest.mark.parametrize("args", [{}, dict(sheet_name='meta')])
def test_load_meta(test_df, args):
    file = TEST_DATA_DIR / 'testing_metadata.xlsx'
    test_df.load_meta(file, **args)
    obs = test_df.meta

    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a', 'scen_b'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_ssp_database_downloaded_file(test_pd_df):
    exp = IamDataFrame(test_pd_df).filter(**FILTER_ARGS).as_pandas()
    file = TEST_DATA_DIR / 'test_SSP_database_raw_download.xlsx'
    obs_df = IamDataFrame(file)
    pd.testing.assert_frame_equal(obs_df.as_pandas(), exp)


def test_load_rcp_database_downloaded_file(test_pd_df):
    exp = IamDataFrame(test_pd_df).filter(**FILTER_ARGS).as_pandas()
    file = TEST_DATA_DIR / 'test_RCP_database_raw_download.xlsx'
    obs_df = IamDataFrame(file)
    pd.testing.assert_frame_equal(obs_df.as_pandas(), exp)


def test_io_datapackage(test_df, tmpdir):
    # add column to `meta` and write to datapackage
    file = Path(tmpdir) / 'foo.zip'
    test_df.set_meta(['a', 'b'], 'string')
    test_df.to_datapackage(file)

    # read from csv assert that IamDataFrame instances are equal
    import_df = read_datapackage(file)
    assert_iamframe_equal(test_df, import_df)
