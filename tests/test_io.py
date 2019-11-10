import os
import pandas as pd
import numpy as np
import pytest

from pyam import IamDataFrame

from conftest import TEST_DATA_DIR


def test_io_csv(meta_df):
    # write to csv
    file = 'testing_io_write_read.csv'
    meta_df.to_csv(file)

    # read from csv
    import_df = IamDataFrame(file)

    # assert that `data` tables are equal and delete file
    pd.testing.assert_frame_equal(meta_df.data, import_df.data)
    os.remove(file)


def test_io_xlsx(meta_df):
    # write to xlsx
    file = 'testing_io_write_read.xlsx'
    meta_df.to_excel(file)

    # read from xlsx
    import_df = IamDataFrame(file)

    # assert that `data` and `meta` tables are equal and delete file
    pd.testing.assert_frame_equal(meta_df.data, import_df.data)

    os.remove(file)


@pytest.mark.parametrize("args", [{}, dict(sheet_name='meta')])
def test_load_metadata(meta_df, args):
    file = os.path.join(TEST_DATA_DIR, 'testing_metadata.xlsx')
    meta_df.load_meta(file, **args)
    obs = meta_df.meta

    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a', 'scen_b'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_ssp_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_SSP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())


def test_load_rcp_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_RCP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())
