import os
import pandas as pd
import numpy as np

from pyam import IamDataFrame

from conftest import TEST_DATA_DIR


def test_read_csv():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'testing_data_2.csv'))
    assert list(df.variables()) == ['Primary Energy']


def test_load_metadata(meta_df):
    meta_df.load_metadata(os.path.join(
        TEST_DATA_DIR, 'testing_metadata.xlsx'), sheet_name='meta')
    obs = meta_df.meta

    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a', 'scen_b'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_SSP_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_SSP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())


def test_load_RCP_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_RCP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())


def test_to_excel(test_df):
    fname = 'foo_testing.xlsx'
    test_df.to_excel(fname)
    assert os.path.exists(fname)
    os.remove(fname)


def test_to_csv(test_df):
    fname = 'foo_testing.csv'
    test_df.to_csv(fname)
    assert os.path.exists(fname)
    os.remove(fname)
