import os
import pandas as pd
import numpy as np
import pytest

from pyam import IamDataFrame, read_datapackage
from pyam.testing import assert_frame_equal

from conftest import TEST_DATA_DIR

FILTER_ARGS = dict(scenario='scen_a')


def test_io_csv(test_df):
    # write to csv
    file = 'testing_io_write_read.csv'
    test_df.to_csv(file)

    # read from csv
    import_df = IamDataFrame(file)

    # assert that `data` tables are equal and delete file
    pd.testing.assert_frame_equal(test_df.data, import_df.data)
    os.remove(file)


@pytest.mark.parametrize("meta_args", [
    [{}, {}],
    [dict(include_meta='foo'), dict(meta_sheet_name='foo')]
])
def test_io_xlsx(test_df, meta_args):
    # add column to `meta`
    test_df.set_meta(['a', 'b'], 'string')

    # write to xlsx (direct file name and ExcelWriter, see bug report #300)
    file = 'testing_io_write_read.xlsx'
    for f in [file, pd.ExcelWriter(file)]:
        test_df.to_excel(f, **meta_args[0])
        if isinstance(f, pd.ExcelWriter):
            f.close()

        # read from xlsx
        import_df = IamDataFrame(file, **meta_args[1])

        # assert that `data` and `meta` tables are equal and delete file
        pd.testing.assert_frame_equal(test_df.data, import_df.data)
        pd.testing.assert_frame_equal(test_df.meta, import_df.meta)
        os.remove(file)


@pytest.mark.parametrize("args", [{}, dict(sheet_name='meta')])
def test_load_meta(test_df, args):
    file = os.path.join(TEST_DATA_DIR, 'testing_metadata.xlsx')
    test_df.load_meta(file, **args)
    obs = test_df.meta

    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a', 'scen_b'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_ssp_database_downloaded_file(test_df_year):
    exp = test_df_year.filter(**FILTER_ARGS).as_pandas()
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_SSP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), exp)


def test_load_rcp_database_downloaded_file(test_df_year):
    exp = test_df_year.filter(**FILTER_ARGS).as_pandas()
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_RCP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), exp)


def test_io_datapackage(test_df):
    file = 'foo.zip'

    # add column to `meta`
    test_df.set_meta(['a', 'b'], 'string')

    # write to datapackage
    test_df.to_datapackage(file)

    # read from csv
    import_df = read_datapackage(file)

    # assert that IamDataFrame instances are equal and delete file
    assert_frame_equal(test_df, import_df)
    os.remove(file)
