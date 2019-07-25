import pyam
import pandas as pd
import numpy as np
import math


input_filename = 'D:/checks/pyam_aggregation/test_pyam2_w_avg.xlsx'
r11_regs = ['R11_AFR', 'R11_CPA', 'R11_EEU', 'R11_FSU', 'R11_LAM',
            'R11_MEA',
            'R11_NAM', 'R11_PAO', 'R11_PAS', 'R11_SAS', 'R11_WEU']
r5_asiaregs = ['R11_CPA', 'R11_SAS']
r5_oecdregs = ['R11_NAM', 'R11_PAO', 'R11_WEU']


def read_file(filename):
    df = pd.read_excel(filename, None)
    pdf = pd.concat(
           [df.get(sheet) for sheet in df.keys()
               if sheet.startswith('data')])
    return pyam.IamDataFrame(pdf)


# idf = read_file(input_filename)

TEST_DF = pd.DataFrame([
   ['model_a', 'scen_a', 'region_a', 'Primary Energy', 'EJ/y', 1, 6.],
   ['model_a', 'scen_a', 'region_b', 'Primary Energy', 'EJ/y', 2, 12.],
   ['model_a', 'scen_a', 'region_a', 'Price', 'EJ/y', 10, 5.],
   ['model_a', 'scen_a', 'region_b', 'Price', 'EJ/y', 4, 5.],
],
   columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)

TEST2_DF = pd.DataFrame([
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_AFR', 'Final Energy', 'EJ/yr', 11.2380033420259, 16.7722439520351, 20.9155319997591, 20.3910504937534, 23.4185746452232, 24.0908731424785, 27.0493304128386, 32.2727626861068, 34.4281582082343, 38.2050348167628],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_CPA', 'Final Energy', 'EJ/yr', 47.9124936604171, 66.6579577931997, 84.0844884583706, 55.9888701426051, 44.8350885136131, 39.0925974139815, 36.0396972011147, 31.3557402323414, 29.0875240102131, 27.2466182997784],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_EEU', 'Final Energy', 'EJ/yr', 8.61383728902148, 8.62677602739205, 8.88929320982673, 6.67036072882424, 5.14058641516896, 4.43773342682888, 4.38053455207121, 4.35167297683965, 4.32940759915981, 4.32591249418421],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_FSU', 'Final Energy', 'EJ/yr', 24.8218794833949, 26.0117231186237, 29.5056956077214, 20.8171191842234, 14.8611651695844, 11.0083031614005, 10.7887365968302, 11.1427037110387, 10.6665971048567, 10.5965475203662],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_LAM', 'Final Energy', 'EJ/yr', 20.6034001729273, 24.2801187303854, 29.4510402854795, 21.3887928403215, 16.797723045125, 15.0464055088463, 14.3039947497728, 13.0923173686848, 13.5317304999339, 12.3247920999932],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_MEA', 'Final Energy', 'EJ/yr', 17.6692349091845, 22.8660921579833, 31.3368735445214, 22.7840601172955, 19.8833968565346, 18.5382840820009, 17.7419743035081, 16.612215925868, 16.3560753534791, 15.7974473016165],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_NAM', 'Final Energy', 'EJ/yr', 74.3133759498589, 71.1440696564263, 73.2123668968657, 51.8192941127069, 41.6604281796529, 35.5732797220564, 33.7619462334469, 31.6034887575803, 30.6425152159152, 29.0129875631882],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_PAO', 'Final Energy', 'EJ/yr', 18.1799955447101, 17.7202439767767, 19.0874694704398, 12.4194880657423, 8.75714708067329, 7.07273272828371, 7.05079623418781, 6.65255315781753, 6.48684902212793, 6.32577983906228],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_PAS', 'Final Energy', 'EJ/yr', 21.3019383196267, 25.2299578431331, 31.5171853264364, 22.763028701824, 19.4980883252468, 16.992054820497, 16.7015531427621, 16.1478310186695, 15.7542275853275, 15.4498875223193],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_SAS', 'Final Energy', 'EJ/yr', 21.3001191550543, 25.9337544852629, 34.5470431678644, 31.4054132169558, 35.6888475648041, 38.9139258685094, 38.6849837039596, 38.3718698859376, 39.3305177198836, 40.2546950027518],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_WEU', 'Final Energy', 'EJ/yr', 50.8013437287921, 49.8550088199168, 46.6378487107687, 33.5761092597795, 25.8833399221577, 24.193525349822, 24.2792884122779, 24.4413643950968, 24.6633064652, 24.7546700707156],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'World', 'Final Energy', 'EJ/yr', 323.041543651266, 361.84435263934, 417.618727775466, 309.320632558154, 266.308706253415, 245.417677261978, 241.069283738894, 236.456643364154, 236.034369942183, 234.973932893226],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_AFR', 'Population', 'million', 718.69, 811.47, 1021.42, 1247.97, 1476.86, 1694.07, 1881.76, 2164.08, 2252.76, 2307.93],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_CPA', 'Population', 'million', 1443.45, 1459.88, 1509.71, 1519.41, 1483, 1407.57, 1305.62, 1081.7, 974.59, 882.24],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_EEU', 'Population', 'million', 125.73, 125.43, 124.47, 122.63, 119.49, 115.91, 111.72, 99.9, 94.14, 88.88],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_FSU', 'Population', 'million', 277.85, 278.93, 281.41, 280.77, 279.33, 277.28, 273.32, 259.98, 251.08, 241.57],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_LAM', 'Population', 'million', 551.95, 584.08, 642.88, 690.44, 723.21, 740.67, 742.69, 717.22, 694.53, 669.25],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_MEA', 'Population', 'million', 382.15, 424.41, 507.06, 581.24, 647.83, 704.39, 745.97, 784.91, 786.1, 779.64],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_NAM', 'Population', 'million', 333.05, 348.15, 377.16, 405.89, 431.02, 452.86, 474.07, 506.03, 513.13, 514.79],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_PAO', 'Population', 'million', 150.93, 153.17, 155.65, 155.25, 153.25, 150.58, 147.53, 137.72, 131.28, 124.03],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_PAS', 'Population', 'million', 535.65, 565.63, 618.82, 659.96, 685.3, 693.2, 687.77, 651.9, 624.88, 594.85],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_SAS', 'Population', 'million', 1514.97, 1630.17, 1861.68, 2067.23, 2240.3, 2373.34, 2448.98, 2440.76, 2375.39, 2285.06],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'R11_WEU', 'Population', 'million', 468.71, 486.07, 510.99, 531.2, 547.53, 559.24, 565.27, 563.06, 556.07, 544.18],
['MESSAGEix-GLOBIOM 1.0', 'LowEnergyDemand', 'World', 'Population', 'million', 6503.13, 6867.39, 7611.25, 8261.99, 8787.12, 9169.11, 9384.7, 9407.26, 9253.95, 9032.42] 
],
   columns=['model', 'scenario', 'region', 'variable', 'unit', 
            2005, 2010, 2020, 2030, 2040, 2050, 2060, 2080, 2090, 2100],
 )


def test_weighted_average_region_basic():
    df = pyam.IamDataFrame(TEST_DF)
    df.weighted_average_region('Price', append=True, weight='Primary Energy')
    #
    assert list(df.filter(region='World').data.value) == [6, 5]


def test_aggregate_region_min(caplog):
    df = pyam.IamDataFrame(TEST_DF)
    caplog.clear()
    df.aggregate_region('Price', append=True, method='min')
    assert list(df.filter(region='World').data.value) == [4, 5]
    #


def test_3(caplog):
    df = pyam.IamDataFrame(TEST_DF)
    caplog.clear()
    df.aggregate_region('Final Energy', region='R5_OECD.avg',
                        subregions=[], append=True, method='avg')
    assert 'cannot aggregate variable' in caplog.text
    # pd.testing.assert_frame_equal(TEST_DF, df.data)


def test_5(caplog):
    """ expect logging.error as there's no "Final Energy' Data"""
    """ https://doc.pytest.org/en/latest/reference.html """
    df = pyam.IamDataFrame(TEST_DF)
    df.aggregate_region('Final Energy', region='R5_OECD-w.avg',
                        subregions=[], append=True, method='ww.avg')
    for record in caplog.records:
        # assert record.levelname != 'CRITICAL'
        assert record.levelname == 'INFO'  # level == logging.level.ERROR
    assert 'cannot aggregate variable' in caplog.text
    assert ' because it does not exist in any subregion' in caplog.text
    # assert caplog.records[0].levelname == 'ERROR' # actually INFO
    # assert len(idf.data.columns) == 7


def test_df2_sum(caplog):
    df1 = pyam.IamDataFrame(TEST2_DF)
    df1.aggregate_region('Final Energy', region='R1_world',
                         subregions=r11_regs, append=True)
    df2 = pyam.IamDataFrame(TEST2_DF)
    df2.aggregate_region('Final Energy', region='R1_world', method='sum',
                         subregions=r11_regs, append=True)
    pd.testing.assert_frame_equal(df1.data, df2.data)
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [
                316.755621555013, 355.097946561135, 409.184836678054,
                300.023586864032, 256.424385717784, 234.959715224705,
                230.78283554277, 226.044520115981, 225.276908784331,
                224.294372530738])


def test_df2_avg(caplog):
    df1 = pyam.IamDataFrame(TEST2_DF)
    df1.aggregate_region('Final Energy', region='R1_world', method=np.mean,
                         subregions=r11_regs, append=True)
    df2 = pyam.IamDataFrame(TEST2_DF)
    df2.aggregate_region('Final Energy', region='R1_world', method='avg',
                         subregions=r11_regs, append=True)
    pd.testing.assert_frame_equal(df1.data, df2.data)
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [
                 28.7959655959103, 32.2816315055577, 37.1986215161867,
                 27.2748715330938, 23.3113077925258, 21.3599741113368,
                 20.9802577766154, 20.5495018287256, 20.4797189803937,
                 20.3903975027944])


def test_df2_single_region_agg(caplog):
    for m in ['min', 'max', 'sum', 'avg']:
        df = pyam.IamDataFrame(TEST2_DF)
        df.aggregate_region('Final Energy', region='R1_WEU', method=m,
                            subregions=['R11_WEU'], append=True)
        np.testing.assert_allclose(
            list(df.filter(region='R1_WEU').data.value),
            list(df.filter(region='R11_WEU',
                           variable='Final Energy').data.value))
        assert list(df.filter(region='R1_WEU').data.value) == \
            list(df.filter(region='R11_WEU',
                           variable='Final Energy').data.dropna()['value'])


def test_df2_single_region_w_agg_with_None(caplog):
    for m in ['min', 'max', 'sum', 'avg']:
        df = pyam.IamDataFrame(TEST2_DF)
        # df.data.loc[df.data['region']='R11_WEU', 2005] = None
        df.data.loc[df.data['year'] == 2005, 'value'] = None
        df.weighted_average_region('Final Energy', region='R1_WEU',
                                   subregions=['R11_WEU'],
                                   append=True, weight='Population')
        exp_l = [v for v in list(df.filter(region='R11_WEU',
                 variable='Final Energy').data.value) if not math.isnan(v)]
        obj_l = [v for v in list(df.filter(region='R1_WEU').data.value)
                 if not math.isnan(v)]
        np.testing.assert_allclose(obj_l, exp_l)


def test_df2_single_region_agg_with_None(caplog):
    for m in ['min', 'max', 'sum', 'avg']:
        df = pyam.IamDataFrame(TEST2_DF)
        # df.data.loc[df.data['region']='R11_WEU', 2005] = None
        df.data.loc[df.data['year'] == 2005, 'value'] = None
        df.aggregate_region('Final Energy', region='R1_WEU', method=m,
                            subregions=['R11_WEU'], append=True)
        exp_l = [v for v in list(df.filter(region='R11_WEU',
                 variable='Final Energy').data.value) if not math.isnan(v)]
        obj_l = [v for v in list(df.filter(region='R1_WEU').data.value)
                 if not math.isnan(v)]
        np.testing.assert_allclose(obj_l, exp_l)
        # assert list(df.filter(region='R1_WEU').data.value) == \
        #    list(df.filter(region='R11_WEU',
        #                   variable='Final Energy').data.value)
