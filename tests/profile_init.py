import string
import numpy as np
import pandas as pd
from functools import wraps
from pathlib import Path
import time

import pyam 

YEARS = range(2010, 2101, 10)



def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return total_time, result
    return timeit_wrapper

def join(a):
    return ''.join(a)

def gen_str(N, k=1):
    return np.random.choice(list(string.ascii_lowercase),  size=(k, N,len(pyam.IAMC_IDX)))

def gen_str_iamc(N, k=1):
    return np.apply_along_axis(join, 0, gen_str(N, k))

def gen_float(N, years=YEARS):
    return np.random.choice(range(10),  size=(N,len(years), ))

@timeit
def gen_frame(data, fast):
    return pyam.IamDataFrame(data, fast=fast)

def profile(max=5):
    data = {'N': [], 'time': [], 'type': []}
    for N in [int(10**n) for n in np.arange(1, max, step=0.5)]:
        print(N)
        for type in ['slow', 'fast']:
            try:
                strdata = pd.DataFrame(gen_str_iamc(N, k=5), columns=pyam.IAMC_IDX)
                fdata = pd.DataFrame(gen_float(N), columns=YEARS)
                _data = pd.concat([strdata, fdata], axis=1)
                time, df = gen_frame(_data, fast=type == 'fast')
                print(N, type, time)
                data['N'].append(N)
                data['type'].append(type)
                data['time'].append(time)
            except:
                continue
    return pd.DataFrame.from_dict(data)

@timeit
def gen_frame_from_file(file, fast):
    return pyam.IamDataFrame(file, fast=fast)

def profile_file(fname):
    data = {'N': [], 'time': [], 'type': []}
    for type in ['slow', 'fast']:
        time, df =  gen_frame_from_file(fname, fast=type == 'fast')
        data['N'].append(len(df))
        data['type'].append(type)
        data['time'].append(time)
    return pd.DataFrame.from_dict(data)

def main():
    # requires downloading AR6 dataset and placing it in the data folder
    import matplotlib.pyplot as plt
    import seaborn as sns
    dfp = profile(max=6)
    df6 = profile_file(fname=Path('./data/AR6_Scenarios_Database_World_v1.0.csv'))
    df = pd.concat([dfp, df6]).reset_index()
    df.to_csv('profile_init.csv')
    print(df)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='N', y='time', hue='type', ax=ax)
    ax.set(xscale='log')
    fig.savefig('profile_init.png')


if __name__ == '__main__':
    main()