import string
import numpy as np
import pandas as pd
from functools import wraps
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
def gen_frame(strdata, fdata, fast):
    return pyam.IamDataFrame(pd.concat([strdata, fdata], axis=1), fast=fast)

def profile(max=5):
    data = {'N': [], 'time': [], 'type': []}
    for N in [int(10**n) for n in np.arange(1, 8, step=0.5)]:
        print(N)
        for type in ['slow', 'fast']:
            try:
                strdata = pd.DataFrame(gen_str_iamc(N, k=5), columns=pyam.IAMC_IDX)
                fdata = pd.DataFrame(gen_float(N), columns=YEARS)
                time, df = gen_frame(strdata, fdata, fast=type == 'fast')
                print(N, type, time)
                data['N'].append(N)
                data['type'].append(type)
                data['time'].append(time)
            except:
                continue
    return pd.DataFrame.from_dict(data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = profile(max=8)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='N', y='time', hue='type', ax=ax)
    ax.set(xscale='log')
    fig.savefig('profile_init.png')
    df.to_csv('profile_init.csv')