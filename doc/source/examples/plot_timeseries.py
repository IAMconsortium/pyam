"""
====================
Plot Timeseries Data
====================

"""
import os

import matplotlib.pyplot as plt
import pyam_analysis as iam

fname = 'msg_input.csv'

df = iam.IamDataFrame(fname)

df = (df
      .filter({'variable': 'Emissions|CO2'})
      .filter({'region': 'World'}, keep=False)
      )

print(df.head())

df.line_plot(color='region', legend=True)

plt.show()
