"""
==================
Plot Regional Data
==================

"""
import matplotlib.pyplot as plt
import pyam_analysis as iam

fname = 'msg_input.csv'

df = iam.IamDataFrame(fname)

df = (df
      .filter({'variable': 'Emissions|CO2', 'year': 2050})
      .filter({'region': 'World'}, keep=False)
      .map_regions('iso')
      )

print(df.head())

df.region_plot()

plt.show()
