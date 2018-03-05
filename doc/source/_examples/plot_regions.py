"""
==================
Plot Regional Data
==================

"""
import matplotlib.pyplot as plt
import pyam

fname = 'data.csv'

df = pyam.IamDataFrame(fname, encoding='ISO-8859-1')

df = (df
      .filter({'variable': 'Emissions|CO2', 'year': 2050})
      .filter({'region': 'World'}, keep=False)
      .map_regions('iso', region_col='R5_region')
      )

print(df.head())

df.region_plot()

plt.show()
