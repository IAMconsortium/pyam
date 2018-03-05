"""
====================
Plot Timeseries Data
====================

"""
import matplotlib.pyplot as plt
import pyam

fname = 'msg_input.csv'

df = pyam.IamDataFrame(fname, encoding='ISO-8859-1')

df = (df
      .filter({'variable': 'Emissions|CO2'})
      .filter({'region': 'World'}, keep=False)
      )

print(df.head())

fig, ax = plt.subplots(figsize=(8, 8))
df.line_plot(ax=ax, color='region')
plt.show()
