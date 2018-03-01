"""
====================
Plot Timeseries Data
====================

"""
import matplotlib.pyplot as plt
import pyam_analysis as iam

fname = 'msg_input.csv'

df = iam.IamDataFrame(fname)

df = (df
      .filter({'variable': 'Emissions|CO2'})
      .filter({'region': 'World'}, keep=False)
      )

print(df.head())

fig, ax = plt.subplots(figsize=(8, 8))
df.line_plot(ax=ax, color='region')
plt.show()
