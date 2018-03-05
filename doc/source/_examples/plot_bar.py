"""
=======================
Plot Data as a Bar Plot
=======================

"""
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import pyam

###############################
# Read in some example data

fname = 'msg_input.csv'
df = pyam.IamDataFrame(fname, encoding='ISO-8859-1')
print(df.head())

###############################
# We generated a simple stacked bar chart as below

data = df.filter({'variable': 'Emissions|CO2|*',
                  'level': 0,
                  'region': 'World'})

fig, ax = plt.subplots(figsize=(10, 10))
data.bar_plot(ax=ax, stacked=True)
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We can flip that round for a horizontal chart

fig, ax = plt.subplots(figsize=(10, 10))
data.bar_plot(ax=ax, stacked=True, orient='h')
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We don't just have to plot variables, any data or metadata associated with the
# IamDataFrame can be used.

data = (df
        .filter({'variable': 'Emissions|CO2'})
        .filter({'region': 'World'}, keep=False)
        )
fig, ax = plt.subplots(figsize=(10, 10))
data.bar_plot(ax=ax, bars='region', stacked=True, cmap='tab20')
plt.show()
