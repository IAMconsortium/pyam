"""
=========================
Plot Data as a Stack Plot
=========================

"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pyam

###############################
# Read in some example data

fname = 'msg_input.csv'
df = pyam.IamDataFrame(fname, encoding='ISO-8859-1')
print(df.head())

###############################
# We generated a simple stacked stack chart as below

data = df.filter({'variable': 'Emissions|CO2|*',
                  'level': 0,
                  'region': 'World'})
data.interpolate(2015)  # some values are missing

fig, ax = plt.subplots(figsize=(10, 10))
data.stack_plot(ax=ax)
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
data.stack_plot(ax=ax, stack='region', cmap='tab20')
plt.show()
