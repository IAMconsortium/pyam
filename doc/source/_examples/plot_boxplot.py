"""
=========================
Plot Data as a boxplot
=========================

"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pyam

###############################
# Read in some example data

fname = 'data.csv'
df = pyam.IamDataFrame(fname, encoding='ISO-8859-1')
print(df.head())

###############################
# We generate a simple boxplot as below

data = df.filter(variable='Emissions|CO2|*',
                 level=0,
                 region='World')

fig, ax = plt.subplots(figsize=(10, 10))
data.boxplot(x='year', ax=ax,)
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We don't just have to plot variables, any data or metadata associated with the
# IamDataFrame can be used. Use the by parameter to add the 3rd dimension, for example for grouping boxplots by  category.

data = (df
        .filter(variable='Emissions|CO2'))
fig, ax = plt.subplots(figsize=(10, 10))
data.boxplot(x='year', by='region', legend=True, ax=ax)
plt.show()
