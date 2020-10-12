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
# IamDataFrame can be used.
# Use the 'by' parameter to add the 3rd dimension. In example below we us 'region' column
# but this can similarly be a column from the metadata

data = df.filter(variable='Emissions|CO2',
                 year=[2010,2050,2100])
data = data.filter(region='World', keep=False)
                # region='World')

fig, ax = plt.subplots(figsize=(10, 10))
data.boxplot(x='year', by='region', legend=True, ax=ax)

# Minor adjustments to plot appearances
ax.hlines(0,-1,5)
ax.set_xlim([-0.5,2.5])
plt.legend(loc=1)
plt.show()
