"""
=======================
Plot Data as a Bar Plot
=======================

"""
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# First, we generate a simple stacked bar chart
# of all components of primary energy supply for one scenario.

data = df.filter(model='WITCH-GLOBIOM 4.4', scenario='CD-LINKS_NPi2020_1000',
                 variable='Primary Energy|*', region='World')

fig, ax = plt.subplots()
data.bar_plot(ax=ax, stacked=True)
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We can flip that round for a horizontal chart.

fig, ax = plt.subplots()
data.bar_plot(ax=ax, stacked=True, orient='h')
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We don't just have to plot subcategories of variables,
# any data or meta indicators from the IamDataFrame can be used.
# Here, we show the contribution by region to total CO2 emissions.

data = (
    df.filter(model='WITCH-GLOBIOM 4.4', scenario='CD-LINKS_NPi2020_1000',
              variable='Emissions|CO2')
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots()
data.bar_plot(ax=ax, bars='region', stacked=True, cmap='tab20')
plt.show()
