"""
============================================
Plot Data as a Bar Plot with Net Value Lines
============================================

"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pyam

from pyam.plotting import add_net_values_to_bar_plot

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# First, we generate a simple stacked bar chart
# of the regional breakdown of CO2 emissions in one scenario.

data = (
    df.filter(model='WITCH-GLOBIOM 4.4', scenario='CD-LINKS_NPi2020_1000',
              variable='Emissions|CO2')
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots()
data.bar_plot(ax=ax, bars='region', stacked=True)
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# Sometimes stacked bar charts have negative entries.
# In that case it helps to add a line showing the net value.

fig, ax = plt.subplots()
data.bar_plot(ax=ax, bars='region', stacked=True)
add_net_values_to_bar_plot(ax, color='k')
fig.subplots_adjust(right=0.55)
plt.show()
