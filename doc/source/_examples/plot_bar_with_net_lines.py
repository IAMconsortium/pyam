"""
======================================
Stacked bar plots with net-value lines
======================================

"""
# sphinx_gallery_thumbnail_number = 2

###############################
# Read in tutorial data and show a summary
# ****************************************
#
# This gallery uses the scenario data from the first-steps tutorial.
#
# If you haven't cloned the **pyam** GitHub repository to your machine,
# you can download the file
# from https://github.com/IAMconsortium/pyam/tree/master/doc/source/tutorials.
#
# Make sure to place the file in the same folder as this script/notebook.

import matplotlib.pyplot as plt
import pyam
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

from pyam.plotting import add_net_values_to_bar_plot

fig, ax = plt.subplots()
data.bar_plot(ax=ax, bars='region', stacked=True)
add_net_values_to_bar_plot(ax, color='k')
fig.subplots_adjust(right=0.55)
plt.show()
