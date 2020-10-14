"""
=========================
Plot Data as a Stack Plot
=========================

"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

##############################
# First, we generate a simple stacked line chart
# of all components of primary energy supply for one scenario.

data = df.filter(model='IMAGE 3.0.1', scenario='CD-LINKS_NPi2020_400',
                 variable='Primary Energy|*', region='World')

fig, ax = plt.subplots()
data.stack_plot(ax=ax)
fig.subplots_adjust(right=0.55)
plt.show()

###############################
# We don't just have to plot subcategories of variables,
# any data or meta indicators from the IamDataFrame can be used.
# Here, we show the contribution by region to total CO2 emissions.

data = (
    df.filter(model='IMAGE 3.0.1', scenario='CD-LINKS_NPi2020_400',
              variable='Emissions|CO2')
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots()
data.stack_plot(ax=ax, stack='region', cmap='tab20', total=True)
plt.show()
