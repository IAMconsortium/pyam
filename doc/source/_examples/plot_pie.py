"""
========================
Plot Data as a Pie Chart
========================

"""
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# We generate a pie plot of all components of primary energy supply
# for one scenario.

data = df.filter(model='AIM/CGE 2.1', scenario='CD-LINKS_NPi',
                 variable='Primary Energy|*', year=2050,
                 region='World')

fig, ax = plt.subplots()
data.pie_plot(ax=ax)
fig.subplots_adjust(right=0.75, left=0.3)
plt.show()

###############################
# Sometimes a legend is preferable to labels, so we can use that instead.

fig, ax = plt.subplots()
data.pie_plot(ax=ax, labels=None, legend=True)
fig.subplots_adjust(right=0.55, left=-0.05)
plt.show()

###############################
# We don't just have to plot subcategories of variables,
# any data or meta indicators from the IamDataFrame can be used.
# Here, we show the contribution by region to CO2 emissions.

data = (df
        .filter(model='AIM/CGE 2.1', scenario='CD-LINKS_NPi',
                variable='Emissions|CO2', year=2050)
        .filter(region='World', keep=False)
        )
data.pie_plot(category='region', cmap='tab20')
plt.show()
