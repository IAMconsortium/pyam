"""
===================
Plot Ranges of Data
===================

"""
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# In this example, we want to highlight the range across a scenario ensemble.
# We do this by utilizing the `fill_between` argument.

data = df.filter(scenario='CD-LINKS*',
                 variable='Emissions|CO2', region='World')

fig, ax = plt.subplots()
data.line_plot(ax=ax, color='scenario', fill_between=True)
plt.show()

###############################
# The keyword argument `fill_between` can be set to true,
# or it can be provided specific arguments as a dictionary:
# in this illustration, we choose a very low transparency value.

fig, ax = plt.subplots()
data.line_plot(ax=ax, color='scenario', fill_between=dict(alpha=0.15))
plt.show()

###############################
# To further highligh the range of data, we can also add a bar showing the
# range of data in the final time period using `final_ranges`. Similar to
# `fill_between` it can either be true or have specific arguments.

fig, ax = plt.subplots()
data.line_plot(ax=ax, color='scenario', fill_between=True,
               final_ranges=dict(linewidth=5))
plt.show()
