"""
==============
Boxplot charts
==============

"""
# sphinx_gallery_thumbnail_number = 2

###############################
# Read in tutorial data and show a summary
# ****************************************
#
# This gallery uses the scenario data from the first-steps tutorial.
#
# If you haven't cloned the **pyam** GitHub repository to your machine,
# you can download the file from
# https://github.com/IAMconsortium/pyam/tree/main/doc/source/tutorials.
#
# Make sure to place the data file in the same folder as this script/notebook.

import matplotlib.pyplot as plt
import pyam

df = pyam.IamDataFrame("tutorial_data.csv")
df

###############################
# A boxplot of CO emissions
# *************************
#
# We generate a simple boxplot of CO2 emissions
# across one scenario implemented by a range of models.

data = df.filter(
    scenario="CD-LINKS_NPi2020_1000", variable="Emissions|CO2", region="World"
)

data.plot.box(x="year")
plt.tight_layout()
plt.show()

###############################
# A grouped boxplot
# *****************
#
# We can add sub-groupings of the data using the keyword argument `by`.

data = df.filter(
    scenario="CD-LINKS_NPi2020_1000",
    variable="Emissions|CO2",
    year=[2010, 2020, 2030, 2050, 2100],
).filter(region="World", keep=False)

data.plot.box(x="year", by="region", legend=True)

# We can use matplotlib arguments to make the figure more appealing.
plt.legend(loc=1)
plt.tight_layout()
plt.show()
