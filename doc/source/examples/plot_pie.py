"""
========================
Pie chart visualizations
========================

"""
# sphinx_gallery_thumbnail_number = 3

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
# A pie chart of the energy supply
# ********************************
#
# We generate a pie plot of all components of primary energy supply
# for one scenario.

data = df.filter(
    model="AIM/CGE 2.1",
    scenario="CD-LINKS_NPi",
    variable="Primary Energy|*",
    year=2050,
    region="World",
)

data.plot.pie()
plt.tight_layout()
plt.show()

###############################
# A pie chart with a legend
# *************************
#
# Sometimes a legend is preferable to labels, so we can use that instead.

data.plot.pie(labels=None, legend=True)
plt.tight_layout()
plt.show()

###############################
# A pie chart of regional contributions
# *************************************
#
# We don't just have to plot subcategories of variables,
# any data or meta indicators from the IamDataFrame can be used.
# Here, we show the contribution by region to CO2 emissions.

data = df.filter(
    model="AIM/CGE 2.1", scenario="CD-LINKS_NPi", variable="Emissions|CO2", year=2050
).filter(region="World", keep=False)
data.plot.pie(category="region", cmap="tab20")
plt.tight_layout()
plt.show()
