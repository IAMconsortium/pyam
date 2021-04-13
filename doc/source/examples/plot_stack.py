"""
===================
Stacked line charts
===================

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

##############################
# First, we generate a simple stacked line chart
# of all components of primary energy supply for one scenario.

model, scenario = "IMAGE 3.0.1", "CD-LINKS_NPi2020_400"

data = df.filter(
    model=model, scenario=scenario, variable="Primary Energy|*", region="World"
)

data.plot.stack(title=scenario)
plt.legend(loc=1)
plt.tight_layout()
plt.show()

###############################
# We don't just have to plot subcategories of variables,
# any data dimension or meta indicators from the IamDataFrame can be used.
# Here, we show the contribution by region to total CO2 emissions.

data = df.filter(model=model, scenario=scenario, variable="Emissions|CO2").filter(
    region="World", keep=False
)

data.plot.stack(stack="region", cmap="tab20", title=scenario, total=True)
plt.legend(loc=1)
plt.tight_layout()
plt.show()
