"""
======================
Timeseries data charts
======================

"""
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

import pyam

df = pyam.IamDataFrame("tutorial_data.csv")
df

###############################
# A simple line chart
# *******************
#
# We show a simple line chart of the regional components
# of CO2 emissions for one scenario.
#
# Then, also show the data as a wide IAMC-style dataframe.

model, scenario = "REMIND-MAgPIE 1.7-3.0", "CD-LINKS_INDCi"

data = df.filter(model=model, scenario=scenario, variable="Emissions|CO2").filter(
    region="World", keep=False
)

data.plot(color="region", title="CO2 emissions by region")
data.timeseries()
