"""
====================
Plot Timeseries Data
====================

"""
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary
# *****************************************************************
#
# If you haven't cloned the **pyam** GitHub repository to your machine,
# you can download the file
# from https://github.com/IAMconsortium/pyam/tree/master/doc/source/tutorials
#
# Make sure to place the file in the same folder as this script/notebook.

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# A simple line chart
# *******************
#
# We show a simple line chart of the regional components
# of CO2 emissions for one scenario.

data = (
    df.filter(model='REMIND-MAgPIE 1.7-3.0', scenario='CD-LINKS_INDCi',
              variable='Emissions|CO2')
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots()
data.line_plot(ax=ax, color='region', title='CO2 emissions by region')
plt.show()