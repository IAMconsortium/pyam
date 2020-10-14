"""
====================
Plot Timeseries Data
====================

"""
import matplotlib.pyplot as plt
import pyam

###############################
# Read in the data from the first-steps tutorial and show a summary

df = pyam.IamDataFrame('tutorial_data.csv')
df

###############################
# We show a simple line chart of the regional components
# of CO2 emissions for one scenario.

data = (
    df.filter(model='REMIND-MAgPIE 1.7-3.0', scenario='CD-LINKS_INDCi',
              variable='Emissions|CO2')
    .filter(region='World', keep=False)
)

fig, ax = plt.subplots(figsize=(8, 8))
data.line_plot(ax=ax, color='region')
plt.show()