"""
=====================================
Composing plots with a secondary axis
=====================================

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

import matplotlib.pyplot as plt
import pyam

df = pyam.IamDataFrame("tutorial_data.csv")
df

###############################
# Create a figure with different units on axes
# ************************************************
#
# To create a chart with multiple axes, we directly use the **matplotlib** package
# and start with a subplot consisting of a figure canvas and
# an :class:`Axes <matplotlib.axes.Axes>` object, which contains the figure elements.
#
# First, we generate a simple line chart with temperature increase in °C
# for one scenario and multiple models.
# We now tell **pyam** to specifically use the :code:`ax` instance for the plot.
#
# Then we use the **pyam** function [convert_unit()](https://pyam-iamc.readthedocs.io/en/stable/api/iamdataframe.html#pyam.IamDataFrame.convert_unit)
# to convert temperature increase in °C to temperature increase in °F.
#
# Finally, we create a second axes using :meth:`Axes.twinx() <matplotlib.axes.Axes.twinx>`
# and place a second plot on this other axes with °F as unit.

fig, ax = plt.subplots()

args = dict(
    scenario="CD-LINKS_NPi2020_1000",
    region="World",
)

temperature = "AR5 climate diagnostics|Temperature|Global Mean|MAGICC6|MED"
data_temperature = df.filter(**args, variable=temperature)
data_temperature.plot(ax=ax, title=None, legend=False)

ax2 = ax.twinx()
data_temperature_fahrenheit = data_temperature.convert_unit('°C', to='°F', factor=1.8)
data_temperature_fahrenheit.plot(ax=ax2, title=None, legend=False)

ax.legend(loc=4)
ax.set_title("Temperature")

plt.tight_layout()
plt.show()

###############################
# Create a composed figure from several plot types
# ************************************************
#
# To create a composed chart, we again use the **matplotlib** package
# and start with a subplot consisting of a figure canvas and
# an :class:`Axes <matplotlib.axes.Axes>` object, which contains the figure elements.
#
# First, we generate a simple stacked chart
# of all components of the primary energy supply for one scenario.
# We now tell **pyam** to specifically use the :code:`ax` instance for the plot.
#
# Then, we create a second axes using :meth:`Axes.twinx() <matplotlib.axes.Axes.twinx>`
# and place a second plot on this other axes.

fig, ax = plt.subplots()

args = dict(
    model="WITCH-GLOBIOM 4.4",
    scenario="CD-LINKS_NPi2020_1000",
    region="World",
)

data_energy = df.filter(**args, variable="Primary Energy|*")
data_energy.plot.stack(ax=ax, title=None, legend=False)

temperature = "AR5 climate diagnostics|Temperature|Global Mean|MAGICC6|MED"
data_temperature = df.filter(**args, variable=temperature)

ax2 = ax.twinx()
format_args = dict(color="black", linestyle="--", marker="o", label="Temperature")
data_temperature.plot(ax=ax2, legend=False, title=None, **format_args)

ax.legend(loc=4)
ax2.legend(loc=1)
ax2.set_ylim(0, 2)
ax.set_title("Primary energy mix and temperature")

plt.tight_layout()
plt.show()
