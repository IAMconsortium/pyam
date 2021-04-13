"""
==============
Sankey diagram
==============

"""
###############################
# Read in example data and show a summary
# ***************************************
#
# This gallery uses a small selection of the data
# compiled for the IPCC's *Special Report on Global Warming of 1.5°C* (SR15_).
# The complete scenario ensemble data is publicly available from the
# `IAMC 1.5°C Scenario Explorer and Data hosted by IIASA`_.
#
# Please read the License_ of the IAMC 1.5°C Scenario Explorer
# before using the full scenario data for scientific analyis or other work.
#
# .. _SR15: http://ipcc.ch/sr15/
#
# .. _`IAMC 1.5°C Scenario Explorer and Data hosted by IIASA` : https://data.ene.iiasa.ac.at/iamc-1.5c-explorer
#
# .. _License : https://data.ene.iiasa.ac.at/iamc-1.5c-explorer/#/license
#
# If you haven't cloned the **pyam** GitHub repository to your machine,
# you can download the data file from
# https://github.com/IAMconsortium/pyam/tree/main/doc/source/examples_source
#
# Make sure to place the data file in the same folder as this script/notebook.

import pyam
import plotly

df = pyam.IamDataFrame("sankey_data.csv")
df

###############################
# A simple Sankey diagram
# ***********************
#
# We show a Sankey diagram of a subset of the energy system
# in the 'CD-LINKS_NPi2020_1000' scenario
# implemented by the 'REMIND-MAgPIE 1.7-3.0' model.
#
# The :meth:`~pyam.figures.sankey` function
# takes a dictionary to define flows, sources and targets:
#
# .. code-block:: python
#
#     {
#         variable: (source, target),
#     }

sankey_mapping = {
    "Primary Energy|Coal": ("Coal Mining", "Coal Trade & Power Generation"),
    "Primary Energy|Gas": ("Natural Gas Extraction", "Gas Network & Power Generation"),
    "Secondary Energy|Electricity|Non-Biomass Renewables": (
        "Non-Biomass Renewables",
        "Electricity Grid",
    ),
    "Secondary Energy|Electricity|Nuclear": ("Nuclear", "Electricity Grid"),
    "Secondary Energy|Electricity|Coal": (
        "Coal Trade & Power Generation",
        "Electricity Grid",
    ),
    "Secondary Energy|Electricity|Gas": (
        "Gas Network & Power Generation",
        "Electricity Grid",
    ),
    "Final Energy|Electricity": ("Electricity Grid", "Electricity Demand"),
    "Final Energy|Solids|Coal": (
        "Coal Trade & Power Generation",
        "Non-Electricity Coal Demand",
    ),
    "Final Energy|Gases": ("Gas Network & Power Generation", "Gas Demand"),
}

fig = df.filter(year=2050).plot.sankey(mapping=sankey_mapping)
# calling `show()` is necessary to have the thumbnail in the gallery overview
plotly.io.show(fig)
