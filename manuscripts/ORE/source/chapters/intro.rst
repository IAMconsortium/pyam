Introduction
============

Towards open-source tools in energy & climate modelling
-------------------------------------------------------

Over the past years, the scientific communities for energy systems modelling and
integrated assessment of climate change mitigation pathways
have made significant strides to "#freethemodels"
:cite:`DeCarolis:2020:macroenergy,Pfenninger:2018:blackbox`.
This includes steps to release input data, assumptions, algebraic formulation,
and processing tools for scenario results under open-source licenses,
in order to facilitate transparency and reproducibility of scientific analysis.
These efforts are part of a larger push towards open science
and FAIR data management principles
(Findable, Accessible, Interoperable, Reusable :cite:`Wilkinson:2016:FAIR`)
supported by stakeholders, funding agencies and researchers themselves,
for example the `openmod initiative <https://openmod-initiative.org>`_.

Alas, the efforts to move to open-source and collaborative (scientific) software
development practices in energy systems modelling, macro-energy research
and integrated assessment have, so far, mostly focused on modelling frameworks
and input data. The processing of scenario results using a common set of tools and
methods has received much less attention.
In many cases, users are either confined to tools for processing of results
that are highly customized to a specific modelling framework,
or they have to develop their own methods and scripts using general-purposes packages.
In a Python environment, for example, users often write their own workflows
and analysis tools from scratch using `pandas <https://pandas.pydata.org>`_,
`numpy <https://numpy.org>`_,
`matplotlib <https://matplotlib.org>`_ :cite:`Hunter:2007:matplotlib`
and `seaborn <https://seaborn.pydata.org>`_ :cite:`Waskom:2021:seaborn`.

The vision of **pyam** is to bridge that gap: to provide a suite of features and methods
that are applicable for scenario processing, analysis and visualization
irrespective of the modelling framework.
At the same time, the package should be sufficiently specific
for energy systems modelling
as well as integrated assessment of climate change and sustainable development
to allow sensible defaults and remove as much clutter as possible from
scenario processing workflows or analysis scripts.

An overview of existing packages and tools
------------------------------------------

Several open-source packages and tools exist in between the general-purpose packages
for data analysis and plotting, on the one hand, and dedicated data processing solutions
specifically built around a specific modelling framework, on the other,
see :numref:`overview`.
These packages are compatible with a variety of data formats
commonly used in energy systems modelling and integrated assessment.

.. _overview:

.. figure:: ../figure/overview.png
   :align: center

   Overview of packages & tools for energy system & integrated assessment modelling
   (see the Appendix for a full list of references and links cited in this figure)

These packages can be grouped into four categories; we provide examples
in each category for illustrative purposes:

1. *Data processing, computation and validation of input data and scenario results*:

   The R package `madrat <https://github.com/pik-piam/madrat>`_ provides a framework
   for improving reproducibility and transparency in data processing
   :cite:`Dietrich:2021:madrat`.

   In comparison, the R package `iamc <https://github.com/iamconsortium/iamc>`_ is a collection
   of functions for data analysis and diagnostics of scenario results in the IAMC format
   (see the following section on data models for more information).

   The Python package `genno <https://genno.readthedocs.io>`_ supports describing and
   executing complex calculations on labelled, multi-dimensional data; it was developed
   as a generalization of data processing in the context of integrated assessment and transport modelling.

2. *Visualization of scenario results in a domain-specific format*:

   The R package `mipplot <https://github.com/UTokyo-mip/mipplot>`_ generates plots
   from climate mitigation scenarios :cite:`Yiyi:2021:mipplot`.
   It is also based on the IAMC format.

3. *Reference data management for model input & calibration*:

   The Public Utility Data Liberation (`PUDL <https://catalyst.coop/pudl>`_) project
   takes publicly available information and makes it usable by cleaning, standardizing,
   and cross-linking utility data from different sources in a single database.

   In a similar effort, `PowerGenome <https://github.com/PowerGenome/PowerGenome>`_
   compiles different data sources into a single database.

   The `PowerSystems.jl <https://github.com/NREL-SIIP/PowerSystems.jl>`_ package
   provides a rigorous data model to enable power systems analysis and modelling
   across several input formats.

4. *Comprehensive database solutions for management of scenario input data and results*:

   The `Open Energy Platform <https://openenergy-platform.org>`_ aims to ensure quality,
   transparency and reproducibility in energy system research. It is a collaborative
   community effort to develop various tools and information that help working
   with energy-related data.

   The `Spine Toolbox <https://spine-toolbox.readthedocs.io>`_ is a modular and
   adaptable end-to-end energy modelling ecosystem to enable open, practical, flexible
   and realistic planning of European energy grids.

The pyam package covers both the data processing and validation aspects (category 1)
as well as a suite of plotting features (category 2).
It also provides direct interfaces to reference data sources (category 3)
and can be integrated with existing community database solutions (category 4).
Due to this wide scope, it is a novel and - we hope - useful addition
to the suite of tools used by the energy systems and integrated-assessment communities.

A Python package for scenario analysis & visualization
------------------------------------------------------

The pyam package grew out of complementary efforts in the Horizon 2020 project
`CRESCENDO <https://www.crescendoproject.eu>`_ and the analysis of integrated-assessment scenarios
supporting the IPCC's *Special Report on Global Warming of 1.5°C*.
Ref :cite:`Gidden:2019:pyam` describes an earlier version of its features and capabilities.
After three years of development, we believe that the package has now reached
a reasonable level of maturity to be useful to a wider audience -
in scientific-software jargon, it is ready for **release 1.0**.

The aim of the package is not to provide complex new methodologies
or sophisticated plotting features. Instead, the aim is to provide a toolbox
for many small operations and processing steps that a researcher or analyst frequently
needs when working with numerical scenarios of climate change mitigation
and the energy system transition:
aggregation & downscaling, unit conversion, validation,
and a simple plotting library to quickly get an intuition of the scenario data.

This manuscript describes the design principles of the package
and the types of data that can be handled.
We present a number of features and recent applications
to illustrate the usefulness of pyam.
In the last section, we identify several forthcoming uses cases
and planned developments.
