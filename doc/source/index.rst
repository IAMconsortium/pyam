**pyam**: analysis and visualization |br| of integrated-assessment & macro-energy scenarios
===========================================================================================

Release v\ |version|.

|license| |pypi| |conda| |latest|

|black| |python| |pytest| |rtd| |codecov|

|doi| |ore| |joss| |groupsio| |slack|

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-black
   :target: https://github.com/IAMconsortium/pyam/blob/main/LICENSE

.. |pypi| image:: https://img.shields.io/pypi/v/pyam-iamc.svg
   :target: https://pypi.python.org/pypi/pyam-iamc/

.. |conda| image:: https://anaconda.org/conda-forge/pyam/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |latest| image:: https://anaconda.org/conda-forge/pyam/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/pyam

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |python| image:: https://img.shields.io/badge/python-3.7_|_3.8_|_3.9-blue?logo=python&logoColor=white
   :target: https://github.com/IAMconsortium/pyam

.. |pytest| image:: https://github.com/IAMconsortium/pyam/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/IAMconsortium/pyam/actions/workflows/pytest.yml

.. |rtd| image:: https://readthedocs.org/projects/pyam-iamc/badge/?version=latest
   :target: https://pyam-iamc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/IAMconsortium/pyam/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/IAMconsortium/pyam

.. |doi| image:: https://zenodo.org/badge/113359260.svg
   :target: https://doi.org/10.5281/zenodo.1470400

.. |ore| image:: https://img.shields.io/badge/ORE-10.12688/openreseurope.13633.2-blue
   :target: https://doi.org/10.12688/openreseurope.13633.2

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.01095/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.01095

.. |groupsio| image:: https://img.shields.io/badge/mail-groups.io-blue
   :target: https://pyam.groups.io/g/forum

.. |slack| image:: https://img.shields.io/badge/chat-Slack-orange.svg
   :target: https://pyam-iamc.slack.com

Overview
--------

The open-source Python package |pyam| :cite:`Gidden:2019:pyam,Huppmann:2021:pyam-v1.0`
provides a suite of tools and functions for analyzing and visualizing
input data (i.e., assumptions/parametrization) 
and results (model output) of integrated-assessment models,
macro-energy scenarios, energy systems analysis, and sectoral studies.

The source code is available on `Github <https://github.com/IAMconsortium/pyam>`_.

Key features
~~~~~~~~~~~~

 - Simple analysis of scenario timeseries data with an interface similar in feel & style
   |br| to the widely used :class:`pandas.DataFrame`
 - Advanced visualization and plotting functions (see the `gallery <gallery/index.html>`_)
 - Features for scripted validation & processing of scenario data and results

Timeseries types & data formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yearly data
^^^^^^^^^^^

The |pyam| package was initially developed to work with the *IAMC template*,
a timeseries format for *yearly data* developed and used by the
`Integrated Assessment Modeling Consortium <https://www.iamconsortium.org>`_ (IAMC).

.. figure:: _static/iamc_template.png

   Illustrative example of IAMC-format timeseries data |br|
   via the `IAMC 1.5°C Scenario Explorer`_ (:cite:`Huppmann:2019:scenario-data`)

.. _`IAMC 1.5°C Scenario Explorer`: https://data.ene.iiasa.ac.at/iamc-1.5c-explorer

Subannual time resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

The package also supports timeseries data with a *sub-annual time resolution*:
 - Continuous-time data using the Python `datetime format <https://docs.python.org/3/library/datetime.html>`_
 - "Representative timeslices" (e.g., "winter-night", "summer-day") |br|
   using the pyam *extra-columns* feature

Please read the `Data Model <data.html>`_ section for more information
or look at the `data-table tutorial <tutorials/data_table_formats.ipynb>`_
to see how to cast from a variety of timeseries formats to an :class:`IamDataFrame`.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   install
   authors
   contributing
   data
   tutorials
   gallery/index
   api

.. toctree::
   :maxdepth: 1

   R_tutorials/pyam_R_tutorial

.. toctree::
   :maxdepth: 1

   references

Scientific publications
-----------------------

The following manuscripts describe the package at specific stages of development.

The source documents are available in the manuscripts_ folder of the GitHub repository.

.. _manuscripts: https://github.com/IAMconsortium/pyam/tree/main/manuscripts

Release v1.0 (June 2021)
~~~~~~~~~~~~~~~~~~~~~~~~

Published to mark the first major release of the |pyam| package.

.. highlights::

  | Daniel Huppmann, Matthew Gidden, Zebedee Nicholls, Jonas Hörsch, Robin Lamboll,
    Paul Natsuo Kishimoto, Thorsten Burandt, Oliver Fricko, Edward Byers, Jarmo Kikstra,
    Maarten Brinkerink, Maik Budzinski, Florian Maczek, Sebastian Zwickl-Bernhard,
    Lara Welder, Erik Francisco Álvarez Quispe, and Christopher J. Smith.
  | *pyam: Analysis and visualisation of integrated assessment and macro-energy scenarios.*
  | **Open Research Europe**, 2021.
    doi: `10.12688/openreseurope.13633.2 <https://doi.org/10.12688/openreseurope.13633.2>`_

Release v0.1.2 (November 2018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Published following the successful application of |pyam|
in the IPCC SR15 and the Horizon 2020 CRESCENDO project.

.. highlights::

  | Matthew Gidden and Daniel Huppmann.
    *pyam: a Python package for the analysis and visualization of models
    of the interaction of climate, human, and environmental systems.*
  | **Journal of Open Source Software (JOSS)**, 4(33):1095, 2019.
    doi: `10.21105/joss.01095 <https://doi.org/10.21105/joss.01095>`_

Copyright & License
-------------------

The development of the |pyam| package was started at the IIASA Energy Program,
with contributions from a number of `individuals & institutions`_ over the years.

The package is available under the open-source `Apache License`_.
Refer to the `NOTICE`_ in the GitHub repository for more information.

.. _individuals & institutions: authors.html

.. _Apache License: http://www.apache.org/licenses/LICENSE-2.0.html

.. _NOTICE: https://github.com/IAMconsortium/pyam/blob/master/NOTICE.md
