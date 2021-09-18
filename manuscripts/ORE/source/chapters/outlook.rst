Outlook
=======

Facilitating assessments in AR6
-------------------------------

As part of the upcoming IPCC Sixth Assessment Report (AR6), pyam has facilitated
increased coordination and consistency across the analysis and data processing steps. 
In addition to being utilized by authors to generate key figures in the report
(`link to the repository <https://github.com/gidden/ar6-wg1-ch6-emissions>`_),
pyam is a critical component to the overall climate assessment pipeline
utilized by AR6 authors across Working Groups (I & III). All
scenario data is supplied in accordance with the IAMC data format to ensure
interoperability. The emissions data is then read in using pyam. Emissions
data is processed using the open-source software packages aneris
:cite:`Gidden:2018:aneris` and silicone :cite:`Lamboll:2020:Silicone` before
it is run using probabilistic reduced-complexity climate models managed through the
package OpenSCM-Runner :cite:`Nicholls:2021:openscm-runner`.

All of these programs natively use the IAMC timeseries data format and pyam serves
as the programmatic interface between the integrated-assessment scenarios and the climate model processing.
The pyam validation features allows for easily checking that the minimum set of emissions data exists for
each scenario, ensuring that no essential data is missing.
While pyam is used for much of the pre- and post-processing,
some analysis steps use pandas or scmdata :cite:`Nicholls:2021:scmdata` instead
because they are better suited to processing large volumes of probabilistic climate data.
The scripts will be released upon publication of the AR6.

Connection to other data resources
----------------------------------

As a next step for increasing the usefulness of the pyam package,
we intend to implement additional connections to data resources:
First, discussions have started with the maintainers of the
`Open Energy Platform <https://openenergy-platform.org>`_ (OEP) to develop an interface
to their database infrastructure and related tools.
Second, the just-starting Horizon 2020 project *European Climate and Energy Modelling Forum*
(ECEMF) will also rely on the pyam package and the underlying data model
to implement linkages between modelling frameworks and make scenario results available
to stakeholders and other researchers.

Community growth and package development
----------------------------------------

To make the development of an open-source, collaborative package like pyam
sustainable over an extended period of time, it is vital to have several developers
and core contributors to implement feature proposals, review pull requests and
respond to bug reports.
At the same time, there is an important role for (non-expert) users:
suggesting new features to improve the usefulness of the package,
contributing to the development of tutorials,
and answering questions from new users via the community
`Slack channel <https://pyam-iamc.slack.com>`_ and
`mailing list <https://pyam.groups.io>`_.

The just-starting Horizon 2020 project *European Climate and Energy Modelling Forum*
(`ECEMF <https://ecemf.eu>`_) will develop model linkages and tools
based on or compatible with the pyam package.
By virtue of being applied in this and several other ongoing Horizon 2020 projects
as well as the IPCC AR6 process,
we are confident that the package will attract new users and continuously evolve
to meet changing requirements for scenario analysis and data visualization.

At the same time, the solid foundation of continuous-integration workflows,
comprehensive test coverage and detailed documentation minimize the risk
of inadvertently breaking existing scripts and causing frustration amongst
the existing user base.

Facilitating best practices of scientific software development and open science
-------------------------------------------------------------------------------

This manuscript introduces the pyam package, a Python toolbox bridging the gap
between scenario processing solutions that are fully customized to specific
integrated assessment or macro-energy modelling frameworks, on the one hand,
and general-purpose data processing and visualization packages, on the other hand.

We believe that this package can enable the adoption of best practices
for scientific software development and facilitate reproducible and open science
through several avenues:
First, an intuitive interface and the many tutorials make it easy for non-expert users
to switch from analysis using Excel spreadsheets to scripted workflows.
Second, by removing clutter from scripts thanks to a well-structured and stable API,
pyam allows to write more concise workflows.
Thereby, scenario processing will become easier to understand,
which can increase the transparency and reproducibility of the scientific analysis.
Third, by implementing a generic and widely adopted data model
with interfaces to several data resources and supporting multiple file types,
the package can increase interoperability between modelling frameworks
and streamline comparison of scenario results across projects and research domains.

Last, but not least: by providing a suite of domain-relevant methods
based on a generic and versatile data model,
it is our hope that using pyam will free up time for researchers and modellers
to perform more scenario validation and analysis.
This can improve the quality and relevance of scientific insights related
to climate change mitigation pathways and the sustainable development goals.
