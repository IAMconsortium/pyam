.. currentmodule:: pyam.iiasa

Databases hosted by IIASA
=========================

The |pyam| package allows to directly query the scenario databases hosted by the
IIASA Energy, Climate and Environment program (ECE), commonly known as
the *Scenario Explorer* infrastructure. It is developed and maintained
by the ECE `Scenario Services and Scientific Software team`_.

.. _`Scenario Services and Scientific Software team` : https://software.ece.iiasa.ac.at

You do not have to provide username/password credentials to connect to any public
database instance using |pyam|. However, to connect to project-internal databases,
you have to create an account at the IIASA-ECE *Manager Service*
(https://manager.ece.iiasa.ac.at). Please contact the respective project coordinator
for permission to access a project-internal database.

To store the credentials on your machine so that |pyam| can use it to query a database,
we depend on the Python package |ixmp4|. You only have to do this once
(unless you change your password).

The credentials will be valid for connecting to *Scenario Apps* based on |ixmp4|
as well as for (legacy) *Scenario Explorer* database backends (see below).

In a console, run the following:

.. code-block:: console

    ixmp4 login <username>

You will be prompted to enter your password.

.. warning::

    Your username and password will be saved locally in plain-text for future use!

*Scenario Apps* instances
-------------------------

Coming soon...

*Scenario Explorer* instances
-----------------------------

The *Scenario Explorer* infrastructure developed by the Scenario Services and Scientific
Software team was developed and used for projects from 2018 until 2023.

See `this tutorial <../tutorials/iiasa.html>`_ for more information.

.. autoclass:: Connection
   :members:

.. autofunction:: read_iiasa
   :noindex:

.. autofunction:: lazy_read_iiasa
   :noindex:
