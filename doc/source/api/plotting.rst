.. currentmodule:: pyam

Plotting library
================

The RunControl configuration
----------------------------

The |pyam| plotting library provides a thin :class:`RunControl` wrapper
around a Python :class:`dictionary` for plotting-style defaults,
like setting colors or linestyles for certain model or scenario names.

.. autofunction:: pyam.run_control

Input can be provided as nested dictionaries of the structure
:code:`type > dimension > name > value`, e.g.,

.. code-block:: python

    pyam.run_control().update(
        {'color': {'scenario': {'test_scenario': 'black'}}}
    )

or as the path to a yaml file with a similar structure:

.. code-block:: python

    pyam.run_control().update(<file.yaml>)

See `this example`_ from the AR6 WG1 work using |pyam| plotting.

.. _`this example`: https://github.com/gidden/ar6-wg1-ch6-emissions/blob/master/plotting.yaml

The :meth:`IamDataFrame.categorize` function also appends any style arguments
to the RunControl.

Plotting functions
------------------

.. autofunction:: pyam.plotting.stackplot

.. autofunction:: pyam.figures.sankey
