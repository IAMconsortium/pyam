.. currentmodule:: pyam

Plotting library
================

The plotting API(s)
-------------------

There are three ways to use the |pyam| plotting library.

1. Using the plot feature as an attribute of the :class:`IamDataFrame`:

   .. code-block:: python

      IamDataFrame.plot.<kind>(**kwargs)

2. Using the plot feature as a function with a `kind` keyword argument:

   .. code-block:: python

      IamDataFrame.plot(kind='<kind>', **kwargs)

   This function defaults to the :meth:`pyam.plotting.line` type.

3. Calling any function of either the :mod:`plotting`
   or the :mod:`figures` module directly via

   .. code-block:: python

      pyam.<module>.<kind>(df, **kwargs)

   where `df` is either an :class:`IamDataFrame`
   or a suitable :class:`pandas.DataFrame`.

Check out the `Plotting Gallery`_ for examples!

.. _`Plotting Gallery` : ../gallery/index.html

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

See `this example`_ from the AR6 WG1 using |pyam| plotting via a `yaml` file.

.. _`this example`: https://github.com/gidden/ar6-wg1-ch6-emissions/blob/master/plotting.yaml

The :meth:`IamDataFrame.categorize` function also appends any style arguments
to the RunControl.

Plotting functions
------------------

.. autofunction:: pyam.plotting.line

.. autofunction:: pyam.plotting.stack

.. autofunction:: pyam.plotting.bar

.. autofunction:: pyam.plotting.box

.. autofunction:: pyam.plotting.scatter

.. autofunction:: pyam.plotting.pie

.. autofunction:: pyam.figures.sankey
