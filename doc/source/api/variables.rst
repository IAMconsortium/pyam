.. currentmodule:: pyam

Variables utilities
===================

The **variable** dimension of the |pyam| data format implements
implements a "semi-hierarchical" structure using the :code:`|` character
(*pipe*, not l or i) to indicate the *depth*.
Read the `data model documentation`_ for more information.

.. _`data model documentation`: ../data.html#the-variable-column

The package provides several functions to work with such strings.

.. autofunction:: concat_with_pipe

.. autofunction:: find_depth

.. autofunction:: reduce_hierarchy

.. autofunction:: get_variable_components
