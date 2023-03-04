.. currentmodule:: pyam

Advanced filtering
==================

|pyam| includes a function to directly downselect a :class:`pandas.DataFrame`
with appropriate columns or index dimensions
(i.e., :code:`['model', 'scenario']`)
using a :class:`IamDataFrame` and keyword arguments similar
to the :meth:`IamDataFrame.filter` function.

.. autofunction:: filter_by_meta
