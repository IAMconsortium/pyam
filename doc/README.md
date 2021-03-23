Building the docs
==================

We use Sphinx and restructured text (rst) for building the documentation pages.
Detailed documentation of the package is built from mark-up docstrings 
in the source code.

Dependencies
------------

To install the **pyam** package and all dependencies, run the following
(in the top-level directory of this repository).

```
pip install --editable .[docs,tutorials]
```

Writing in Restructured Text
----------------------------

There are a number of guides to get started, for example
on [sourceforge](https://docutils.sourceforge.io/docs/user/rst/quickref.html).

Building the documentation pages
--------------------------------

On *nix, from the command line, run::

    make html

On Windows, from the command line, run::

    ./make.bat

The rendered html pages will be located in `doc/build/html/index.html`.