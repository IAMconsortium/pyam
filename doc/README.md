Building the docs
==================

We use Sphinx and restructured text for building the documentation pages.
Detailed documentation of the package is built from mark-up docstrings
in the source code.

Dependencies
------------

These can be found in ``environment.yml``.

Writing in Restructed Text
--------------------------

There are a number of guides out there, e.g. on `docutils
<http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_.

Building the documentation pages
--------------------------------

On *nix, from the command line, run::

    make html

On Windows, from the command line, run::

    ./make.bat

You can then view the site by::

    cd build
    python -m SimpleHTTPServer

and pointing your browser at http://localhost:8000/html/
