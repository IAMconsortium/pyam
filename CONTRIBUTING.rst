Have a question? Get in touch!
------------------------------

- Send bug reports, suggest new features or view the source code `on GitHub`_,
- Reach out via our community mailing list hosted by `groups.io`_,
- Or send us an `email`_ to join our Slack_ workspace!

.. _on GitHub: http://github.com/IAMconsortium/pyam
.. _`groups.io`: https://groups.io/g/pyam
.. _`email`: mailto:pyam+owner@groups.io?subject=[pyam]%20Please%20add%20me%20to%20the%20Slack%20workspace
.. _Slack: https://slack.com

Interested in contributing? Join the team!
------------------------------------------

The pyam package has been developed with the explicit aim to facilitate
open and collaborative analysis of integrated assessment and climate models.
We appreciate contributions to the code base and development of new features.

Please use the GitHub *Issues* feature to raise questions concerning potential
bugs or to propose new features, but search and read resolved/closed topics on
similar subjects before raising a new issue.

For contributions to the code base, please use GitHub *Pull Requests*,
including a detailed description of the new feature and unit tests
to illustrate the intended functionality.
Code submitted via pull requests must adhere to the `pep8`_ style formats
and the documentation should follow  the `numpydoc docstring guide`_. We are 
using `ruff`_ to check the code style.

We do not require users to sign a *Contributor License Agreement*, because we
believe that when posting ideas or submitting code to an open-source project,
it should be obvious and self-evident that any such contributions
are made in the spirit of open collaborative development.

Setup
-----

.. code-block:: bash

    # Install Poetry, minimum version >=1.2 required
    curl -sSL https://install.python-poetry.org | python -

    # You may have to reinitialize your shell at this point.
    source ~/.bashrc

    # Activate in-project virtualenvs
    poetry config virtualenvs.in-project true

    # Add dynamic versioning plugin
    poetry self add "poetry-dynamic-versioning[plugin]"

    # Install dependencies
    # (using "--with docs" if docs dependencies should be installed as well)
    poetry install --with docs,server,dev

    # Activate virtual environment
    poetry shell

    # Copy the template environment configuration
    cp template.env .env

Update poetry
^^^^^^^^^^^^^

Developing ixmp4 requires poetry ``>= 1.2``.

If you already have a previous version of poetry installed you will need to update. The
first step is removing the old poetry version:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 - --uninstall


after that, the latest poetry version can be installed using:

.. code-block:: bash
    curl -sSL https://install.python-poetry.org | python3 -


details can be found here in the poetry docs:
https://python-poetry.org/docs/#installation.

Resolve conflicts in poetry.lock
--------------------------------

When updating dependencies it can happen that a conflict between the current and the
target poetry.lock file occurs. In this case the following steps should be taken to
resolve the conflict.

#. Do not attempt to manually resolve in the GitHub web interface.
#. Instead checkout the target branch locally and merge into your branch:

.. code-block:: bash
    git checkout main
    git pull origin main
    git checkout my-branch
    git merge main


#. After the last step you'll have a merge conflict in poetry.lock.
#. Instead of resolving the conflict, directly checkout the one from main and rewrite
   it:

.. code-block:: bash
    # Get poetry.lock to look like it does in master
    git checkout main poetry.lock
    # Rewrite the lock file
    poetry lock --no-update

#. After that simply add poetry.lock to mark the conflict as resolved and commit to
   finalize the merge:

.. code-block:: bash
    git add poetry.lock
    git commit

    # and most likely needed
    poetry install

(Taken from https://www.peterbe.com/plog/how-to-resolve-a-git-conflict-in-poetry.lock)

.. _`pep8`: https://www.python.org/dev/peps/pep-0008/

.. _`numpydoc docstring guide`: https://numpydoc.readthedocs.io/en/latest/format.html

.. _`ruff`: https://docs.astral.sh/ruff/
