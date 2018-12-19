#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil
import versioneer

from setuptools import setup, Command
from subprocess import call

import requirements

# Thanks to http://patorjk.com/software/taag/
logo = r"""
 ______   __  __     ______     __    __
/\  == \ /\ \_\ \   /\  __ \   /\ "-./  \
\ \  _-/ \ \____ \  \ \  __ \  \ \ \-./\ \
 \ \_\    \/\_____\  \ \_\ \_\  \ \_\ \ \_\
  \/_/     \/_____/   \/_/\/_/   \/_/  \/_/
"""

INFO = {
    'version': '0.1.2',
}


# thank you https://stormpath.com/blog/building-simple-cli-interfaces-in-python
class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=skele', '--cov-report=term-missing'])
        raise SystemExit(errno)


CMDCLASS = versioneer.get_cmdclass()
CMDCLASS.update({"test": RunTests})


def main():
    print(logo)
    classifiers = [
        "License :: OSI Approved :: Apache Software License",
    ]
    packages = [
        'pyam',
    ]
    pack_dir = {
        'pyam': 'pyam',
    }
    entry_points = {
        'console_scripts': [
            # list CLIs here
        ],
    }
    package_data = {
        'pyam': ['region_mappings/*'],
    }
    install_requirements = requirements.install_requirements
    extra_requirements = {
        'tests': ['coverage', 'pytest', 'pytest-cov', 'pytest-mpl'],
        'docs': ['sphinx', 'sphinx_rtd_theme'],
        'deploy': ['twine', 'setuptools', 'wheel'],
    }
    setup_kwargs = {
        "name": "pyam-iamc",
        "version": versioneer.get_version(),
        "cmdclass": CMDCLASS,
        "description": 'Analyze & Visualize Assessment Model Results',
        "classifiers": classifiers,
        "license": "Apache License 2.0",
        "author": 'Matthew Gidden & Daniel Huppmann',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'https://github.com/IAMconsortium/pyam',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "package_data": package_data,
        "install_requires": install_requirements,
        "extras_require": extra_requirements,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
