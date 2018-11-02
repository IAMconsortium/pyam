#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil
from subprocess import call

from setuptools import setup, Command, find_packages
from setuptools.command.install import install

# Thanks to http://patorjk.com/software/taag/
logo = r"""
 ______   __  __     ______     __    __
/\  == \ /\ \_\ \   /\  __ \   /\ "-./  \
\ \  _-/ \ \____ \  \ \  __ \  \ \ \-./\ \
 \ \_\    \/\_____\  \ \_\ \_\  \ \_\ \ \_\
  \/_/     \/_____/   \/_/\/_/   \/_/  \/_/
"""

INFO = {
    'version': '0.1.1',
}


class Cmd(install):
    """Custom clean command to tidy up the project root."""

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        dirs = [
            'pyam.egg-info',
            'build',
        ]
        for d in dirs:
            if os.path.exists(d):
                print('removing {}'.format(d))
                shutil.rmtree(d)


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
    cmdclass = {
        # 'install': Cmd,
    }
    install_requirements = [
        "argparse",
        "numpy",
        "pandas >=0.21.0",
        "PyYAML",
        "xlrd",
        "xlsxwriter",
        "matplotlib",
        "seaborn",
        "six",
    ]
    extra_requirements = {
        'tests': ['coverage', 'pytest', 'pytest-cov', 'pytest-mpl'],
    }
    setup_kwargs = {
        "name": "pyam-iamc",
        "version": INFO['version'],
        "description": 'Analyze & Visualize Assessment Model Results',
        "classifiers": classifiers,
        "license": "Apache License 2.0",
        "author": 'Matthew Gidden & Daniel Huppmann',
        "author_email": 'matthew.gidden@gmail.com',
        "url": 'https://github.com/IAMconsortium/pyam',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "cmdclass": cmdclass,
        "package_data": package_data,
        "install_requires": install_requirements,
        "extras_require": extra_requirements,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
