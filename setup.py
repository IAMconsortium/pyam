#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil
import versioneer

from setuptools import setup, Command
from subprocess import call

# Thanks to http://patorjk.com/software/taag/
logo = r"""
 ______   __  __     ______     __    __
/\  == \ /\ \_\ \   /\  __ \   /\ "-./  \
\ \  _-/ \ \____ \  \ \  __ \  \ \ \-./\ \
 \ \_\    \/\_____\  \ \_\ \_\  \ \_\ \ \_\
  \/_/     \/_____/   \/_/\/_/   \/_/  \/_/
"""

REQUIREMENTS = [
    'argparse',
    'numpy',
    'requests',
    'pandas>=0.25.0',
    'pint',
    'PyYAML',
    'xlrd',
    'xlsxwriter',
    'matplotlib<=3.0.2',
    'seaborn',
    'six',
]

EXTRA_REQUIREMENTS = {
    'tests': ['coverage', 'coveralls', 'pytest', 'pytest-cov', 'pytest-mpl'],
    'optional-io-formats': ['datapackage'],
    'deploy': ['twine', 'setuptools', 'wheel'],
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
CMDCLASS.update({'test': RunTests})


def main():
    print(logo)
    classifiers = [
        'License :: OSI Approved :: Apache Software License',
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
        'pyam': ['region_mappings/*', '../units/definitions.txt',
                 '../units/modules/**/*.txt'],
    }
    install_requirements = REQUIREMENTS
    extra_requirements = EXTRA_REQUIREMENTS
    setup_kwargs = {
        'name': 'pyam-iamc',
        'version': versioneer.get_version(),
        'cmdclass': CMDCLASS,
        'description': 'Analyze & Visualize Assessment Model Results',
        'classifiers': classifiers,
        'license': 'Apache License 2.0',
        'author': 'Matthew Gidden & Daniel Huppmann',
        'author_email': 'matthew.gidden@gmail.com',
        'url': 'https://github.com/IAMconsortium/pyam',
        'packages': packages,
        'package_dir': pack_dir,
        'entry_points': entry_points,
        'package_data': package_data,
        'install_requires': install_requirements,
        'extras_require': extra_requirements,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
