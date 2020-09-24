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
    'iam-units >= 2020.4.12',
    'numpy',
    'requests',
    'pandas>=0.25.0',
    'pint',
    'PyYAML',
    'matplotlib',
    'seaborn',
    'six',
]

EXTRA_REQUIREMENTS = {
    'tests': ['coverage', 'coveralls', 'pytest<6.0.0', 'pytest-cov',
              'pytest-mpl'],
    'optional-io-formats': ['datapackage', 'pandas-datareader'],
    'deploy': ['twine', 'setuptools', 'wheel'],
    'tutorials': ['pypandoc', 'nbformat', 'nbconvert', 'jupyter_client',
                  'ipykernel'],
    'docs': ['sphinx', 'nbsphinx', 'sphinx-gallery', 'cloud_sptheme',
             'pillow', 'sphinxcontrib-bibtex', 'sphinxcontrib-programoutput',
             'numpydoc', 'openpyxl']  # docs also requires 'tutorials'
}

# building the docs on readthedocs fails with a FileNotFoundError
# https://github.com/IAMconsortium/pyam/issues/363
try:
    with open('README.md', 'r') as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = ''


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
    description = 'Analysis & visualization of integrated-assessment scenarios'
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
        'pyam': ['region_mappings/*'],
    }
    install_requirements = REQUIREMENTS
    extra_requirements = EXTRA_REQUIREMENTS
    setup_kwargs = dict(
        name='pyam-iamc',
        version=versioneer.get_version(),
        cmdclass=CMDCLASS,
        description=description,
        classifiers=classifiers,
        license='Apache License 2.0',
        author='Matthew Gidden & Daniel Huppmann',
        author_email='pyam+owner@groups.io',
        url='https://github.com/IAMconsortium/pyam',
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=packages,
        package_dir=pack_dir,
        entry_points=entry_points,
        package_data=package_data,
        install_requires=install_requirements,
        extras_require=extra_requirements,
    )
    setup(**setup_kwargs)


if __name__ == "__main__":
    main()
