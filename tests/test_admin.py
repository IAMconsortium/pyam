"""This file is a **NOT** used to test specific functions of pyam, but rather
to print configuration information of the loaded package(s) to users!"""
import sys

from test_plotting import MPL_KWARGS


def test_config(capsys):
    modules = {}
    for m in sys.modules.values():
        if m:
            version = getattr(m, '__version__', None)
            if version:
                modules[m.__name__] = version

    with capsys.disabled():
        print()
        print('Plotting function decorator kwargs:')
        for k, v in MPL_KWARGS.items():
            print('{}: {}'.format(k, v))

        print()
        print('Module versions:')
        for key in sorted(list(modules.keys())):
            print('{}: {}'.format(key, modules[key]))
