"""This file is a **NOT** used to test specific functions of pyam, but rather
to print configuration information of the loaded package(s) to users!"""
import sys


def test_print_imports(capsys):
    print('Module versions:')
    modules = {}
    for m in sys.modules.values():
        if m:
            version = getattr(m, '__version__', None)
            if version:
                modules[m.__name__] = version

    with capsys.disabled():
        for key in sorted(list(modules.keys())):
            print('{}: {}'.format(key, modules[key]))
