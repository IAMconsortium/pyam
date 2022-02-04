"""This file is a **NOT** used to test specific functions of pyam, but rather
to print configuration information of the loaded package(s) to users!"""
import sys

from .test_plotting import MPL_KWARGS
from .conftest import IIASA_UNAVAILABLE


def test_config(capsys):
    modules = {}
    for m in list(sys.modules.values()):
        if m:
            version = getattr(m, "__version__", None)
            if version:
                modules[m.__name__] = version

    with capsys.disabled():
        print("\nPlotting function decorator kwargs:")
        for k, v in MPL_KWARGS.items():
            print("{}: {}".format(k, v))

        print("\nModule versions:")
        for key in sorted(list(modules.keys())):
            print("{}: {}".format(key, modules[key]))

        if IIASA_UNAVAILABLE:
            print("\nWARNING: IIASA-API unavailable, skipping related tests\n")

        # add empty spaces equivalent to length of file name
        print("tests/test_admin.py ", end="")
