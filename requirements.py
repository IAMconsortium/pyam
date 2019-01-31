install_requirements = [
    "argparse",
    "numpy",
    "requests",
    "pandas>=0.21.0, <=0.23.4",
    "PyYAML",
    "xlrd",
    "xlsxwriter",
    "matplotlib",
    "seaborn",
    "six",
]


def display():
    for x in install_requirements:
        print(x)
