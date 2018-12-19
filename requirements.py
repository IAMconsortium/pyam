install_requirements = [
    "argparse",
    "numpy",
    "requests",
    "pandas >=0.21.0",
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
