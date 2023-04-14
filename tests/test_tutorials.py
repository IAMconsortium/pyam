import pytest

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ModuleNotFoundError:
    pytest.skip(
        "Missing Jupyter Notebook and related dependencies", allow_module_level=True
    )

from .conftest import here, IIASA_UNAVAILABLE

nb_path = here.parent / "docs" / "tutorials"


def _run_notebook(file, timeout=30):
    """Execute a notebook file"""
    with open(nb_path / f"{file}.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout)
    ep.preprocess(nb, {"metadata": {"path": nb_path}})


@pytest.mark.parametrize(
    "file",
    [
        "pyam_first_steps",
        "data_table_formats",
        "unit_conversion",
        "aggregating_downscaling_consistency",
        "subannual_time_resolution",
        "pyam_logo",
        "ipcc_colors",
        "legends",
        "algebraic_operations",
        "aggregating_variables_and_plotting_with_negative_values",
    ],
)
def test_tutorial_notebook(file):
    _run_notebook(file)


@pytest.mark.skipif(IIASA_UNAVAILABLE, reason="IIASA database API unavailable")
def test_tutorial_iiasa_dbs():
    _run_notebook("iiasa_dbs")
