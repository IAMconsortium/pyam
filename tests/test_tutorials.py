import io
import os
import subprocess
import sys
import pytest

from .conftest import here, IIASA_UNAVAILABLE

try:
    import nbformat
except:
    pytest.skip(
        "Missing Jupyter Notebook and related dependencies", allow_module_level=True
    )

tut_path = os.path.join(here, "..", "doc", "source", "tutorials")


# taken from the excellent example here:
# https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/


def _notebook_run(path, kernel=None, timeout=60, capsys=None):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)
    """
    major_version = sys.version_info[0]
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    fname = os.path.join(here, "test.ipynb")
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout={}".format(timeout),
        "--output",
        fname,
        path,
    ]
    subprocess.check_call(args)

    nb = nbformat.read(io.open(fname, encoding="utf-8"), nbformat.current_nbformat)

    errors = [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]

    # removing files fails on CI (GitHub Actions) on Windows & py3.8
    try:
        os.remove(fname)
    except PermissionError:
        pass

    return nb, errors


def test_pyam_first_steps(capsys):
    fname = os.path.join(tut_path, "pyam_first_steps.ipynb")
    nb, errors = _notebook_run(fname, capsys=capsys)
    assert errors == []
    assert os.path.exists(os.path.join(tut_path, "tutorial_export.xlsx"))

    def has_log_output(cell):
        return cell["cell_type"] == "code" and any(
            "Running in a notebook" in output.get("text", "")
            for output in cell["outputs"]
        )

    assert any(has_log_output(cell) for cell in nb["cells"])


def test_data_table_formats():
    fname = os.path.join(tut_path, "data_table_formats.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_unit_conversion():
    fname = os.path.join(tut_path, "unit_conversion.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_aggregating_downscaling_consistency():
    fname = os.path.join(tut_path, "aggregating_downscaling_consistency.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_subannual_time_resolution():
    fname = os.path.join(tut_path, "subannual_time_resolution.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_pyam_logo():
    fname = os.path.join(tut_path, "pyam_logo.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_ipcc_colors():
    fname = os.path.join(tut_path, "ipcc_colors.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_legends():
    fname = os.path.join(tut_path, "legends.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


def test_ops():
    fname = os.path.join(tut_path, "algebraic_operations.ipynb")
    nb, errors = _notebook_run(fname)
    assert errors == []


@pytest.mark.skipif(IIASA_UNAVAILABLE, reason="IIASA database API unavailable")
def test_iiasa_dbs():
    fname = os.path.join(tut_path, "iiasa_dbs.ipynb")
    nb, errors = _notebook_run(fname, timeout=600)
    assert errors == []


def test_aggregating_variables_and_plotting_with_negative_values():
    fname = os.path.join(
        tut_path, "aggregating_variables_and_plotting_with_negative_values.ipynb"
    )
    nb, errors = _notebook_run(fname)
    assert errors == []
