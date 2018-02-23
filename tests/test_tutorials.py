import io
import os
import subprocess
import sys
import tempfile
import pytest

from testing_utils import here

try:
    import nbformat
    jupyter_installed = True
except:
    jupyter_installed = False

tut_path = os.path.join(here, '..', 'tutorial')

# taken from the execellent example here:
# https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/


def _notebook_run(path, kernel=None, capsys=None):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)
    """
    major_version = sys.version_info[0]
    kernel = kernel or 'python{}'.format(major_version)
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    fname = os.path.join(here, 'test.ipynb')
    args = [
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=60",
        "--ExecutePreprocessor.kernel_name={}".format(kernel),
        "--output", fname, path]
    subprocess.check_call(args)

    nb = nbformat.read(io.open(fname, encoding='utf-8'),
                       nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell
        for output in cell["outputs"] if output.output_type == "error"
    ]

    os.remove(fname)

    return nb, errors


@pytest.mark.skipif(not jupyter_installed, reason='requires Jupyter Notebook to be installed')
def test_pyam_first_steps(capsys):
    fname = os.path.join(tut_path, 'pyam_first_steps.ipynb')
    nb, errors = _notebook_run(fname, capsys=capsys)
    assert errors == []


@pytest.mark.skipif(not jupyter_installed, reason='requires Jupyter Notebook to be installed')
def test_plotting():
    fname = os.path.join(tut_path, 'plotting.ipynb')
    nb, errors = _notebook_run(fname)
    assert errors == []
