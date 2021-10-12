.DEFAULT_GOAL := help

DOC_DIR=./doc
DOC_ENVIRONMENT_CONDA_FILE=$(DOC_DIR)/environment.yml


ifndef CONDA_PREFIX
$(error Conda not active, please install conda and then activate it using \`conda activate\`))
else
ifeq ($(CONDA_DEFAULT_ENV),base)
$(error Do not install to conda base environment. Source a different conda environment e.g. \`conda activate pyam\` or \`conda create --name pyam python=3.7\` and rerun make))
endif
VENV_DIR=$(CONDA_PREFIX)
endif

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


.PHONY: clean  ## clean all build artifacts
clean:
	rm -rf build dist *egg-info __pycache__

# first time setup, follow the 'Register for PyPI' section in this
# https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi: $(VENV_DIR)  ## publish release on test PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
		echo run git status --porcelain to find dirty files >&2; \
	fi;

.PHONY: publish-on-pypi
publish-on-pypi: $(VENV_DIR)  ## publish release on PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
		echo run git status --porcelain to find dirty files >&2; \
	fi;

.PHONY: test
test: $(VENV_DIR)  ## run all the tests
	$(VENV_DIR)/bin/pytest tests --cov=./ -r a --cov-report term-missing

.PHONY: test-with-mpl
test-with-mpl: $(VENV_DIR)  ## run all the tests including matplotlib
	$(VENV_DIR)/bin/pytest tests --cov=./ --mpl -r a --cov-report term-missing

.PHONY: docs
docs: $(VENV_DIR)  ## make the docs
	cd doc; make html

.PHONY: virtual-environment
virtual-environment: $(VENV_DIR)  ## make virtual environment for development

$(VENV_DIR):
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e .[tests,optional_io_formats]
	$(CONDA_EXE) env update --name $(CONDA_DEFAULT_ENV) --file $(DOC_ENVIRONMENT_CONDA_FILE)

	touch $(VENV_DIR)
