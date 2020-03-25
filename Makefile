.DEFAULT_GOAL := help

CI_DIR=./ci
CI_ENVIRONMENT_CONDA_DEFAULT_FILE=$(CI_DIR)/environment-conda-default.txt
CI_ENVIRONMENT_CONDA_FORGE_FILE=$(CI_DIR)/environment-conda-forge.txt

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

.PHONY: new-release
new-release:  ## make a new release of pyam
	@echo 'For a new release on PyPI:'
	@echo 'git tag vX.Y.Z'
	@echo 'make publish-on-pypi'

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

.PHONY: regenerate-test-figures
regenerate-test-figures: $(VENV_DIR)  ## re-generate all test figures
	$(VENV_DIR)/bin/pytest --mpl-generate-path=tests/expected_figs tests/test_plotting.py

.PHONY: test
test: $(VENV_DIR)  ## run all the tests
	cd tests && \
	@pyampath=$$(dirname $$(python -c "import pyam; print(pyam.__file__)")) && \
	$(VENV_DIR)/bin/pytest "$${pyampath}"/tests --mpl --cov=pyam --cov-config ../ci/.coveragerc -rfsxEX --cov-report term-missing

.PHONY: install
install: $(VENV_DIR)  ## install pyam in virtual env
	$(VENV_DIR)/bin/python setup.py install

.PHONY: docs
docs: $(VENV_DIR)  ## make the docs
	cd doc; make html

.PHONY: virtual-environment
virtual-environment: $(VENV_DIR)  ## make virtual environment for development

$(VENV_DIR):  $(CI_ENVIRONMENT_CONDA_DEFAULT_FILE) $(CI_ENVIRONMENT_CONDA_FORGE_FILE)
	# TODO: unify with ci install instructions somehow
	$(CONDA_EXE) config --add channels conda-forge # sets conda-forge as highest priority
	$(CONDA_EXE) install --yes $(shell cat $(CI_ENVIRONMENT_CONDA_DEFAULT_FILE) $(CI_ENVIRONMENT_CONDA_FORGE_FILE) | tr '\n' ' ')
	# Install development setup
	$(VENV_DIR)/bin/pip install -e .[tests,deploy,optional-io-formats]
	# install docs requirements
	# --name $(CONDA_DEFAULT_ENV) ensures we install in active environment (check at
	# top of Makefile ensures that environment is not the base one)
	$(CONDA_EXE) env update --name $(CONDA_DEFAULT_ENV) --file $(DOC_ENVIRONMENT_CONDA_FILE)
# 	touch $(VENV_DIR)

.PHONY: release-on-conda
release-on-conda:  ## release pyam on conda
	@echo 'For now, this is all very manual'
	@echo 'Checklist:'
	@echo '- version number'
	@echo '- sha'
	@echo '- README.md badge'
	@echo '- release notes up to date'
