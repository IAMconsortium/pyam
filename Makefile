.PHONY: new_release
new_release:
	@echo 'For a new release on PyPI:'
	@echo 'git tag vX.Y.Z'
	@echo 'make publish-on-pypi'

# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi:
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
			python setup.py sdist bdist_wheel --universal; \
			twine upload -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-testpypi-install:
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	# Install dependencies not on testpypi registry
	# may have to update this line
	$(TEMPVENV)/bin/pip install pandas
	# Install pyam without dependencies.
	$(TEMPVENV)/bin/pip install \
		-i https://testpypi.python.org/pypi pyam-iamc \
		--no-dependencies --pre
	# Remove local directory from path to get actual installed version.
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import pyam; print(pyam.__file__)"

.PHONY: publish-on-pypi
publish-on-pypi:
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
			python setup.py sdist bdist_wheel --universal; \
			twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install:
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install pyam-iamc --pre
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import pyam; print(pyam.__file__)"

.PHONY: release-on-conda
release-on-conda:
	@echo 'For now, this is all very manual'
	@echo 'Checklist:'
	@echo '- version number'
	@echo '- sha'
	@echo '- README.md badge'
	@echo '- release notes up to date'
