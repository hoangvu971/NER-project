# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
pip-tools-train:
	pip install pip-tools==7.4.1 setuptools==70.3.0
	pip-compile requirements/dev.in
	pip install datasets -U
	pip-sync requirements/dev.txt

pip-tools-prod:
	pip install pip-tools==7.4.1 setuptools==70.3.0
	pip-compile requirements/prod.in
	pip-sync requirements/prod.txt

# Compile and install the requirements for local linting (optional)
pip-tools-lint:
	pip install pip-tools==7.4.1 setuptools==70.3.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in && pip-compile requirements/dev-lint.in
	pip-sync requirements/prod.txt requirements/dev.txt requirements/dev-lint.txt

# Bump versions of transitive dependencies
pip-tools-upgrade:
	pip install pip-tools==7.4.1 setuptools==70.3.0
	pip-compile --upgrade requirements/prod.in && pip-compile --upgrade requirements/dev.in && pip-compile --upgrade requirements/dev-lint.in

# Lint
lint:
	tasks/lint.sh