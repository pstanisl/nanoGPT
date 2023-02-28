CONDA_VIRTUAL_ENVIRONMENT_PREFIX?=./.venv
ENV_PREFIX := $(CONDA_VIRTUAL_ENVIRONMENT_PREFIX)
PYTHON_VERSION := 3.10

PATH := $(ENV_PREFIX)/bin:$(PATH)

help:
	@sed -n 's/^##//p' ./Makefile

all: init

## $(ENV_PREFIX)  : creates a new virtual environment with python${PYTHON_VERSION}
$(ENV_PREFIX):
	$(CONDA_EXE) create --prefix $(ENV_PREFIX) python=${PYTHON_VERSION} --yes

## check          : runs checks using isort and black, doesn't change any files (doesn't depend on poetry.install)
check:
	@echo "Checking imports (isort)"
	isort --check --diff ./src ./tests
	@echo "Checking code style (black)"
	black --check --diff ./src ./tests

check-types:
	pyright --skipunannotated -v $(ENV_PREFIX)

## clean          : remove compiled files and caches
clean:
	@echo "Cleaning the project"
	# $(CONDA_EXE) deactivate
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

## full-clean     : remove $(ENV_PREFIX), compiled files and caches
full-clean: clean
	rm -rf $(ENV_PREFIX)
	# rm .venv_path

## coverage       : run tests and coverage report
coverage:
	rm -rf tests-results/*
	poetry run coverage run --source=./src -m pytest tests/ --junitxml=test-results/results.xml
	poetry run coverage xml -o test-results/coverage.xml

## fix            : reformats all files using isort and black
fix: init
	isort ./src ./tests
	black ./src ./tests

## init           : initialize a project, i.e., install dependencies
init: poetry.install

poetry.install:
	poetry install
	echo "$(sha1sum ./poetry.lock 2>/dev/null || shasum ./poetry.lock)" > poetry.install

## test           : runs all tests in ./tests
test: init
	pytest ./tests

##
## ci/cd targets
## ci-init        : installs isort and black using pip, used in CI/CD pipelines
ci-init:
	pip install black==23.1.*
	pip install isort==5.12.*

## ci-check       : runs check using isort and black
ci-check: ci-init check

## ci-poetry-init : installs poetry and dependencies (without conda env)
ci-poetry-init:
	pip install poetry
	poetry install

.PHONY: help init check fix test clean full-clean coverage ci-check ci-init ci-poetry-init
