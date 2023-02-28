# nanogpt

## Tools used in this project

* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6)

## Project structure

```bash
.
├── .github
│   └── workflows
│       ├── nanogpt-cheks.yaml  # Code checks CI pipeline
│       └── nanogpt-tests.yaml  # Code tests CI pipeline
├── conf
│   └── default.yml                 # Default configuration file
├── docs                            # documentation for your project
├── notebooks                       # store notebooks
│   ├── example.ipynb               # example notebook with classification
│   └── README.md                   # describe notebooks
├── src                             # store source code
│   └── nanogpt
│       ├── __init__.py             # make it a Python module
│       ├── config.py               # config definition for example.py
│       └── example.py              # example with classification
├── tests                           # store tests
│   └── nanogpt
│       └── test_example.py         # tests for example.py
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── poetry.toml                     # poetry config
├── pyproject.toml                  # dependencies for poetry
└── README.md                       # describe your project
```

## Set up the environment

1. Install [Poetry](https://python-poetry.org/docs/#installation)
1. Create the environment:

    ```bash
    make ./.venv
    ```

1. Activate the environment

    ```bash
    conda activate ./.venv
    ```

1. Install dependencies

    ```bash
    make init
    ```

## Install new packages

```bash
poetry add <package-name>
```

## Check the project

To check imports and code style of the project run:

```bash
make check
```

### Static type checking

For static type checking [pyright](https://github.com/microsoft/pyright) library is used.

```bash
make check-types
```

## Project clean

To remove ccompiled files and caches, run

```bash
make clean
```

> For removing `.venv` use `make full-clean`.
