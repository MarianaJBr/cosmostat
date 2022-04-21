# [cosmostat][repo-url]

Python library to estimate chi-square constraints on cosmology models using
background quantities.

## Installation 🧱

Cosmostat project uses _poetry_ as the package and dependency manager. If _poetry_ is not installed
in your system, you can follow the installation instructions at the [poetry website][poetry-url].

To install cosmostat, it is recommended to create a virtual environment isolated from the main
Python interpreter. We have several tools to create a virtual environment, being **conda** the
recommended tool for handling virtual environments and software packages (for now).

### Using conda to manage virtual environments

The following sections assume we have ``conda`` already installed on our computers. The recommended
way to obtain ``conda`` is by installing [Miniconda][miniconda-site], which is available for Linux,
Mac OS, and Windows. Please refer to the [conda user guide][conda-guide] for installation, help, and
usage instructions.

#### Creating a virtual environment

Since poetry handles cosmostat dependencies, we only need conda to create a minimum, isolated
virtual environment. Cosmostat requires Python 3.7 and above to properly work, so we can create the
virtual environment as follows:

```shell
conda create -n cosmostatenv python=3.7
```

This environment contains python 3.7 as the interpreter and a few packages. Once created, we only
must activate it,

```shell
conda activate cosmostatenv
```

#### Installing cosmostat in development mode

Once the virtual environment has been activated, type the following instruction at the cosmostat
root directory:

```shell
poetry install
```

This command will install cosmostat and all of its dependencies in the ``cosmostatenv`` virtual
environment. We can have a look at all the installed packages in the environment by calling ``conda
list``. For most packages, conda will indicate they were installed from [PyPI][pypi-url], since they
were installed by poetry, not by conda.

In addition to installing cosmostat dependencies, poetry will install the cosmostat package (located
in the ``src`` subdirectory) in development mode. Therefore, the cosmostat package will be
importable from any python script or jupyter notebook as a regular python package.

## Command Line Interface

After installing in development mode, we can use the **CLI** of the library, whose functionality is
contained in the ``cosmostat`` (``cosmostat.exe`` on Windows) executable.

We can access the command help pages through

```shell
cosmostat --help
```

### Information about the models and datasets

We can get information about the implemented models and observational datasets with the ``info``
subcommand,

```shell
cosmostat info
```

For example, the CPL model accepts the following parameters,

```text
  Name            Parameters
         ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
  CPL    ┃    Name     ┃  Default   ┃
         ┃             ┃   Value    ┃
         ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
         │     w0      │    ---     │
         │     w1      │    ---     │
         │      h      │   0.6731   │
         │  omegabh2   │  0.02222   │
         │  omegach2   │   0.1197   │
         └─────────────┴────────────┘
```

### $\chi^2$-fitting and optimization procedure

By using the CLI, we can fit a cosmological model to one or more observational datasets. We do this
by minimizing the $\chi^2$ as a function of the model parameters. This procedure is realized through
the ``chi-square-fit`` subcommand,

```shell
cosmostat chi-square-fit
```

This command takes two arguments: the ``EOS_MODEL``, and the ``DATASET`` (or a union of one or more
datasets). We must then specify the optimization parameters bounds, or fix one or more of them with
the ``--param`` option. We also must supply an output file to save the best-fit parameters through
the ``-o`` option.

For example, we could try to fit the CPL model to the BAO dataset. According to the model
information, the CPL model requires specifying, at least, the parameters ``w0`` and ``w1``. The rest
of the parameters are optional; if omitted, the routine assumes they take their default values. We
can define the fitting procedure as follows:

* We want ``w0`` to lie in the interval ``[-3, 1]``. We pass the option
  ``--param w0 -3:1``.
* We want ``w1`` to lie in the interval ``[0, 1]``. We pass the option
  ``--param w1 -0:1``.
* We have to fix ``h`` to ``0.5``. We use ``--param h 0.5``.
* Parameters ``omegabh2`` and ``omegach2`` keep their default values. We can
  omit them from the command.
* The best-fit result should be saved in a HDF5 file named
  ``./best-fit-cpl#1.h5`` in the current working directory. We pass the option
  ``-o ./best-fit-cpl#1.h5``.

In summary, the full command line for this fitting becomes

```shell
cosmostat chi-square-fit CPL BAO --param w0 -3:1 --param w1 0:1 --param h 0.5 -o ./best-fit-cpl#1.h5
```

The program shows the execution progress and the best-fit result.

We can always see the detailed ``chi-square-fit`` subcommand usage through the
``--help`` option.

### Evaluation of $\chi^2$ in a parameter grid

In addition to fitting, we can evaluate the $\chi^2$ on a parameter grid. We do this through the
``chi-square-grid``  subcommand,

```shell
cosmostat chi-square-grid
```

Its usage is similar to the ``chi-square-fit`` command: we pass a model and a dataset as arguments.
We also specify the parameters that define the grid and/or parameters that should be held fixed. For
non-fixed parameters, we have to indicate a partition to evaluate $\chi^2$, i.e., the partition
bounds and how many elements it should have.

For example, we could evaluate the $\chi^2$ of the CPL model, together with the BAO dataset. We can
define the procedure as follows:

* We want a partition over ``w0`` in the interval ``[-3, 1]`` with ``100`` items. We pass the option
  ``--param w0 -3:1:100``.
* We want a partition over ``w1`` in the interval ``[0, 1]`` with ``100`` elements. We use the
  option ``--param w1 -0:1:100``.
* Parameters ``h``, ``omegach2``, and ``omegabh2`` keep their default values. We can omit them from
  the command.
* The best-fit result should be saved in an HDF5 file named ``./grid-cpl-bao#1.h5`` in the current
  working directory. We pass the option ``-o ./grid-cpl-bao#1.h5``.

The full command line to evaluate the grid becomes

```shell
cosmostat chi-square-grid CPL BAO --param w0 -3:1:100 --param w1 0:1:100 -o ./grid-cpl-bao#1.h5
```

The program shows the grid evaluation progress. By default, it uses all of the system's available
processes for evaluating the $\chi^2$ in parallel.
## Developing the code
If you want to develop the code, we suggest that you download it from the github webpage  

https://github.com/MarianaJBr/cosmostat/

Then you will enjoy all the feature of git repositories. You can even develop your own branch and get it merged to the public distribution.
## Using the code
You can use this software freely, provided that in your publications, you cite at least the paper "One parameterisation to fit them all" <http://arxiv.org/abs/2102.08561> but feel free to also cite "Modified gravity for surveys" <https://arxiv.org/abs/1804.04284> and "Probing a Steep EoS for Dark Energy with latest observations" <https://arxiv.org/abs/1708.08529>

## Authors

* Mariana Jaber, [https://github.com/MarianaJBr][gh-mjaber], [INSPIRE Profile][inspire-mjaber]
* Luisa Jaime, [INSPIRE Profile][inspire-ljaime]
* Gustavo Arciniega, [https://github.com/gustavoarciniega][gh-garciniega], [INSPIRE Profile][inspire-garciniega]
* Omar Abel Rodríguez-López , [https://github.com/oarodriguez/][gh-oarodriguez]

<!-- Links -->

[miniconda-site]: https://docs.conda.io/en/latest/miniconda.html
[conda-guide]: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
[poetry-url]: https://python-poetry.org/
[pypi-url]: https://pypi.org/
[repo-url]: https://github.com/oarodriguez/cosmostat
[gh-mjaber]: https://github.com/MarianaJBr
[inspire-mjaber]: https://inspirehep.net/authors/1707914
[inspire-ljaime]: https://inspirehep.net/authors/1258854
[gh-garciniega]: https://github.com/gustavoarciniega
[inspire-garciniega]: https://inspirehep.net/authors/1272389
[gh-oarodriguez]: https://github.com/oarodriguez
