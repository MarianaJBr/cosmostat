# [cosmostat][repo-url]

Python library to estimate chi-square constraints on cosmology models using
background quantities.

## Installation 🧱

We use ``conda`` as the package and dependency manager of our project.

The following sections assume we have ``conda`` already installed on our
computers. The recommended way to obtain ``conda`` is by installing
[Miniconda][miniconda-site], which is available for Linux, Mac OS, and Windows.
Please refer to the [conda user guide][conda-guide] for installation, help, and
usage instructions.

### Creating a conda environment

The basics requirements of the project are listed in the
``conda/env-specs/default.yml`` environment file. We create a new conda
environment by typing the following instruction at the project root directory:

```shell
conda env create -f ./conda/env-specs/default.yml
```

After this operation completes, ``conda`` will create a new environment named
``cosmostat-dev``. We can verify this information by executing ``conda info
-e``. Finally, we activate the environment with

```shell
conda activate cosmostat-dev
```

The environment contains python 3.7 as the interpreter. We can look at all of
the installed packages by calling ``conda list``.

### Installing in development mode

We can install the library in development mode. This way, the packages and
modules can be imported from any python script or jupyter notebook, as long as
the ``cosmostat-dev`` environment remains active.

Installing in development mode is achieved by executing the following
instruction at the project root:

```shell
 pip install -e . --no-deps --no-build-isolation
```

## Command Line Interface

After installing in development mode, we can use the **CLI** of the library,
whose functionality is contained in the ``cosmostat``
(``cosmostat.exe`` on Windows) executable.

We can access the command help pages through

```shell
cosmostat --help
```

### Information about the models and datasets

We can get information about the implemented models and observational datasets
with the ``info`` subcommand,

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

By using the CLI, we can fit a cosmological model to one or more observational
datasets. We do this by minimizing the $\chi^2$ as a function of the model
parameters. This procedure is realized through the ``chi-square-fit``
subcommand,

```shell
cosmostat chi-square-fit
```

This command takes two arguments: the ``EOS_MODEL``, and the ``DATASET`` (or a
union of one or more datasets). We must then specify the optimization parameters
bounds, or fix one or more of them with the ``--param`` option. We also must
supply an output file to save the best-fit parameters through the ``-o`` option.

For example, we could try to fit the CPL model to the BAO dataset. According to
the model information, the CPL model requires specifying, at least, the
parameters ``w0`` and ``w1``. The rest of the parameters are optional; if
omitted, the routine assumes they take their default values. We can define the
fitting procedure as follows:

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

In addition to fitting, we can evaluate the $\chi^2$ on a parameter grid. We do
this through the ``chi-square-grid``  subcommand,

```shell
cosmostat chi-square-grid
```

Its usage is similar to the ``chi-square-fit`` command: we pass a model and a
dataset as arguments. We also specify the parameters that define the grid
and/or parameters that should be held fixed. For non-fixed parameters, we have
to indicate a partition to evaluate $\chi^2$, i.e., the partition bounds and how
many elements it should have.

For example, we could evaluate the $\chi^2$ of the CPL model, together with the
BAO dataset. We can define the procedure as follows:

* We want a partition over ``w0`` in the interval ``[-3, 1]`` with ``100``
  items. We pass the option ``--param w0 -3:1:100``.
* We want a partition over ``w1`` in the interval ``[0, 1]`` with ``100``
  elements. We use the option ``--param w1 -0:1:100``.
* Parameters ``h``, ``omegach2``, and ``omegabh2`` keep their default values. We
  can omit them from the command.
* The best-fit result should be saved in an HDF5 file named
  ``./grid-cpl-bao#1.h5`` in the current working directory. We pass the option
  ``-o ./grid-cpl-bao#1.h5``.

The full command line to evaluate the grid becomes

```shell
cosmostat chi-square-grid CPL BAO --param w0 -3:1:100 --param w1 0:1:100 -o ./grid-cpl-bao#1.h5
```

The program shows the grid evaluation progress. By default, it uses all of the
system's available processes for evaluating the $\chi^2$ in parallel.

## Authors

* Mariana Jaber, [https://github.com/MarianaJBr][gh-mjaber], [INSPIRE Profile][inspire-mjaber]
* Luisa Jaime, [INSPIRE Profile][inspire-ljaime]
* Gustavo Arciniega, [https://github.com/gustavoarciniega][gh-garciniega], [INSPIRE Profile][inspire-garciniega]
* Omar Abel Rodríguez-López , [https://github.com/oarodriguez/][gh-oarodriguez]

<!-- Links -->

[miniconda-site]: https://docs.conda.io/en/latest/miniconda.html
[conda-guide]: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
[repo-url]: https://github.com/oarodriguez/cosmostat
[gh-mjaber]: https://github.com/MarianaJBr
[inspire-mjaber]: https://inspirehep.net/authors/1707914
[inspire-ljaime]: https://inspirehep.net/authors/1258854
[gh-garciniega]: https://github.com/gustavoarciniega
[inspire-garciniega]: https://inspirehep.net/authors/1272389
[gh-oarodriguez]: https://github.com/oarodriguez
