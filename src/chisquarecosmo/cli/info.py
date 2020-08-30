import click
import numpy
from chisquarecosmo import (
    get_dataset, get_model, registered_dataset_joins,
    registered_datasets, registered_models
)
from chisquarecosmo.util import console
from rich.padding import Padding
from rich.table import Table

_models = registered_models()
_datasets = registered_datasets()
_dataset_joins = registered_dataset_joins()


@click.command()
@click.option("--models",
              is_flag=True,
              help="List the currently implemented cosmological models.")
@click.option("--datasets",
              is_flag=True,
              help="List the currently implemented observational datasets, "
                   "including combinations of them.")
def info(models: bool, datasets: bool):
    """Display information about chisquarecosmo library."""
    # Show all if all flags are false.
    show_any = any([models, datasets])
    show_all = True if not show_any else False
    # Display the models
    if show_all or models:
        # Initialize grid for displaying models.
        models_grid = Table.grid(padding=(0, 2), expand=False)
        models_grid.add_column(justify="left", ratio=1)
        models_grid.add_column(justify="center", ratio=1)

        for model_name in _models:
            model = get_model(model_name)
            # Initialize parameters table.
            data_table = Table(expand=True)
            data_table.add_column("Name", justify="center", ratio=1)
            data_table.add_column("Default Value", justify="center", ratio=1)
            param_names = model.params_cls._fields
            param_defaults = model.params_cls._field_defaults
            for param_name in param_names:
                value = param_defaults.get(param_name, "---")
                data_table.add_row(f"{param_name}", f"{value}")
            models_grid.add_row(f"Name", f"Parameters")
            models_grid.add_row(Padding(f"[red]{model_name}[/]",
                                        pad=(1, 1, 1, 0)),
                                data_table)
            models_grid.add_row()

        # Display models grid.
        console.print(Padding("[yellow underline]MODELS", pad=(1, 2)),
                      justify="left")
        console.print(Padding(models_grid, pad=(0, 2)))

    # Display the datasets
    if show_all or datasets:
        # Initialize grid for displaying models.
        datasets_grid = Table.grid(padding=(0, 2), expand=True)
        # datasets_grid.add_column(justify="left", ratio=1)
        datasets_grid.add_column(justify="center", ratio=3)

        for dataset_name in _datasets:
            dataset = get_dataset(dataset_name)
            # Initialize parameters table.
            data_table = Table(expand=True, show_header=False)
            data_table.add_column(justify="center", ratio=1)
            data_table.add_column(justify="center", ratio=6)
            redshifts, observed, error = dataset.data.T
            with numpy.printoptions(threshold=50):
                data_table.add_row(f"Redshifts", f"{redshifts}")
                data_table.add_row(f"Observed", f"{observed}")
                data_table.add_row(f"Errors", f"{error}")
            datasets_grid.add_row(Padding(f"Name: [red]{dataset_name}[/] \n"
                                          f"Label: [red]{dataset.label}[/]"))
            datasets_grid.add_row(Padding(data_table, pad=(0, 0, 1, 0)))

        console.print(Padding("[yellow underline]DATASETS", pad=(1, 2)),
                      justify="center")
        console.print(Padding(datasets_grid, pad=(0, 2)))

        if _dataset_joins:
            console.print(Padding("[red]Dataset Joins", pad=(0, 2)),
                          justify="center")
            joins = []
            for join_name in _dataset_joins:
                if join_name in _datasets:
                    continue
                joins.append(join_name)
            console.print(Padding(console.highlighter(str(joins)),
                                  pad=(0, 2, 1, 2)), justify="center")
