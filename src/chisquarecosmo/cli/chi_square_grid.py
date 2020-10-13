import pathlib
import typing as t

import click
import h5py
import numpy as np
from chisquarecosmo.chi_square import (
    FixedParamSpec, Grid, ParamPartitionSpec, has_grid
)
from chisquarecosmo.cosmology import (
    get_dataset_join, get_model, registered_dataset_joins,
    registered_models
)
from chisquarecosmo.exceptions import CLIError
from chisquarecosmo.util import DaskProgressBar, console
from click import BadParameter
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
# Accepted parameter partition scales.
from toolz import groupby

VALID_PARTITION_SCALES = ["linear", "log", "geom"]

T_CLIParam = t.Tuple[str, str]
T_CLIParams = t.Tuple[T_CLIParam, ...]
T_GridParamSpec = t.Union[t.Type[FixedParamSpec], t.Type[ParamPartitionSpec]]
T_GridParamSpecs = t.Dict[T_GridParamSpec, t.List]


def _get_partition_spec(name: str, string: str):
    """Parse the partition bounds, and its number of elements."""
    partition_spec = string.split(":", maxsplit=2)
    if len(partition_spec) != 3:
        if len(partition_spec) == 1:
            value = partition_spec[0]
            try:
                value = float(value)
            except ValueError:
                err_msg = f"invalid value '{value}' for parameter " \
                          f"'{name}'"
                raise BadParameter(err_msg)
            # Return a fixed parameter spec.
            return value,
        else:
            err_msg = f"invalid partition '{string}' for parameter " \
                      f"'{name}'"
            raise BadParameter(err_msg)
    lower, upper, num_parts = partition_spec
    # Lower limit.
    try:
        lower = float(lower)
    except ValueError:
        err_msg = f"invalid lower bound '{lower}' for " \
                  f"parameter '{name}' partition"
        raise BadParameter(err_msg)
    # Upper limit.
    try:
        upper = float(upper)
    except ValueError:
        err_msg = f"invalid upper bound '{upper}' for " \
                  f"parameter '{name}' partition"
        raise BadParameter(err_msg)
    # Number of parts of partition.
    if not num_parts:
        err_msg = f"the number of elements for parameter '{name}' partition " \
                  f"is required"
        raise BadParameter(err_msg)
    else:
        try:
            num_parts = int(num_parts)
            if num_parts < 1:
                err_msg = f"'{num_parts}' is not a valid number of elements " \
                          f"for parameter '{name}' partition; must be " \
                          f"greater or equal than one"
                raise BadParameter(err_msg)
        except ValueError:
            err_msg = f"'{num_parts}' is not a valid number of elements for " \
                      f"parameter '{name}' partition"
            raise BadParameter(err_msg)
    return lower, upper, num_parts


def _get_partition_scale(name: str, string: str):
    """Parse the scale and base of a grid parameter."""
    scale_parts = string.split(":", maxsplit=1)
    if len(scale_parts) == 1:
        scale = scale_parts[0]
        base = 10 if scale == "log" else None
    else:
        # We pass scale and base.
        scale, base = scale_parts
        try:
            base = int(base)
            # Watch out with zero and negative bases.
            if base < 2:
                raise ValueError
        except ValueError:
            err_msg = f"invalid base '{base}' for parameter " \
                      f"'{name}' partition scale; must be a integer " \
                      f"greater than 2"
            raise BadParameter(err_msg)
        # Watch out with geometric scales...
        if scale == "geom":
            err_msg = f"base '{base}' is not allowed for a geometric scale " \
                      f"in parameter '{name}' partition scale"
            raise BadParameter(err_msg)
    # Verify that the scale is valid.
    if scale not in VALID_PARTITION_SCALES:
        err_msg = f"invalid scale '{scale}' for parameter " \
                  f"'{name}' partition; must be one of " \
                  f"{VALID_PARTITION_SCALES}"
        raise BadParameter(err_msg)
    return scale, base


def validate_param(param: T_CLIParam):
    """Validates a single parameter."""
    name, value_str = param
    value_parts = value_str.split("@", maxsplit=1)
    num_parts = len(value_parts)
    if num_parts != 2:
        if num_parts == 1:
            string = value_parts[0]
            partition_spec = _get_partition_spec(name, string)
            try:
                value, = partition_spec
                return FixedParamSpec(name, value)
            except ValueError:
                lower, upper, num_parts = partition_spec
                return ParamPartitionSpec.from_range_spec(name, lower, upper,
                                                          num_parts)
        else:
            err_msg = f"invalid partition spec '{value_str}' for parameter " \
                      f"'{name}'"
            raise BadParameter(err_msg)
    partition_str, scale_str = value_parts
    partition_spec = _get_partition_spec(name, partition_str)
    lower, upper, num_parts = partition_spec
    scale, base = _get_partition_scale(name, scale_str)
    if (lower < 0 or upper < 0) and (scale in ["geom"]):
        err_msg = f"can not mix negative boundaries and geom scales " \
                  f"in parameter '{name}' spec " \
                  f"'{lower}:{upper}:{num_parts}@{scale}'"
        raise BadParameter(err_msg)
    return ParamPartitionSpec.from_range_spec(name, lower, upper, num_parts,
                                              scale=scale, base=base)


def validate_params(ctx, param, values: T_CLIParams):
    """Validates the parameter list for the chi2 function."""
    return groupby(type, map(validate_param, values))


# Use these values as possible choices for EOS_MODEL and DATASET arguments.
_models = registered_models()
_datasets = registered_dataset_joins()


@click.command()
@click.argument("eos_model",
                type=click.Choice(_models),
                metavar="EOS_MODEL")
@click.argument("datasets",
                type=click.Choice(_datasets),
                metavar="DATASETS")
@click.option("--param",
              type=(str, str),
              multiple=True,
              default=None,
              callback=validate_params,
              help="Defines a partition for a parameter values. "
                   "The first text element indicates "
                   "the parameter name, and it is dependent on the equation "
                   "of stated used in the chi-square fitting process. "
                   "On the one "
                   "hand, as a single floating-point number, the second text "
                   "element fixes the parameter value during the minimization. "
                   "On the other hand, as a string with format 'lower:upper' "
                   "(where 'lower' and 'upper' are floating-point numbers, "
                   "including 'inf' and '-inf'), it defines the interval "
                   "bounds where the parameter will be varied. If 'lower' or "
                   "'upper' are omitted (but not the middle colon), then the "
                   "lower bound becomes '-inf', while the upper bound becomes "
                   "'inf'. If both 'lower' and 'upper' are equal, the "
                   "parameter is kept fixed during the minimization.")
@click.option("-o", "--output",
              type=click.Path(exists=False),
              required=True,
              help="HDF5 file where the fitting result will be saved.")
@click.option("-g", "--hdf5-group",
              type=str,
              default="/",
              help="Group where the fitting result will be saved. If "
                   "omitted, the result is saved in the root group.")
@click.option("-f", "--force-output",
              is_flag=True,
              default=False,
              help="If the output file already contains a best-fit result, "
                   "replace it with the new result.")
def chi_square_grid(eos_model: str, datasets: str, param: T_GridParamSpecs,
                    output: str, hdf5_group: str, force_output: bool):
    """Evaluate the chi-square over a grid according to best-fit in FILE.

    The chi-square is evaluated for the same model, dataset, and best-fit
    parameters stored in
    """
    out_file = pathlib.Path(output).resolve()
    base_group_name = hdf5_group.strip() or "/"
    # Append to existing file, or create a new one.
    file_mode = "a" if out_file.exists() else "w"
    _eos_model = get_model(eos_model)
    datasets = get_dataset_join(datasets)
    params_cls = _eos_model.params_cls
    fixed_specs: t.List[FixedParamSpec] = param.get(FixedParamSpec, [])
    partition_specs: t.List[ParamPartitionSpec] = \
        param.get(ParamPartitionSpec, [])

    # Parameters defined for the current model/eos.
    param_names = list(getattr(params_cls, "_fields"))
    defaults_dict = getattr(params_cls, "_field_defaults")
    names_with_defaults = set(defaults_dict)
    fixed_names_set = {spec.name for spec in fixed_specs}
    partition_names_set = {spec.name for spec in partition_specs}
    spec_names_set = fixed_names_set | partition_names_set
    required_names = [name for name in param_names if
                      name not in names_with_defaults]
    missing_names = [name for name in required_names if
                     name not in spec_names_set]
    unknown_names = [name for name in spec_names_set if name not in param_names]

    # Raise error if we have both missing names and unknown names.
    if missing_names and unknown_names:
        err_msg = f"the required parameters {missing_names} " \
                  f"were not specified, while the supplied parameters " \
                  f"{unknown_names} are unknown; the model only accepts " \
                  f"the following parameters: {param_names}"
        raise CLIError(f"{err_msg}")
    # Raise error if there are missing names.
    elif missing_names:
        err_msg = f"the required parameters {missing_names} were not specified"
        raise CLIError(f"{err_msg}")
    # Raise error if there are unknown names.
    elif unknown_names:
        err_msg = f"unknown parameters {unknown_names}. The model accepts " \
                  f"the following parameters: {param_names}"
        raise CLIError(f"{err_msg}")

    # Get all parameters that are fixed by default.
    fixed_names_set = (
        names_with_defaults - partition_names_set - fixed_names_set)
    fixed_specs.extend([
        FixedParamSpec(name, defaults_dict[name]) for name in fixed_names_set
    ])

    # Perform checks.
    if out_file.exists():
        if out_file.is_dir():
            raise CLIError(f"the output path {out_file} is a directory")
        if not force_output:
            with h5py.File(out_file, "r") as h5f:
                base_group = h5f.get(base_group_name, None)
                if base_group is not None:
                    if has_grid(base_group):
                        message = f"a grid result already exists in " \
                                  f"{base_group}"
                        raise CLIError(message)
    else:
        # Create the parent directories.
        out_file.parent.mkdir(exist_ok=True, parents=True)

    def _by_name_order(spec: ParamPartitionSpec):
        """"""
        name = spec.name
        # TODO: Improve performance.
        return param_names.index(name)

    # Sort parameters.
    fixed_specs.sort(key=_by_name_order)
    partition_specs.sort(key=_by_name_order)

    if not partition_specs:
        raise CLIError("no parameter partition has been defined")

    # Table of fixed parameters.
    fixed_params_table = Table(expand=True, pad_edge=True)
    fixed_params_table.add_column("Name", justify="center", ratio=1)
    fixed_params_table.add_column("Value", justify="center", ratio=1)
    # Add param information to table.
    for spec in fixed_specs:
        fixed_params_table.add_row(
            Text(f"{spec.name}", style="bold"),
            Text(f"{spec.value}"),
        )

    # Table of partition parameters.
    grid_specs_table = Table(expand=True)
    grid_specs_table.add_column("Name", justify="center", ratio=1)
    grid_specs_table.add_column("Data Array", justify="center", ratio=6)
    grid_specs_table.add_column("# of Grid Points", justify="center", ratio=1)
    # Add grid information to table.
    with np.printoptions(threshold=50):
        for spec in partition_specs:
            grid_specs_table.add_row(
                Text(str(spec.name), style="bold"),
                console.highlighter(str(spec.data)),
                Text(f"{spec.data.size}"),
            )

    # Display grid evaluation information.
    title_text = Text("chisquarecosmo - Chi-Square Grid Evaluator")
    title_text.stylize("bold red")
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)
    console.print(title_panel, justify="center")

    # Show execution summary.
    console.print(Padding("[underline magenta3]Execution Summary[/]",
                          (1, 0, 1, 0)), justify="center")
    model_text = f"Model: [red bold]{eos_model}[/]"
    console.print(Padding(model_text, (1, 0, 0, 0)), justify="center")
    dataset_text = f"Dataset(s): [red bold]{datasets.label}[/]"
    console.print(Padding(dataset_text), justify="center")
    out_file_text = f"Output file: [red bold]{out_file}[/]"
    console.print(Padding(out_file_text), justify="center")
    hdf5_group_text = f"HDF5 group: [red bold]{base_group_name}[/]"
    console.print(Padding(hdf5_group_text, (0, 0, 1, 0)), justify="center")

    # Show fixed parameters.
    console.print(Padding(f"[magenta3 underline bold]Fixed Parameters[/]",
                          (1, 0, 0, 0)), justify="center")
    console.print(Padding(fixed_params_table, (0, 5, 0, 5)))

    # Show partition parameters.
    console.print(Padding(f"[magenta3 underline bold]Partition Parameters[/]",
                          (1, 0, 0, 0)), justify="center")
    console.print(Padding(grid_specs_table, (0, 5, 0, 5)))

    # Show progress.
    console.print(Padding("Grid Evaluation Progress...", (1, 0, 1, 0)),
                  justify="center")

    # Grid object.
    grid = Grid(eos_model=_eos_model,
                datasets=datasets,
                fixed_params=fixed_specs,
                param_partitions=partition_specs)

    with DaskProgressBar():
        grid_result = grid.eval()

    with h5py.File(out_file, file_mode) as h5f:
        grid_result.save(h5f, base_group_name, force_output)

    console.print(Padding(f"[underline bold green]Optimization process "
                          f"successfully finished[/]",
                          pad=(1, 0)), justify="center")
