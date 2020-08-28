import pathlib
import typing as t

import click
import h5py
import numpy as np
from chisquarecosmo.chi_square import (
    FixedParamSpec, FreeParamSpec, find_best_fit, has_best_fit
)
from chisquarecosmo.cosmology import (
    get_dataset_join, get_model, registered_dataset_joins,
    registered_models
)
from chisquarecosmo.exceptions import CLIError
from chisquarecosmo.util import console, plug_external_models
from click import BadParameter
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from toolz import groupby

# By default, the routine saves the best-fit results in this file.
# Its full path is relative the current working directory.
DEFAULT_BEST_FIT_OUT_FILE = "chi-square-fit.h5"

T_CLIParam = t.Tuple[str, str]
T_CLIParams = t.Tuple[T_CLIParam, ...]
T_FitParamSpec = t.Union[t.Type[FixedParamSpec], t.Type[FreeParamSpec]]
T_FitParamSpecs = t.Dict[T_FitParamSpec, t.List]

# Plug external models. Raise exceptions normally.
# TODO: Needs more testing ⚠.
plug_external_models()


def validate_param(param: T_CLIParam):
    """Validates a single parameter."""
    name, value_str = param
    value_parts = value_str.split(":", maxsplit=1)
    num_parts = len(value_parts)
    if num_parts != 2:
        if num_parts == 1:
            value = value_parts[0]
            try:
                value = float(value)
            except ValueError:
                err_msg = f"invalid value '{value}' for parameter " \
                          f"'{name}'"
                raise BadParameter(err_msg)
            return FixedParamSpec(name, value)
        else:
            err_msg = f"invalid parameter bounds '{value_str}' for parameter " \
                      f"'{name}'"
            raise BadParameter(err_msg)
    lower, upper = value_parts
    if not lower:
        lower = -np.inf
    else:
        try:
            lower = float(lower)
        except ValueError:
            err_msg = f"invalid lower bound '{lower}' for parameter " \
                      f"'{name}'"
            raise BadParameter(err_msg)
    if not upper:
        upper = np.inf
    else:
        try:
            upper = float(upper)
        except ValueError:
            err_msg = f"invalid upper bound '{upper}' for parameter " \
                      f"'{name}'"
            raise BadParameter(err_msg)
    if lower == upper:
        return FixedParamSpec(name, lower)
    return FreeParamSpec(name, lower, upper)


def validate_params(ctx, param, values: T_CLIParams):
    """Validates the parameter list for the chi2 function."""
    return groupby(type, map(validate_param, values))


# Use these values as possible choices for EOS_MODEL and DATASET arguments.
_models = registered_models()
_datasets = registered_dataset_joins()


@click.command()
@click.argument("eos_model", type=click.Choice(_models), metavar="EOS_MODEL")
@click.argument("datasets", type=click.Choice(_datasets), metavar="DATASETS")
@click.option("--param",
              type=(str, str),
              multiple=True,
              default=None,
              callback=validate_params,
              help="Defines a parameter for the minimization function, as well "
                   "as its value or bounds. The first text element indicates "
                   "the parameter name, and it is dependent on the equation "
                   "of stated used in the minimization process. On the one "
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
def chi_square_fit(eos_model: str, datasets: str, param: T_FitParamSpecs,
                   output: str, hdf5_group: str, force_output: bool):
    """Make a chi-square fitting of a EOS_MODEL to certain DATASETS.

    EOS_MODEL is the name of the model/equation of state.

    DATASETS is one or more dataset which the model should be fitted against to.
    """
    out_file = pathlib.Path(output).resolve()
    base_group_name = hdf5_group.strip() or "/"
    # Append to existing file, or create a new one.
    file_mode = "a" if out_file.exists() else "w"
    _eos_model = get_model(eos_model)
    datasets = get_dataset_join(datasets)
    params_cls = _eos_model.params_cls
    fixed_specs: t.List[FixedParamSpec] = param.get(FixedParamSpec, [])
    free_specs: t.List[FreeParamSpec] = param.get(FreeParamSpec, [])

    # Parameters defined for the current model/eos.
    param_names = list(params_cls._fields)
    defaults_dict = params_cls._field_defaults
    names_with_defaults = set(defaults_dict.keys())
    fixed_spec_name_set = set(spec.name for spec in fixed_specs)
    free_spec_name_set = set(spec.name for spec in free_specs)
    spec_name_set = fixed_spec_name_set | free_spec_name_set
    required_names = [name for name in param_names if
                      name not in names_with_defaults]
    missing_names = [name for name in required_names if
                     name not in spec_name_set]
    unknown_names = [name for name in spec_name_set if name not in param_names]

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

    # Perform checks.
    if not force_output and out_file.exists():
        with h5py.File(out_file, "r") as h5f:
            base_group = h5f.get(base_group_name, None)
            if base_group is not None:
                if has_best_fit(base_group):
                    message = f"a best-fit result already exists in " \
                              f"{base_group}"
                    raise CLIError(message)

    def _by_name_order(spec: FreeParamSpec):
        """"""
        name = spec.name
        # TODO: Improve performance.
        return param_names.index(name)

    # Get all parameters that are fixed by default. These parameters
    # automatically are best fit parameters.
    fixed_spec_name_set = (names_with_defaults - free_spec_name_set -
                           fixed_spec_name_set)
    fixed_specs.extend([
        FixedParamSpec(name, defaults_dict[name]) for name in
        fixed_spec_name_set
    ])
    # Sort parameters.
    fixed_specs.sort(key=_by_name_order)
    free_specs.sort(key=_by_name_order)
    free_spec_names = [spec.name for spec in free_specs]
    fixed_specs_dict = {spec.name: spec.value for spec in fixed_specs}

    # Table of fixed parameters.
    fixed_params_table = Table(expand=True, pad_edge=True)
    fixed_params_table.add_column("Name", justify="center", ratio=1)
    fixed_params_table.add_column("Value", justify="center", ratio=1)
    # Add param information to table.
    for spec in fixed_specs:
        fixed_params_table.add_row(
            Text(f"{spec.name}", style="bold blue"),
            Text(f"{spec.value}"),
        )

    # Table of variable parameters.
    var_params_table = Table(expand=True)
    var_params_table.add_column("Name", justify="center", ratio=1)
    var_params_table.add_column("Lower Bound", justify="center", ratio=1)
    var_params_table.add_column("Upper Bound", justify="center", ratio=1)
    # Add param information to table.
    for spec in free_specs:
        var_params_table.add_row(
            Text(str(spec.name), style="bold blue"),
            Text(f"{spec.lower_bound}"),
            Text(f"{spec.upper_bound}")
        )

    # Display minimization process information.
    title_text = Text("chisquarecosmo - Chi-Square Minimizer")
    title_text.stylize("bold red")
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)
    console.print(title_panel, justify="center")
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
    console.print(Padding(f"[magenta3 underline bold]Fixed Parameters[/]",
                          (1, 0, 0, 0)), justify="center")
    console.print(Padding(fixed_params_table, (0, 5, 0, 5)))
    console.print(Padding(f"[magenta3 underline bold]Variable Parameters[/]",
                          (1, 0, 0, 0)), justify="center")
    console.print(Padding(var_params_table, (0, 5, 0, 5)))
    console.print(Padding("Best-Fit Params Progress...", (1, 0, 1, 0)),
                  justify="center")

    def optimization_callback(x: t.Tuple[float, ...],
                              convergence: float = None):
        """Show a progress message for each iteration."""
        params_dict = dict(zip(free_spec_names, x))
        params_dict.update(fixed_specs_dict)
        params_obj = params_cls(**params_dict)
        console.print("---", justify="center")
        text = f"[bold deep_sky_blue1]{params_obj}[/]"
        console.print(Padding(text, (0, 1, 0, 1)), justify="center")

    best_fit_result = find_best_fit(_eos_model,
                                    datasets,
                                    fixed_specs,
                                    free_specs,
                                    callback=optimization_callback)
    with h5py.File(out_file, file_mode) as h5f:
        best_fit_result.save(h5f, base_group_name, force_output)

    # Print result information.
    console.print()
    console.print(f"[bold underline]Optimization Result[/]", justify="center")
    best_fit_msg = Padding(f"[yellow]{best_fit_result}[/]", (0, 1, 0, 1))
    console.print(best_fit_msg, justify="center")
    result_msg = Padding(f"[bold underline]Result saved in file[/]",
                         (1, 0, 0, 0))
    console.print(result_msg, justify="center")
    file_path_msg = Padding(f"[yellow]{out_file.absolute()}[/]", (0, 1, 0, 1))
    console.print(file_path_msg, justify="center")
    success_text = "Optimization process successfully finished"
    success_msg = Padding(f"[underline bold green]{success_text}[/]",
                          (1, 0, 1, 0))
    console.print(success_msg, justify="center")
