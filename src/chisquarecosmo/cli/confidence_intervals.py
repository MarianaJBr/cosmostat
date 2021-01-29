import json
from dataclasses import asdict
from pathlib import Path

import click
import h5py
from chisquarecosmo.chi_square import BestFit, Grid, ParamGrid
from chisquarecosmo.exceptions import CLIError
from chisquarecosmo.util import console
from numpy import ma

# Chi-square delta according to the confidence interval for a single
# parameter.
# TODO: Move to a dedicated module.
CONF_INTERVALS_CHI_SQUARE_DELTA = {
    "1-SIGMA": 1,
    "2-SIGMA": 4,
    "3-SIGMA": 9
}


@click.command()
@click.argument("best-fit-file", type=click.Path(exists=True, dir_okay=False))
@click.argument("grid-file", type=click.Path(exists=True, dir_okay=False))
@click.option("-s", "--save-plots",
              is_flag=True,
              default=False,
              help="Plot the confidence intervals and save the resulting "
                   "figures.")
@click.option("-o", "--output-dir",
              type=click.Path(exists=False),
              required=True,
              help="A directory for storing the output resources, like the "
                   "plots with the confidence intervals. If the directory "
                   "does not exists, it will be created.")
@click.option("-f", "--force-output",
              is_flag=True,
              default=False,
              help="If an output resource already exists in the output "
                   "directory, it will be overwritten.")
@click.option("--as-json",
              is_flag=True,
              help="Report all output as JSON.")
def confidence_intervals(best_fit_file: str,
                         grid_file: str,
                         save_plots: bool,
                         output_dir: str,
                         force_output: bool,
                         as_json: bool):
    """"""
    best_fit_path = Path(best_fit_file).resolve()
    grid_path = Path(grid_file).resolve()
    if output_dir is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_dir).resolve()

    with h5py.File(best_fit_file) as h5_fp:
        best_fit = BestFit.load(h5_fp)

    with h5py.File(grid_file) as h5_fp:
        grid_result = Grid.load(h5_fp)

    best_fit_params = dict(best_fit.params._asdict)
    partition_arrays = grid_result.partition_arrays
    grid_params = list(partition_arrays)
    if not len(grid_params) == 1:
        raise CLIError("the grid must be one-dimensional")
    param_name = grid_params[0]
    best_param = best_fit_params[param_name]
    param_array = partition_arrays[param_name]

    chi_square_data = grid_result.chi_square_data
    chi_square_min = chi_square_data.min()

    chi_square_mask = chi_square_data >= chi_square_min + 10
    masked_chi_square_data = ma.masked_array(chi_square_data,
                                             mask=chi_square_mask)

    # Calculate the confidence intervals
    param_grid = ParamGrid(param_array, masked_chi_square_data)

    confidence_intvals = {}
    chi_square_delta_info = CONF_INTERVALS_CHI_SQUARE_DELTA.items()
    for label, delta in chi_square_delta_info:
        interval = param_grid.get_confidence_interval(delta)
        confidence_intvals[label] = asdict(interval)

    if as_json:
        json_repr = json.dumps(confidence_intvals)
        console.print(json_repr, justify="left")
