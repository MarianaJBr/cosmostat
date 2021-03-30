import json
import pathlib
from dataclasses import asdict

import click
import h5py
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from cosmostat.chi_square import BestFit, has_best_fit
from cosmostat.exceptions import CLIError
from cosmostat.util import console


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-g",
    "--hdf5-group",
    type=str,
    default="/",
    help="Group where the fitting result is stored. If omitted, "
    "assume it is saved in the root group.",
)
@click.option("--as-json", is_flag=True, help="Report all output as JSON.")
def describe_fit(file: str, hdf5_group: str, as_json: bool):
    """Describe a best-fit result in an existing FILE."""
    file = pathlib.Path(file)
    with h5py.File(file, "r") as h5f:
        group_name = hdf5_group.strip()
        base_group = h5f.get(group_name, None)
        if base_group is None:
            err_msg = f"unable to open group '{group_name}'"
            raise CLIError(err_msg)
        if not has_best_fit(base_group):
            err_msg = f"no best-fit result found in {base_group}"
            raise CLIError(err_msg)
        fit_result = BestFit.load(h5f, group_name)

    if as_json:
        # Report best-fit result as JSON and exit.
        fit_result_dict = asdict(fit_result)
        del fit_result_dict["optimization_info"]
        fit_result_dict["eos_model"] = fit_result.eos_model.name
        fit_result_dict["datasets"] = fit_result.datasets.name
        fit_result_dict["params"] = dict(fit_result.params._asdict())
        json_repr = json.dumps(fit_result_dict)
        console.print(json_repr, justify="left")
        return

    title_text = Text("cosmostat - Best-Fit Result Description")
    title_text.stylize("bold red")
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)
    console.print(title_panel, justify="center")
    file_text = f"Output file: [red bold]{file}[/]"
    console.print(Padding(file_text, (1, 0, 0, 0)), justify="center")
    hdf5_group_text = f"HDF5 group: [red bold]{group_name}[/]"
    console.print(Padding(hdf5_group_text, (0, 0, 1, 0)), justify="center")
    result_text = Padding(console.highlighter(str(fit_result)), (1, 1))
    console.print(result_text, justify="center")
