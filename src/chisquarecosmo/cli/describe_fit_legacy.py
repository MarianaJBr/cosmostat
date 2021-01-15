import json
import pathlib

import click
import numpy
from chisquarecosmo import (
    get_model, registered_models
)
from chisquarecosmo.exceptions import CLIError
from chisquarecosmo.util import console
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

# Use these values as possible choices for EOS_MODEL and DATASET arguments.
_all_models = registered_models()


@click.command()
@click.argument("eos_model",
                type=click.Choice(_all_models),
                metavar="EOS_MODEL")
@click.argument("file",
                type=click.Path(exists=True, dir_okay=False))
@click.option("--as-json",
              is_flag=True,
              help="Report all output as JSON.")
def describe_fit_legacy(eos_model: str, file: str, as_json: bool):
    """Describe a "legacy" best-fit result in an existing FILE.

    EOS_MODEL is the name of the model/equation of state fitted.
    """
    # Parameters defined for the current model/eos.
    _eos_model = get_model(eos_model)
    params_cls = _eos_model.params_cls
    param_names = list(getattr(params_cls, "_fields"))

    # Load data
    file = pathlib.Path(file).resolve()
    try:
        data = numpy.loadtxt(file)
    except ValueError:
        raise CLIError("the best-fit file contains invalid data")

    # Extract data manually and build a dictionary with this information.
    eos_today = data[0]
    chi_square_min = data[1]
    chi_square_reduced = data[2]
    num_params = len(param_names)
    params = _eos_model.params_cls(*data[3:3 + num_params])
    omega_m = data[3 + num_params]
    aic = data[3 + num_params + 1]
    bic = data[3 + num_params + 2]
    fit_result = dict(
        params=dict(params._asdict()),
        eos_today=eos_today,
        chi_square_min=chi_square_min,
        chi_square_reduced=chi_square_reduced,
        omega_m=omega_m,
        aic=aic,
        bic=bic
    )

    if as_json:
        # Report best-fit result as JSON and exit.
        json_repr = json.dumps(fit_result)
        console.print(json_repr, justify="left")
        return

    # Display information with rich formatting.
    title_text = Text("chisquarecosmo - Best-Fit Result Description (Legacy)")
    title_text.stylize("bold red")
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)
    console.print(title_panel, justify="center")
    file_text = f"Output file: [red bold]{file}[/]"
    console.print(Padding(file_text, (1, 0, 0, 0)), justify="center")
    fit_result_text = Pretty(fit_result,
                             highlighter=console.highlighter,
                             justify="left")
    result_text = Padding(fit_result_text, (1, 1))
    console.print(result_text, justify="center")
