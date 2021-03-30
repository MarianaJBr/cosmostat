from dataclasses import dataclass, field
from functools import partial

from cosmostat import Dataset, Model, Params
from cosmostat.likelihood import T_LikelihoodFunc

from .Statistics.chi2_likelihood import chi2BAO, chi2CMB, chi2hz, chi2SNe


def get_cpl_params(params: Params):
    """"""
    w_params = params[:2]
    cosmo_params = params[2:]
    return w_params, cosmo_params


def get_one_params(params: Params):
    """"""
    w_params = params[:3]
    cosmo_params = params[3:]
    return w_params, cosmo_params


chi_square_functions_table = {
    "BAO": chi2BAO,
    "Hz": chi2hz,
    "SNe": chi2SNe,
    "CMB": chi2CMB,
}


def chi_square_base(params: Params, model_name: str, dataset_name: str):
    """"""
    # A inelegant way...
    if model_name == "ONE":
        w_params, cosmo_params = get_one_params(params)
    elif model_name == "CPL":
        w_params, cosmo_params = get_cpl_params(params)
    else:
        # TODO: Define a custom exception.
        raise ValueError("model name is unknown or is not implemented")

    # Evaluate the corresponding chi-square function.
    chi_square_func = chi_square_functions_table[dataset_name]
    return chi_square_func(w_params, cosmo_params)


@dataclass
class Likelihood:
    """Group likelihood functions for a specific model and dataset."""

    model: Model
    dataset: Dataset

    # Protected attributes.
    chi_square: T_LikelihoodFunc = field(init=False, default=None)

    def __post_init__(self):
        """"""
        model_name = self.model.name
        dataset_name = self.dataset.name
        self.chi_square = partial(
            chi_square_base, model_name=model_name, dataset_name=dataset_name
        )
