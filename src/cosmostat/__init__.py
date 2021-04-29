"""
    cosmostat.

    Python code to estimate chi-square constraints on cosmology models using
    background quantities.
"""
from .cosmology import (
    Dataset,
    DatasetJoin,
    Model,
    Params,
    get_dataset,
    get_dataset_join,
    get_model,
    register_dataset,
    register_dataset_join,
    register_model,
    registered_dataset_joins,
    registered_datasets,
    registered_models,
)
from .data import init_register_datasets
from .likelihood import Likelihood
from .models import init_register_models
from .util import plug_external_models

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

# Export package information.
__version__ = importlib_metadata.version("cosmostat")

#################
# Initialization
#################
# Register built-in datasets.
init_register_datasets()
init_register_models()

# Plug external models. Raise exceptions normally.
# TODO: Needs more testing ⚠.
plug_external_models()

# Exported symbols.
__all__ = [
    "Dataset",
    "DatasetJoin",
    "Likelihood",
    "Model",
    "Params",
    "__version__",
    "get_dataset",
    "get_dataset_join",
    "get_model",
    "plug_external_models",
    "register_dataset",
    "register_dataset_join",
    "register_model",
    "registered_dataset_joins",
    "registered_datasets",
    "registered_models",
]
