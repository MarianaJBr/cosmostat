"""
    chisquarecosmo.

    Python code to estimate chi-square constraints on cosmology models using
    background quantities.
"""

import numpy as np

from .cosmology import (
    Dataset, DatasetJoin, Likelihood, Model, Params, get_dataset,
    get_dataset_join, get_model, register_dataset, register_dataset_join,
    register_model, registered_dataset_joins, registered_datasets,
    registered_models
)
from .data import bao_data, hz_data, sneia_union2_1
from .models import cpl, one
from .util import plug_external_models

#################
# Initialization
#################

# Register built-in models.
register_model(cpl.model)
register_model(one.model)

# Register built-in datasets.
_bao_dataset = bao_data.bao_dataset
_sne_dataset = sneia_union2_1.dataset
_hz_dataset = hz_data.dataset

register_dataset(_bao_dataset)
register_dataset(_sne_dataset)
register_dataset(_hz_dataset)

# CMB Dataset instance holds no real data.
_cmb_dataset = Dataset(name="CMB",
                       label="CMB:Red Planck 2015 (3)",
                       # CMB data has three items, all rubbish.
                       data=np.empty(3),
                       cosmology_func="y_vec_cmb")

# Register dataset unions.
_bao_dataset_join = DatasetJoin([_bao_dataset])
_sne_dataset_join = DatasetJoin([_sne_dataset])
_hz_dataset_join = DatasetJoin([_hz_dataset])
_bao_cmb_dataset_join = DatasetJoin([_bao_dataset, _cmb_dataset])
_bao_sne_hz_dataset_join = DatasetJoin([_bao_dataset,
                                        _sne_dataset,
                                        _hz_dataset])
_total_dataset_join = DatasetJoin([_bao_dataset,
                                   _sne_dataset,
                                   _hz_dataset,
                                   _cmb_dataset])

register_dataset_join(_bao_dataset_join)
register_dataset_join(_sne_dataset_join)
register_dataset_join(_hz_dataset_join)
register_dataset_join(_bao_cmb_dataset_join)
register_dataset_join(_bao_sne_hz_dataset_join)
register_dataset_join(_total_dataset_join)

# Exported symbols.
__all__ = [
    "Dataset",
    "DatasetJoin",
    "Likelihood",
    "Model",
    "Params",
    "get_dataset",
    "get_model",
    "plug_external_models",
    "register_dataset",
    "register_dataset_join",
    "register_model",
    "registered_dataset_joins",
    "registered_datasets",
    "registered_models",
]
