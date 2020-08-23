"""
    chisquarecosmo.

    Python code to estimate chi-square constraints on cosmology models using
    background quantities.
"""

import numpy as np

from .cosmology import (
    Dataset, DatasetUnion, register_dataset,
    register_dataset_union, register_model
)
from .data import bao_data, hz_data, sneia_union2_1
from .models import cpl

#################
# Initialization
#################
__all__ = []

# Register built-in models.
register_model(cpl.model)

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
bao_dataset = DatasetUnion([_bao_dataset])
sne_dataset = DatasetUnion([_sne_dataset])
hz_dataset = DatasetUnion([_hz_dataset])
bao_cmb_dataset = DatasetUnion([_bao_dataset, _cmb_dataset])
bao_sne_hz_dataset = DatasetUnion([_bao_dataset, _sne_dataset, _hz_dataset])
total_dataset = DatasetUnion([_bao_dataset,
                              _sne_dataset,
                              _hz_dataset,
                              _cmb_dataset])

register_dataset_union(bao_dataset)
register_dataset_union(sne_dataset)
register_dataset_union(hz_dataset)
register_dataset_union(bao_cmb_dataset)
register_dataset_union(bao_sne_hz_dataset)
register_dataset_union(total_dataset)
