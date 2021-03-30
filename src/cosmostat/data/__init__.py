import numpy as np

from cosmostat import (Dataset, DatasetJoin, register_dataset,
                       register_dataset_join)

from . import bao_data, hz_data, sneia_union2_1


def init_register_datasets():
    """Function for registering the built-in datasets."""

    # Register built-in datasets.
    _bao_dataset = bao_data.bao_dataset
    _sne_dataset = sneia_union2_1.dataset
    _hz_dataset = hz_data.dataset

    # CMB Dataset instance holds no real data.
    _cmb_dataset = Dataset(
        name="CMB",
        label="CMB:Red Planck 2015 (3)",
        # CMB data has three items, all rubbish.
        data=np.empty(3),
        cosmology_func="y_vec_cmb",
    )

    register_dataset(_bao_dataset)
    register_dataset(_sne_dataset)
    register_dataset(_hz_dataset)
    register_dataset(_cmb_dataset)

    # Register dataset unions.
    _bao_dataset_join = DatasetJoin([_bao_dataset])
    _sne_dataset_join = DatasetJoin([_sne_dataset])
    _hz_dataset_join = DatasetJoin([_hz_dataset])
    _bao_cmb_dataset_join = DatasetJoin([_bao_dataset, _cmb_dataset])
    _bao_sne_hz_dataset_join = DatasetJoin(
        [_bao_dataset, _sne_dataset, _hz_dataset]
    )
    _total_dataset_join = DatasetJoin(
        [_bao_dataset, _sne_dataset, _hz_dataset, _cmb_dataset]
    )

    register_dataset_join(_bao_dataset_join)
    register_dataset_join(_sne_dataset_join)
    register_dataset_join(_hz_dataset_join)
    register_dataset_join(_bao_cmb_dataset_join)
    register_dataset_join(_bao_sne_hz_dataset_join)
    register_dataset_join(_total_dataset_join)
