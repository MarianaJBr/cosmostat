import os
import typing as t
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
from chisquarecosmo import DatasetJoin, Model, Params, get_model
from chisquarecosmo.chi_square import (
    Grid as BaseGrid, GridExecutor as GridExecutorBase,
    GridIterator, PARAM_PARTITIONS_GROUP_LABEL, ROOT_GROUP_NAME,
    T_GridExecutorCallback, fixed_specs_as_array,
    fixed_specs_from_array
)

from .likelihood import Likelihood

PARTITION_GRID_DATASET_LABEL = "DataGrid"
CHI_SQUARE_DATASET_LABEL = "Chi2"


def grid_func_base(params: Params,
                   callback_func: T_GridExecutorCallback,
                   chi_square_funcs: t.List[t.Callable[[Params], float]]):
    """Grid function.

    Evaluates the chi-square function.
    """
    chi_square = sum(func(params) for func in chi_square_funcs)
    if callback_func is not None:
        callback_func(params)
    return chi_square


@dataclass
class PoolGridExecutor(GridExecutorBase):
    """Executes a grid."""

    @staticmethod
    def _make_chi_square_funcs(model: Model,
                               datasets: DatasetJoin):
        """"""
        likelihoods = [Likelihood(model, dataset) for dataset in datasets]
        return [lk.chi_square for lk in likelihoods]

    def _make_grid_func(self, model: Model,
                        datasets: DatasetJoin):
        """Make the function to evaluate on the grid."""
        chi_square_funcs = self._make_chi_square_funcs(model, datasets)
        callback_func = self.callback
        return partial(grid_func_base,
                       callback_func=callback_func,
                       chi_square_funcs=chi_square_funcs)

    def map(self, iterator: GridIterator):
        """Evaluates the chi-square on the grid."""
        grid_size = iterator.size
        grid_func = self._make_grid_func(iterator.eos_model,
                                         iterator.datasets)
        # NOTE: there is no a special reason to use the given
        #  max tasks per child and chunk size values. However,
        #  we set maxtasksperchild=2 to avoid python using a
        #  lot of memory.
        # TODO: Choose a better value for chunksize and maxtasksperchild
        #  arguments.
        cpu_count = os.cpu_count()
        chunk_size = min(grid_size // cpu_count, 64)
        with Pool(maxtasksperchild=2) as pool_exec:
            pool_imap = pool_exec.imap(grid_func, iterator,
                                       chunksize=chunk_size)
            yield from pool_imap


@dataclass
class Grid(BaseGrid):
    """Represent a chi-square evaluation over a parameter grid."""

    def save(self, file: h5py.File,
             group_name: str = None,
             force: bool = False):
        """Save a grid result to an HDF5 file."""
        group = file[ROOT_GROUP_NAME]

        # Save the attributes that define the result.
        group.attrs["eos_model"] = self.eos_model.name
        group.attrs["datasets"] = self.datasets.name
        group.attrs["datasets_label"] = self.datasets.label
        group.attrs["fixed_params"] = fixed_specs_as_array(self.fixed_params)

        # Create a group to save the grid partition arrays.
        arrays_group = group.create_group(PARAM_PARTITIONS_GROUP_LABEL)
        for name, data in self.partition_arrays.items():
            arrays_group.create_dataset(name, data=data)

        # Save the chi-square grid data.
        partition_names = self.partition_arrays.keys()
        grid_arrays = self.partition_arrays.values()
        grid_shape = tuple(data.size for data in grid_arrays)
        grid_matrices = np.meshgrid(*grid_arrays,
                                    indexing="ij",
                                    sparse=True)
        partition_matrices = dict(zip(partition_names, grid_matrices))
        grid_dataset = group.create_dataset(PARTITION_GRID_DATASET_LABEL,
                                            shape=grid_shape,
                                            dtype="f8")
        params_names = getattr(self.eos_model.params_cls, "_fields")
        for idx, name in params_names:
            # Save the parameter values in the dataset.
            if name in partition_matrices:
                grid_dataset[idx] = partition_matrices[name]
            else:
                grid_dataset[idx] = self.fixed_params[name]

        # Save the chi-square grid data.
        group.create_dataset(CHI_SQUARE_DATASET_LABEL,
                             data=self.chi_square_data)

    @classmethod
    def load(cls, file: h5py.File,
             group_name: str = None):
        """Load a grid result from an HDF5 file."""
        group: h5py.Group = file["/"]

        # Load grid result attributes.
        group_attrs = dict(group.attrs)
        eos_model = get_model(group_attrs["eos_model"])
        datasets = DatasetJoin.from_name(group_attrs["datasets"])
        fixed_params = fixed_specs_from_array(group_attrs["fixed_params"])

        # Load partition arrays.
        partition_items = []
        arrays_group: h5py.Group = group[PARAM_PARTITIONS_GROUP_LABEL]
        arrays_group_keys = list(arrays_group.keys())
        for key in arrays_group_keys:
            item = arrays_group[key]
            # Load numpy arrays from data sets.
            if isinstance(item, h5py.Dataset):
                name = key
                data = item[()]
                partition_items.append((name, data))

        # Load chi-square data.
        chi_square = group[CHI_SQUARE_DATASET_LABEL][()]

        # Make a new instance.
        return cls(eos_model=eos_model,
                   datasets=datasets,
                   fixed_params=fixed_params,
                   partition_arrays=dict(partition_items),
                   chi_square_data=chi_square)
