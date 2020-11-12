import os
import typing as t
from dataclasses import astuple, dataclass
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
from chisquarecosmo import DatasetJoin, Params, get_model
from chisquarecosmo.chi_square import (
    Grid as BaseGrid, GridExecutor as GridExecutorBase,
    PARAM_PARTITIONS_GROUP_LABEL, ROOT_GROUP_NAME, T_ParamPartitionSpecs,
    fixed_specs_as_array, fixed_specs_from_array
)

from .likelihood import Likelihood

PARTITION_GRID_DATASET_LABEL = "DataGrid"
CHI_SQUARE_DATASET_LABEL = "Chi2"


def grid_func_base(value_indexes: t.Tuple[int, ...],
                   param_partition: T_ParamPartitionSpecs,
                   free_names: t.List[str],
                   fixed_params_dict: t.Dict[str, float],
                   params_cls: t.Type[Params],
                   chi_square_funcs: t.List[t.Callable[[Params], float]]):
    """Grid function.

    Evaluates the chi-square function.
    """
    grid_values = [param_partition[grid_idx].data[value_idx] for
                   grid_idx, value_idx in enumerate(value_indexes)]
    params_dict = dict(zip(free_names, grid_values))
    params_dict.update(fixed_params_dict)
    params_obj = params_cls(**params_dict)
    return sum(func(params_obj) for func in chi_square_funcs)


@dataclass
class GridExecutor(GridExecutorBase):
    """Executes a grid."""

    def __post_init__(self):
        """Post-initialization.

        Set private attributes and other related initialization tasks.
        """
        model = self.eos_model
        datasets = self.datasets
        param_partitions = self.param_partitions
        param_partition_names = [spec.name for spec in param_partitions]
        likelihoods = [Likelihood(model, dataset) for dataset in datasets]
        # Set private attributes.
        self.param_partition_names = param_partition_names
        self.likelihoods = likelihoods

    def _make_grid_func(self):
        """Make the function to evaluate on the grid."""
        eos_model = self.eos_model
        likelihoods = self.likelihoods
        fixed_params = self.fixed_params
        free_names = self.param_partition_names
        param_partition = self.param_partitions
        params_cls = eos_model.params_cls
        chi_square_funcs = [lkh.chi_square for lkh in likelihoods]
        fixed_params_dict = {spec.name: spec.value for spec in fixed_params}

        return partial(grid_func_base,
                       param_partition=param_partition,
                       free_names=free_names,
                       fixed_params_dict=fixed_params_dict,
                       params_cls=params_cls,
                       chi_square_funcs=chi_square_funcs)

    def eval(self):
        """Evaluates the chi-square on the grid."""
        grid_func = self._make_grid_func()
        param_partitions = self.param_partitions
        grid_shape = tuple(
            partition.data.size for partition in param_partitions)
        grid_size = int(np.prod(grid_shape))

        # Evaluate the grid using a multidimensional iterator. This
        # way we do not allocate memory for all the combinations of
        # parameter values that form the grid.
        grid_indexes = np.ndindex(*grid_shape)
        num_processes = os.cpu_count()
        # NOTE: there is no a special reason to use this chunk size.
        chunk_size = max(grid_size // num_processes, 16)
        chi_square_data = []
        with Pool() as pool_exec:
            results_imap = pool_exec.imap(grid_func, grid_indexes,
                                          chunksize=chunk_size)
            for idx, grid_elem in enumerate(results_imap):
                chi_square_data.append(grid_elem)
                yield idx
        chi_square_array = np.asarray(chi_square_data).reshape(grid_shape)
        fixed_params_dict = {spec.name: spec.value for spec in
                             self.fixed_params}
        # The parameter order used to evaluate the grid is defined by the
        # param_partition list order. Converting to a dictionary as follows
        # preserves this order since, in Python>=3.7, dictionaries keep their
        # items sorted according to their insertion order.
        partition_arrays = dict(map(astuple, param_partitions))
        return Grid(eos_model=self.eos_model,
                    datasets=self.datasets,
                    fixed_params=fixed_params_dict,
                    partition_arrays=partition_arrays,
                    chi_square_data=chi_square_array)

    def __iter__(self):
        raise NotImplementedError


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
