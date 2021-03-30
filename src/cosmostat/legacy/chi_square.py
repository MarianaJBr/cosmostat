import os
import typing as t
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
from cosmostat import DatasetJoin, Model, Params, get_model
from cosmostat.chi_square import (
    BestFitFinder as BaseBestFitFinder, FreeParamSpec, Grid as BaseGrid,
    GridExecutor as GridExecutorBase, GridIterator,
    PARAM_PARTITIONS_GROUP_LABEL, ROOT_GROUP_NAME,
    T_BestFitFinderCallback, T_FixedParamSpecs, T_GridExecutorCallback,
    fixed_specs_as_array, fixed_specs_from_array
)
from cosmostat.likelihood import T_LikelihoodFunc
from scipy.optimize import differential_evolution

from .likelihood import Likelihood

PARTITION_GRID_DATASET_LABEL = "DataGrid"
CHI_SQUARE_DATASET_LABEL = "Chi2"


def chi_square_base(params: Params,
                    chi_square_funcs: t.List[T_LikelihoodFunc]):
    """Total chi-square function."""
    try:
        return sum((func(params) for func in chi_square_funcs))
    except ValueError:
        return np.inf


def objective_func_base(x: t.Tuple[float, ...],
                        params_cls: t.Type[Params],
                        chi_square_func: T_LikelihoodFunc,
                        fixed_specs: T_FixedParamSpecs,
                        free_specs: t.List[FreeParamSpec]):
    """Objective function.

    Evaluates the chi-square function.
    """
    fixed_specs_dict = {spec.name: spec.value for spec in fixed_specs}
    var_names = [spec.name for spec in free_specs]
    params_dict = dict(zip(var_names, x))
    params_dict.update(fixed_specs_dict)
    params = params_cls(**params_dict)
    return chi_square_func(params)


def callback_base(x: t.Tuple[float, ...],
                  convergence: float = None, *,
                  params_cls: t.Type[Params],
                  chi_square_func: T_LikelihoodFunc,
                  fixed_specs: T_FixedParamSpecs,
                  free_specs: t.List[FreeParamSpec],
                  callback_func: T_BestFitFinderCallback):
    """Callback for SciPy optimizer."""
    fixed_specs_dict = {spec.name: spec.value for spec in fixed_specs}
    var_names = [spec.name for spec in free_specs]

    params_dict = dict(zip(var_names, x))
    params_dict.update(fixed_specs_dict)
    params = params_cls(**params_dict)
    chi_square = chi_square_func(params)
    return callback_func(params, chi_square)


@dataclass
class DEBestFitFinder(BaseBestFitFinder):
    """Find the best chi-square fitting params of a model to certain data."""

    @property
    def chi_square_funcs(self):
        """"""
        model = self.eos_model
        datasets = self.datasets
        likelihoods = [Likelihood(model, dataset) for dataset in datasets]
        return [lk.chi_square for lk in likelihoods]

    def _make_chi_square_func(self):
        """Get the chi-square function."""
        return partial(chi_square_base,
                       chi_square_funcs=self.chi_square_funcs)

    def _make_objective_func(self):
        """"""
        params_cls = self.eos_model.params_cls
        fixed_specs = self.fixed_specs
        free_specs = self.free_specs
        chi_square_func = self.chi_square
        return partial(objective_func_base,
                       params_cls=params_cls,
                       chi_square_func=chi_square_func,
                       fixed_specs=fixed_specs,
                       free_specs=free_specs)

    def _make_callback_func(self):
        """"""
        params_cls = self.eos_model.params_cls
        fixed_specs = self.fixed_specs
        free_specs = self.free_specs
        chi_square_func = self.chi_square
        _callback_func = self.callback

        if _callback_func is None:
            return None

        return partial(callback_base,
                       params_cls=params_cls,
                       chi_square_func=chi_square_func,
                       fixed_specs=fixed_specs,
                       free_specs=free_specs,
                       callback_func=_callback_func)

    def _exec(self):
        """Optimization routine."""
        # Start the optimization procedure.
        return differential_evolution(self.objective_func,
                                      bounds=self.free_specs_bounds,
                                      callback=self.callback_func,
                                      polish=True)


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
        if PARAM_PARTITIONS_GROUP_LABEL in group and force:
            del group[PARAM_PARTITIONS_GROUP_LABEL]
        if PARTITION_GRID_DATASET_LABEL in group and force:
            del group[PARTITION_GRID_DATASET_LABEL]
        if CHI_SQUARE_DATASET_LABEL in group and force:
            del group[CHI_SQUARE_DATASET_LABEL]

        # Save the attributes that define the result.
        fixed_params = self.fixed_params
        group.attrs["eos_model"] = self.eos_model.name
        group.attrs["datasets"] = self.datasets.name
        group.attrs["datasets_label"] = self.datasets.label
        group.attrs["fixed_params"] = fixed_specs_as_array(fixed_params)

        # Create a group to save the grid partition arrays.
        arrays_group = group.create_group(PARAM_PARTITIONS_GROUP_LABEL)
        for name, data in self.partition_arrays.items():
            arrays_group.create_dataset(name, data=data)

        # Save the chi-square grid data.
        params_names = getattr(self.eos_model.params_cls, "_fields")
        partition_names = self.partition_arrays.keys()
        grid_arrays = self.partition_arrays.values()
        grid_shape = tuple(data.size for data in grid_arrays)
        grid_shape += (len(params_names),)
        grid_matrices = np.meshgrid(*grid_arrays,
                                    indexing="ij",
                                    sparse=True)
        partition_matrices = dict(zip(partition_names, grid_matrices))
        grid_dataset = group.create_dataset(PARTITION_GRID_DATASET_LABEL,
                                            shape=grid_shape,
                                            dtype="f8")
        for idx, name in enumerate(params_names):
            # Save the parameter values in the dataset.
            if name in partition_names:
                grid_dataset[..., idx] = partition_matrices[name]
            else:
                grid_dataset[..., idx] = fixed_params[name]

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
