import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import astuple, dataclass, field
from functools import partial

import h5py
import numpy as np
from dask import bag
from scipy.interpolate import UnivariateSpline
from scipy.optimize import (
    OptimizeResult, differential_evolution,
    minimize_scalar, newton
)

from .cosmology import (
    DatasetJoin, Model, Params, get_model
)
from .likelihood import Likelihood, T_LikelihoodFunc

# Name of the HDF5 groups used to store best-fit and grid results.
ROOT_GROUP_NAME = "/"
BEST_FIT_GROUP_LABEL = "best_fit"
OPT_INFO_GROUP_LABEL = "optimization_info"
GRID_GROUP_LABEL = "grid"
PARAM_PARTITIONS_GROUP_LABEL = "param_partitions"

# Optimization info type. We expect SciPy OptimizeResult instances,
# objects with a Mapping interface, or dictionaries.
T_OptimizationInfo = t.Union[OptimizeResult, t.Mapping[str, t.Any], dict]


@dataclass
class FreeParamSpec:
    """Description of an optimization parameter."""
    name: str
    lower_bound: float
    upper_bound: float

    @property
    def fixed(self):
        if self.upper_bound == self.lower_bound:
            return True


@dataclass
class ParamPartitionSpec:
    """Describe a parameter partition."""
    name: str
    data: np.ndarray

    @classmethod
    def from_range_spec(cls, name: str,
                        lower_bound: float,
                        upper_bound: float,
                        num_parts: int,
                        scale: str = "linear",
                        base: int = None):
        """Create a partition spec from a range specification."""
        lower = lower_bound
        upper = upper_bound
        num_values = num_parts + 1
        if scale == "linear":
            data = np.linspace(lower, upper, num=num_values)
        elif scale == "geom":
            data = np.geomspace(lower, upper, num=num_values)
        elif scale == "log":
            base = base or 10
            data = np.logspace(lower, upper, num=num_values, base=base)
        else:
            raise
        return cls(name=name, data=np.asarray(data))


@dataclass
class FixedParamSpec:
    """"""
    name: str
    value: float


T_ParamPartitionSpecs = t.List[ParamPartitionSpec]
T_FixedParamSpecs = t.List[FixedParamSpec]


@dataclass
class BestFit:
    """Represent the result of a SciPy optimization routine."""
    eos_model: Model
    datasets: DatasetJoin
    params: Params
    free_params: t.List[FreeParamSpec]
    eos_today: float
    chi_square_min: float
    chi_square_reduced: float
    omega_m: float
    aic: float = None
    bic: float = None
    optimization_info: T_OptimizationInfo = None

    def save(self, file: h5py.File, group_name: str = None,
             force: bool = False):
        """Save a best-fit result to an HDF5 file."""
        if group_name is None or group_name == ROOT_GROUP_NAME:
            base_group = file[ROOT_GROUP_NAME]
        else:
            base_group = file.get(group_name, None)
            if base_group is None:
                base_group = file.create_group(group_name)
        if BEST_FIT_GROUP_LABEL in base_group and force:
            del base_group[BEST_FIT_GROUP_LABEL]
        group = base_group.create_group(BEST_FIT_GROUP_LABEL)

        # Save best fit parameters to group.
        params = self.params
        free_params = self.free_params
        group.attrs["eos_model"] = self.eos_model.name
        group.attrs["datasets"] = self.datasets.name
        group.attrs["datasets_label"] = self.datasets.label
        group.attrs["params"] = best_fit_params_as_array(params)
        group.attrs["free_params"] = list(
            map(free_spec_as_array, free_params))
        group.attrs["eos_today"] = self.eos_today
        group.attrs["chi_square_min"] = self.chi_square_min
        group.attrs["chi_square_reduced"] = self.chi_square_reduced
        group.attrs["omega_m"] = self.omega_m
        if self.bic is not None:
            group.attrs["bic"] = self.bic
        if self.aic is not None:
            group.attrs["aic"] = self.aic

        # Save SciPy Optimization result in a subgroup.
        opt_info = self.optimization_info
        if opt_info is not None:
            opt_info_group = group.create_group(OPT_INFO_GROUP_LABEL)
            self.save_opt_info(opt_info_group, opt_info)

    @classmethod
    def load(cls, file: h5py.File, group_name: str = None):
        """Load a best-fit result from an HDF5 file."""
        base_group: h5py.Group = file["/"] if group_name is None else \
            file[group_name]
        group = base_group[BEST_FIT_GROUP_LABEL]

        # Load best-fit result attributes.
        group_attrs = dict(group.attrs)
        eos_model = group_attrs.pop("eos_model")
        model = get_model(eos_model)
        params_cls = model.params_cls
        params_data = group_attrs.pop("params")
        params = best_fit_params_from_array(params_data, params_cls)
        free_params_data = group_attrs.pop("free_params")
        free_params = list(map(free_spec_from_array, free_params_data))
        datasets = DatasetJoin.from_name(group_attrs["datasets"])
        eos_today = group_attrs["eos_today"]
        chi_square_min = group.attrs["chi_square_min"]
        chi_square_reduced = group.attrs["chi_square_reduced"]
        omega_m = group.attrs["omega_m"]
        bic = group_attrs.pop("bic", None)
        aic = group_attrs.pop("aic", None)

        # Load SciPy Optimization result in a subgroup.
        if OPT_INFO_GROUP_LABEL in group:
            opt_info_group = group[OPT_INFO_GROUP_LABEL]
            opt_info = cls.load_opt_info(opt_info_group)
        else:
            opt_info = None
        return cls(eos_model=model,
                   datasets=datasets,
                   params=params,
                   free_params=free_params,
                   eos_today=eos_today,
                   chi_square_min=chi_square_min,
                   chi_square_reduced=chi_square_reduced,
                   omega_m=omega_m,
                   aic=aic, bic=bic,
                   optimization_info=opt_info)

    @staticmethod
    def save_opt_info(group: h5py.Group, opt_info: T_OptimizationInfo):
        """Saves an OptimizeResult instance to an HDF5 group."""
        opt_info = dict(opt_info)  # Create a copy.
        opt_result_keys = list(opt_info.keys())
        # First, save numpy arrays as data sets.
        for key in opt_result_keys:
            value = opt_info[key]
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
                opt_info.pop(key)
        # Save the rest of items as group attributes
        group.attrs.update(opt_info)

    @staticmethod
    def load_opt_info(group: h5py.Group):
        """Loads an OptimizeResult instance from an HDF5 group."""
        # Load group attributes
        opt_result_data = dict(group.attrs)  # Create a copy.
        opt_result_keys = list(group.keys())
        for key in opt_result_keys:
            item = group[key]
            # Load numpy arrays from data sets.
            if isinstance(item, h5py.Dataset):
                opt_result_data[key] = item[()]
        # Convert the optimization info to an OptimizeResult instance.
        return opt_result_data


def _bounds(spec: FreeParamSpec):
    """"""
    return spec.lower_bound, spec.upper_bound


def _is_fixed(spec: FreeParamSpec):
    """"""
    return spec.fixed


def _is_none(_obj: t.Any):
    """Test if an arbitrary object is None."""
    return _obj is None


# Define the dtype we use to save the best-fit params in an HDF5 file.
_best_fit_params_dtype = np.dtype([
    ("name", h5py.string_dtype()), ("value", "f8")
])

# Define the dtype we use to save a fixed param in an HDF5 file.
_fixed_param_dtype = np.dtype([
    ("name", h5py.string_dtype()), ("value", "f8")
])

# Define the dtype we use to save a free param in an HDF5 file.
_free_param_dtype = np.dtype([
    ("name", h5py.string_dtype()),
    ("lower_bound", "f8"),
    ("upper_bound", "f8")
])

# Define the dtype we use to save a grid param in an HDF5 file.
_grid_param_dtype = np.dtype([
    ("name", h5py.string_dtype()),
    ("lower_bound", "f8"),
    ("upper_bound", "f8"),
    ("num_parts", "i4"),
    ("scale", h5py.string_dtype()),
    ("base", "i4"),
])


def best_fit_params_as_array(param: Params):
    """Convert the best-fit parameters to a numpy array."""
    param_items = list(param._asdict().items())
    return np.array(param_items, dtype=_best_fit_params_dtype)


def best_fit_params_from_array(data: np.ndarray, params_cls: t.Type[Params]):
    """Retrieve the best-fit parameters from a numpy array."""
    params_dict = dict(map(tuple, data))
    return params_cls(**params_dict)


def fixed_specs_as_array(specs: t.Dict[str, float]):
    """Convert several fixed parameter specs to a numpy array."""
    param_items = [tuple(param) for param in specs.items()]
    return np.array(param_items, dtype=_fixed_param_dtype)


def fixed_specs_from_array(data: np.ndarray):
    """Retrieve the fixed parameter specs from a numpy array."""
    return dict(tuple(item) for item in data)


def free_spec_as_array(spec: FreeParamSpec):
    """Convert a free parameter spec to a numpy array."""
    return np.array(astuple(spec), dtype=_free_param_dtype)


def free_spec_from_array(data: np.ndarray):
    """Retrieve a free parameter spec from a numpy array."""
    return FreeParamSpec(name=data["name"], lower_bound=data["lower_bound"],
                         upper_bound=data["upper_bound"])


def make_objective_func(chi_square_funcs: t.List[T_LikelihoodFunc],
                        params_cls: t.Type[Params],
                        fixed_specs: t.List[FixedParamSpec],
                        free_specs: t.List[FreeParamSpec]):
    """Create the objective function to minimize."""
    fixed_specs_dict = {spec.name: spec.value for spec in fixed_specs}
    var_names = [spec.name for spec in free_specs]

    def objective_func(x: t.Tuple[float, ...]):
        """Objective function.

        Evaluates the chi-square function.
        """
        params_dict = dict(zip(var_names, x))
        params_dict.update(fixed_specs_dict)
        params_obj = params_cls(**params_dict)
        return sum(func(params_obj) for func in chi_square_funcs)

    return objective_func


def has_best_fit(group: h5py.Group):
    """Check if a best fit result exists in an HDF5 group."""
    return BEST_FIT_GROUP_LABEL in group


T_BestFitFinderCallback = t.Callable[[Params, float], float]


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
class BestFitFinder(metaclass=ABCMeta):
    """Find the best chi-square fitting params of a model to certain data."""
    eos_model: Model
    datasets: DatasetJoin
    fixed_specs: T_FixedParamSpecs
    free_specs: t.List[FreeParamSpec]
    callback: t.Callable = T_BestFitFinderCallback

    # Private attributes.
    chi_square: T_LikelihoodFunc = field(default=None,
                                         init=False,
                                         repr=False)
    objective_func: T_LikelihoodFunc = field(default=None,
                                             init=False,
                                             repr=False)
    callback_func: T_BestFitFinderCallback = field(default=None,
                                                   init=False,
                                                   repr=False)

    def __post_init__(self):
        """"""
        self.chi_square = self._make_chi_square_func()
        self.objective_func = self._make_objective_func()
        self.callback_func = self._make_callback_func()

    @property
    def free_specs_bounds(self):
        return [_bounds(spec) for spec in self.free_specs]

    @property
    def chi_square_funcs(self):
        """Get the chi-square functions."""
        model = self.eos_model
        datasets = self.datasets
        likelihoods = [Likelihood(model, dataset) for dataset in datasets]
        return [lk.chi_square for lk in likelihoods]

    @abstractmethod
    def _make_objective_func(self) -> T_LikelihoodFunc:
        """"""
        pass

    @abstractmethod
    def _make_callback_func(self) -> t.Optional[T_LikelihoodFunc]:
        """"""
        pass

    @abstractmethod
    def _make_chi_square_func(self) -> T_LikelihoodFunc:
        """"""
        pass

    @abstractmethod
    def _exec(self) -> T_OptimizationInfo:
        """Optimization routine."""
        pass

    def exec(self):
        """Start the optimization procedure."""
        optimize_result = self._exec()
        return self._make_result(optimize_result)

    def _make_result(self, optimization_info: T_OptimizationInfo):
        """Create a proper BestFitResult instance from data."""
        free_specs = self.free_specs
        free_spec_names = [spec.name for spec in free_specs]
        # Extract the best-fit parameters from the result.
        fixed_specs_dict = {spec.name: spec.value for spec in self.fixed_specs}
        optimize_best_fit_params_dict = \
            dict(zip(free_spec_names, optimization_info["x"]))
        fixed_specs_dict.update(optimize_best_fit_params_dict)
        params = self.eos_model.params_cls(**fixed_specs_dict)

        # EOS today.
        eos_model = self.eos_model
        datasets = self.datasets
        free_specs = self.free_specs
        eos_today = eos_model.functions.wz(0, params)
        # Minimum of chi-square.
        chi_square_min = optimization_info["fun"]
        num_data = datasets.length
        # Reduced chi-square.
        num_free_params = len(free_specs)
        chi_square_reduced = chi_square_min / (num_data - num_free_params)
        h = params.h
        omegabh2 = params.omegabh2
        omegach2 = params.omegach2
        omega_m = (omegabh2 + omegach2) / h ** 2
        bic = chi_square_min + num_free_params * np.log(num_data)
        aic = chi_square_min + 2 * num_data
        result = BestFit(eos_model=eos_model,
                         datasets=datasets,
                         params=params,
                         free_params=free_specs,
                         eos_today=eos_today,
                         chi_square_min=chi_square_min,
                         chi_square_reduced=chi_square_reduced,
                         omega_m=omega_m,
                         aic=aic, bic=bic,
                         optimization_info=optimization_info)
        return result


@dataclass
class DEBestFitFinder(BestFitFinder):
    """A best-fit finder that uses SciPy differential evolution algorithm."""

    def _make_objective_func(self) -> T_LikelihoodFunc:
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

    def _make_callback_func(self) -> t.Optional[T_LikelihoodFunc]:
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

    def _make_chi_square_func(self) -> T_LikelihoodFunc:
        """Get the chi-square function."""
        return partial(chi_square_base,
                       chi_square_funcs=self.chi_square_funcs)

    def _exec(self) -> T_OptimizationInfo:
        """Optimization routine."""
        # Start the optimization procedure.
        return differential_evolution(self.objective_func,
                                      bounds=self.free_specs_bounds,
                                      callback=self.callback_func,
                                      polish=True)


def has_grid(group: h5py.Group):
    """Check if a grid result exists in an HDF5 group."""
    return GRID_GROUP_LABEL in group


T_GridFunc = t.Callable[[Params], float]
T_GridExecutorCallback = t.Callable[[Params], float]


@dataclass
class GridIterator(t.Iterable):
    """Iterates over a grid."""
    eos_model: Model
    datasets: DatasetJoin
    fixed_params: T_FixedParamSpecs
    param_partitions: T_ParamPartitionSpecs

    @property
    def shape(self):
        """The grid shape."""
        param_partitions = self.param_partitions
        return tuple(partition.data.size for partition in param_partitions)

    @property
    def size(self):
        """The grid total size."""
        return int(np.prod(self.shape))

    def __iter__(self):
        """"""
        eos_model = self.eos_model
        params_cls = eos_model.params_cls
        fixed_params = self.fixed_params
        free_names = [spec.name for spec in self.param_partitions]
        param_partitions = self.param_partitions
        fixed_params_dict = {spec.name: spec.value for spec in fixed_params}
        # Iterate over the grid using a multidimensional iterator. This
        # way we do not allocate memory for all the combinations of
        # parameter values that form the grid.
        for ndindex in np.ndindex(*self.shape):
            grid_values = [param_partitions[grid_idx].data[value_idx] for
                           grid_idx, value_idx in enumerate(ndindex)]
            params_dict = dict(zip(free_names, grid_values))
            params_dict.update(fixed_params_dict)
            params_obj = params_cls(**params_dict)
            yield params_obj


@dataclass
class GridExecutor(metaclass=ABCMeta):
    """Executes a grid."""
    callback: T_GridExecutorCallback = None

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

        def grid_func(params: Params):
            """Evaluate the chi-square function."""
            chi_square = sum(func(params) for func in chi_square_funcs)
            if callback_func is not None:
                callback_func(params)
            return chi_square

        return grid_func

    @abstractmethod
    def map(self, iterator: GridIterator):
        """Evaluates the chi-square on the grid."""
        pass


@dataclass
class DaskGridExecutor(GridExecutor):
    """Executes a grid using Dask."""

    def map(self, iterator: GridIterator):
        """Evaluates the chi-square on the grid."""
        grid_func = self._make_grid_func(iterator.eos_model,
                                         iterator.datasets)

        # Evaluate the grid using a multidimensional iterator. This
        # way we do not allocate memory for all the combinations of
        # parameter values that form the grid.
        params_bag = bag.from_sequence(iterator)
        chi_square_data = params_bag.map(grid_func).compute()
        return chi_square_data


@dataclass
class Grid:
    """Represent a chi-square evaluation over a parameter grid."""
    eos_model: Model
    datasets: DatasetJoin
    fixed_params: t.Dict[str, float]
    partition_arrays: t.Dict[str, np.ndarray]
    chi_square_data: np.ndarray

    @classmethod
    def from_data(cls, data: t.Iterable[float],
                  iterator: GridIterator):
        """Evaluates the chi-square on the grid."""
        grid_shape = iterator.shape
        chi_square_array: np.ndarray = np.asarray(data).reshape(grid_shape)
        fixed_params_dict = {spec.name: spec.value for spec in
                             iterator.fixed_params}
        # The parameter order used to evaluate the grid is defined by the
        # param_partition list order. Converting to a dictionary as follows
        # preserves this order since, in Python>=3.7, dictionaries keep their
        # items sorted according to their insertion order.
        partition_arrays = dict(map(astuple, iterator.param_partitions))
        return cls(eos_model=iterator.eos_model,
                   datasets=iterator.datasets,
                   fixed_params=fixed_params_dict,
                   partition_arrays=partition_arrays,
                   chi_square_data=chi_square_array)

    def save(self, file: h5py.File,
             group_name: str = None,
             force: bool = False):
        """Save a grid result to an HDF5 file."""
        if group_name is None or group_name == ROOT_GROUP_NAME:
            base_group = file[ROOT_GROUP_NAME]
        else:
            base_group = file.get(group_name, None)
            if base_group is None:
                base_group = file.create_group(group_name)
        if GRID_GROUP_LABEL in base_group and force:
            del base_group[GRID_GROUP_LABEL]
        group = base_group.create_group(GRID_GROUP_LABEL)

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
        group.create_dataset("chi_square", data=self.chi_square_data)

    @classmethod
    def load(cls, file: h5py.File,
             group_name: str = None):
        """Load a grid result from an HDF5 file."""
        base_group: h5py.Group = file["/"] if group_name is None else \
            file[group_name]
        group = base_group[GRID_GROUP_LABEL]

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
        chi_square = group["chi_square"][()]

        # Make a new instance.
        return cls(eos_model=eos_model,
                   datasets=datasets,
                   fixed_params=fixed_params,
                   partition_arrays=dict(partition_items),
                   chi_square_data=chi_square)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Represent a parameter confidence interval."""
    best_fit: float
    lower_bound: float
    upper_bound: float
    lower_error: float
    upper_error: float

    def __post_init__(self):
        """"""
        lower_error = abs(self.best_fit - self.lower_bound)
        upper_error = abs(self.upper_bound - self.best_fit)
        object.__setattr__(self, "lower_error", lower_error)
        object.__setattr__(self, "upper_error", upper_error)


@dataclass
class ParamGrid:
    """Represent a chi-square grid over a single parameter partition."""
    partition: np.ndarray
    chi_square: np.ndarray

    # Private attributes.
    chi_square_min: float = field(default=None, init=False, repr=False)
    spl_func: UnivariateSpline = \
        field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Post-initialization stage."""
        self.chi_square_min = self.chi_square.min()
        spl_func = UnivariateSpline(self.partition, self.chi_square, s=0)
        self.spl_func = spl_func

    def make_confidence_func(self, chi_square_delta: float,
                             chi_square_min: float):
        """Build a function whose zeros give confidence interval."""

        def confidence_func(x: t.Any):
            """The actual confidence function."""
            return self.spl_func(x) - (chi_square_min + chi_square_delta)

        return confidence_func

    def get_confidence_interval(self, chi_square_delta: float,
                                chi_square_min: float = None,
                                ini_guess: float = None,
                                delta_ini_guess: float = None):
        """Return the confidence interval for the given delta chi-square."""
        partition = self.partition
        chi_square = self.chi_square
        chi_square_min = chi_square_min or self.chi_square_min
        # Build the confidence function.
        confidence_func = self.make_confidence_func(chi_square_delta,
                                                    chi_square_min)
        lower_lim = partition.min()
        upper_lim = partition.max()
        if ini_guess is None:
            # Find the index of the ``x`` parameter that corresponds to the
            # minimum chi-square.
            min_index = np.nonzero(chi_square == self.chi_square_min)
            assert np.size(min_index) == 1
            ini_guess = partition[min_index][0]
        else:
            assert lower_lim <= ini_guess <= upper_lim
        # Try to find the lower bound. Use the secant method, where one of
        # the starting approximations is the best-fit value of the parameter.
        # The second approximation is a value slightly lower than the best-fit.
        left_range = np.linspace(lower_lim, ini_guess, num=256)
        chi_square_left = confidence_func(left_range)
        chi_square_greater = left_range[chi_square_left > 0]
        if not chi_square_greater.size:
            x_lower = lower_lim
        else:
            xl = chi_square_greater[-1]
            xu = left_range[chi_square_left < 0][0]
            try:
                x_lower, *info = newton(confidence_func, xl, x1=xu, disp=True,
                                        full_output=True)
            except RuntimeError:
                x_lower = lower_lim
        # Try to find the upper bound.
        right_range = np.linspace(ini_guess, upper_lim, num=256)
        chi_square_right = confidence_func(right_range)
        chi_square_greater = right_range[chi_square_right > 0]
        if not chi_square_greater.size:
            x_upper = upper_lim
        else:
            xl = right_range[chi_square_right < 0][0]
            xu = chi_square_greater[0]
            try:
                x_upper, *info = newton(confidence_func, xl, x1=xu, disp=True,
                                        full_output=True)
            except RuntimeError:
                x_upper = upper_lim
        # Return the confidence interval.
        opt_result = minimize_scalar(confidence_func, method="bounded",
                                     bounds=(x_lower, x_upper))
        x_bfv = opt_result["x"]
        return ConfidenceInterval(x_bfv, x_lower, x_upper)
