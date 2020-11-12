import typing as t
from collections import Iterable
from dataclasses import astuple, dataclass, field

import h5py
import numpy as np
from dask import bag
from scipy.interpolate import UnivariateSpline
from scipy.optimize import OptimizeResult, differential_evolution, newton

from .cosmology import (
    Dataset, DatasetJoin, Model, Params, get_model
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


@dataclass
class Minimizer:
    """"""
    eos_model: Model
    datasets: DatasetJoin
    fixed_specs: T_FixedParamSpecs
    free_specs: t.List[FreeParamSpec]

    def _make_callback(self):
        """"""


def find_best_fit(eos_model: Model,
                  datasets: DatasetJoin,
                  fixed_specs: t.List[FixedParamSpec],
                  free_specs: t.List[FreeParamSpec],
                  callback: t.Callable = None):
    """Execute the optimization procedure and return the best-fit result."""

    def _chi_square_func(_dataset: Dataset):
        """Return the chi-square function for a given dataset."""
        return Likelihood(eos_model, _dataset).chi_square

    params_cls = eos_model.params_cls
    chi_square_funcs = [_chi_square_func(dataset) for dataset in datasets]
    free_spec_names = [spec.name for spec in free_specs]
    free_spec_bounds = [_bounds(spec) for spec in free_specs]
    fixed_specs_dict = {spec.name: spec.value for spec in fixed_specs}
    obj_func = make_objective_func(chi_square_funcs, params_cls, fixed_specs,
                                   free_specs)
    # Start the optimization procedure.
    optimize_result = differential_evolution(obj_func,
                                             bounds=free_spec_bounds,
                                             callback=callback,
                                             polish=True)
    # Extract the best-fit parameters from the result.
    optimize_best_fit_params_dict = \
        dict(zip(free_spec_names, optimize_result["x"]))
    fixed_specs_dict.update(optimize_best_fit_params_dict)
    best_fit_params = params_cls(**fixed_specs_dict)
    return make_best_fit_result(eos_model,
                                datasets,
                                best_fit_params,
                                free_specs,
                                optimize_result)


def make_best_fit_result(eos_model: Model,
                         datasets: DatasetJoin,
                         best_fit_params: Params,
                         free_specs: t.List[FreeParamSpec],
                         optimization_info: T_OptimizationInfo):
    """Create a proper BestFitResult instance from data."""
    # EOS today.
    eos_today = eos_model.functions.wz(0, best_fit_params)
    # Minimum of chi-square.
    chi_square_min = optimization_info["fun"]
    num_data = datasets.length
    # Reduced chi-square.
    num_free_params = len(free_specs)
    chi_square_reduced = chi_square_min / (num_data - num_free_params)
    h = best_fit_params.h
    omegabh2 = best_fit_params.omegabh2
    omegach2 = best_fit_params.omegach2
    omega_m = (omegabh2 + omegach2) / h ** 2
    bic = chi_square_min + num_free_params * np.log(num_data)
    aic = chi_square_min + 2 * num_data
    best_fit_result = BestFit(eos_model=eos_model,
                              datasets=datasets,
                              params=best_fit_params,
                              free_params=free_specs,
                              eos_today=eos_today,
                              chi_square_min=chi_square_min,
                              chi_square_reduced=chi_square_reduced,
                              omega_m=omega_m,
                              aic=aic, bic=bic,
                              optimization_info=optimization_info)
    return best_fit_result


@dataclass
class Grid:
    """Represent a chi-square evaluation over a parameter grid."""
    eos_model: Model
    datasets: DatasetJoin
    fixed_params: t.Dict[str, float]
    partition_arrays: t.Dict[str, np.ndarray]
    chi_square_data: np.ndarray

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


def has_grid(group: h5py.Group):
    """Check if a grid result exists in an HDF5 group."""
    return GRID_GROUP_LABEL in group


T_GridFunc = t.Callable[[t.Tuple[int, ...]], float]


@dataclass
class GridExecutor(Iterable):
    """Executes a grid."""

    eos_model: Model

    datasets: DatasetJoin
    fixed_params: T_FixedParamSpecs
    param_partitions: T_ParamPartitionSpecs
    param_partition_names: t.List[str] = field(init=False,
                                               default=None,
                                               repr=False)

    # Private attributes.
    likelihoods: t.List[Likelihood] = field(init=False,
                                            default=None,
                                            repr=False)

    grid_func: T_GridFunc = field(init=False,
                                  default=None,
                                  repr=False)

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
        self.grid_func = self._make_grid_func()

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

        def grid_func(value_indexes: t.Tuple[int, ...]):
            """Grid function.

            Evaluates the chi-square function.
            """
            grid_values = [param_partition[grid_idx].data[value_idx] for
                           grid_idx, value_idx in enumerate(value_indexes)]
            params_dict = dict(zip(free_names, grid_values))
            params_dict.update(fixed_params_dict)
            params_obj = params_cls(**params_dict)
            return sum(func(params_obj) for func in chi_square_funcs)

        return grid_func

    def eval(self):
        """Evaluates the chi-square on the grid."""
        grid_func = self._make_grid_func()
        param_partitions = self.param_partitions
        grid_shape = tuple(
            partition.data.size for partition in param_partitions)

        # Evaluate the grid using a multidimensional iterator. This
        # way we do not allocate memory for all the combinations of
        # parameter values that form the grid.
        grid_indexes = np.ndindex(*grid_shape)
        indexes_bag = bag.from_sequence(grid_indexes)
        chi_square_data = indexes_bag.map(grid_func).compute()
        chi_square_array: np.ndarray = np.asarray(chi_square_data).reshape(
            grid_shape)
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
class ConfidenceInterval:
    """Represent a parameter confidence interval."""
    best_fit: float
    lower_bound: float
    upper_bound: float

    @property
    def lower_error(self):
        return abs(self.best_fit - self.lower_bound)

    @property
    def upper_error(self):
        return abs(self.upper_bound - self.best_fit)


@dataclass
class ParamGrid:
    """Represent a chi-square grid over a single parameter partition."""
    partition: np.ndarray
    chi_square: np.ndarray

    # Private attributes.
    chi_square_min: float = field(default=None, init=False, repr=False)
    spl_func: t.Callable[[np.ndarray], np.ndarray] = \
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
        if ini_guess is None:
            # Find the index of the ``x`` parameter that corresponds to the
            # minimum chi-square.
            min_index = np.nonzero(chi_square == self.chi_square_min)
            assert np.size(min_index) == 1
            ini_guess = partition[min_index][0]
        min_diff = min(partition.max() - ini_guess,
                       ini_guess - partition.min())
        # Try to find the lower bound. Use the secant method, where one of
        # the starting approximations is the best-fit value of the parameter.
        # The second approximation is a value slightly lower than the best-fit.
        delta_ini_guess = delta_ini_guess or min_diff / 4
        xl, xu = ini_guess - delta_ini_guess, ini_guess
        x_lower, *info = newton(confidence_func, xl, x1=xu, disp=True,
                                full_output=True)
        # Try to find the upper bound.
        xl, xu = ini_guess, ini_guess + delta_ini_guess
        x_upper, *info = newton(confidence_func, xl, x1=xu, full_output=True)
        # Return the confidence interval.
        return ConfidenceInterval(ini_guess, x_lower, x_upper)
