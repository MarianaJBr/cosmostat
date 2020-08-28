import typing as t
from dataclasses import astuple, dataclass, field

import h5py
import numpy as np
from dask import bag
from scipy.optimize import OptimizeResult, differential_evolution

from .cosmology import (
    Dataset, DatasetUnion, Likelihood, Model, Params, T_LikelihoodFunc,
    get_model
)

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
class BestFitResult:
    """Represent the result of a SciPy optimization routine."""
    eos_model: Model
    datasets: DatasetUnion
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
        datasets = DatasetUnion.from_name(group_attrs["datasets"])
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


def fixed_specs_as_array(specs: T_FixedParamSpecs):
    """Convert several fixed parameter specs to a numpy array."""
    param_items = [astuple(param) for param in specs]
    return np.array(param_items, dtype=_fixed_param_dtype)


def fixed_specs_from_array(data: np.ndarray):
    """Retrieve the fixed parameter specs from a numpy array."""
    return [FixedParamSpec(*tuple(item)) for item in data]


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
    datasets: DatasetUnion
    fixed_specs: T_FixedParamSpecs
    free_specs: t.List[FreeParamSpec]

    def _make_callback(self):
        """"""


def find_best_fit(eos_model: Model,
                  datasets: DatasetUnion,
                  fixed_specs: t.List[FixedParamSpec],
                  free_specs: t.List[FreeParamSpec],
                  callback: t.Callable):
    """Execute the optimization procedure and return the best-fit result."""

    def _chi_square_func(_dataset: Dataset):
        """Return the chi-square function for a given dataset."""
        return eos_model.make_likelihood(_dataset).chi_square

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
                         datasets: DatasetUnion,
                         best_fit_params: Params,
                         free_specs: t.List[FreeParamSpec],
                         optimization_info: T_OptimizationInfo):
    """Create a proper BestFitResult instance from data."""
    # EOS today.
    eos_today = eos_model.wz(0, best_fit_params)
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
    best_fit_result = BestFitResult(eos_model=eos_model,
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
class GridResult:
    """"""
    eos_model: Model
    datasets: DatasetUnion
    fixed_params: T_FixedParamSpecs
    param_partitions: T_ParamPartitionSpecs
    chi_square_data: np.ndarray

    @property
    def partition_arrays(self):
        """A dictionary with the partition arrays."""
        return dict(map(astuple, self.param_partitions))

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
        for partition in self.param_partitions:
            arrays_group.create_dataset(partition.name, data=partition.data)

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
        datasets = DatasetUnion.from_name(group_attrs["datasets"])
        fixed_params = fixed_specs_from_array(group_attrs["fixed_params"])

        # Load partition arrays.
        param_partitions = []
        arrays_group: h5py.Group = group[PARAM_PARTITIONS_GROUP_LABEL]
        arrays_group_keys = list(arrays_group.keys())
        for key in arrays_group_keys:
            item = arrays_group[key]
            # Load numpy arrays from data sets.
            if isinstance(item, h5py.Dataset):
                name = key
                data = item[()]
                param_partitions.append(ParamPartitionSpec(name, data))

        # Load chi-square data.
        chi_square = group["chi_square"][()]

        # Make a new instance.
        return cls(eos_model=eos_model,
                   datasets=datasets,
                   fixed_params=fixed_params,
                   param_partitions=param_partitions,
                   chi_square_data=chi_square)


def has_grid(group: h5py.Group):
    """Check if a grid result exists in an HDF5 group."""
    return GRID_GROUP_LABEL in group


@dataclass
class Grid:
    """Represent a grid """
    eos_model: Model
    datasets: DatasetUnion
    fixed_params: T_FixedParamSpecs
    param_partitions: T_ParamPartitionSpecs

    # Private attributes.
    param_partition_names: t.List[str] = field(init=False,
                                               default=None,
                                               repr=False)
    likelihoods: t.List[Likelihood] = field(init=False,
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
        likelihoods = [model.make_likelihood(dataset) for dataset in datasets]
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
        return GridResult(eos_model=self.eos_model,
                          datasets=self.datasets,
                          fixed_params=self.fixed_params,
                          param_partitions=param_partitions,
                          chi_square_data=chi_square_array)
