"""
© Mariana Jaber, 2017(2018)(2019)

This part of the program contains the basic cosmological parameters
such as energy and physical densities for a Flat LCDM
cosmology.

It is based on Planck 2015 Cosmological Parameters report
(arXiv:1502.01589) table 3 column 1:
"Planck + TT + lowP".

-----------------------------------------
1502.01589 (Cosmological parameters)
1502.01590 (DE & MG)
-----------------------------------------
"""
import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, field
from math import sqrt

import numpy as np
from scipy import integrate

from .constants_units import (
    COVMATCMB_3, H0P, OMEGAG0, OMEGAR0, RHOCR0, ZDEC,
    ZDRAG
)

# String used to join/separate the labels of several Datasets grouped
# in a DatasetUnion object.
DATASET_UNION_LABEL_SEP = " + "

QUAD_EPS_ABS = 1.49e-8


# NOTE: Here, we use named-tuples to group the function parameters.
#  The main reason for making this choice is because numba-compiled
#  functions can deal with named-tuples, but not with dataclasses (in
#  a simple way). Of course, we could use dataclasses, since it is a better
#  approach to structure the function parameters, but at some point, we would
#  have to convert the dataclass to a named-tuple before passing the
#  parameters to a numba-compiled function.
class Params(t.NamedTuple):
    """"""
    h: float
    omegabh2: float
    omegach2: float


def b_to_g(z: float, params: Params):
    """Baryon-to-photon ratio as function of z"""
    # return 3 * OMEGAB0 / (4 * OMEGAG0 * (1 + z))
    h = params.h
    omegabh2 = params.omegabh2
    return 3 * (omegabh2 / h ** 2) / (4 * OMEGAG0 * (1 + z))


@dataclass
class Dataset:
    """Represent a database information."""
    name: str
    label: str
    data: np.ndarray = field(repr=False)
    # The name of the cosmology function suitable to work with this data.
    cosmology_func: str = field(repr=False)

    @property
    def length(self):
        """Number of data items."""
        return len(self.data)


# Type hints for cosmological functions.
T_CosmologyFunc = t.Callable[[float, Params], float]
T_CMBCosmologyFunc = t.Callable[[Params], float]

# Type hints for likelihood functions.
T_LikelihoodTheoryFunc = T_CosmologyFunc
T_LikelihoodResidualsFunc = t.Callable[[Params], np.ndarray]
T_LikelihoodFunc = t.Callable[[Params], float]


@dataclass
class Likelihood:
    """Groups likelihood functions for a specific model and dataset."""
    dataset: Dataset
    theory_func: T_LikelihoodTheoryFunc = field(default=None)
    residuals_func: T_LikelihoodResidualsFunc = field(init=False, default=None)
    chi_square: T_LikelihoodFunc = field(init=False, default=None)

    def __post_init__(self):
        """"""
        self.residuals_func = self._make_residuals_func()
        self.chi_square = self._make_chi_square_func()

    def _make_residuals_func(self) -> t.Optional[T_LikelihoodResidualsFunc]:
        """Build the residuals function."""
        data = self.dataset.data
        theory_func = self.theory_func

        def residuals_func(params: Params):
            """Returns a vector/array with the residuals."""
            zi_data, obs_data, err_data = data.T
            th_func = np.array([theory_func(zi, params) for zi in zi_data])
            return (obs_data - th_func) / err_data

        return residuals_func

    def _make_chi_square_func(self):
        """Build the chi-square function."""
        residuals_func = self.residuals_func

        def chi_square(params: Params):
            """Standard chi-square function."""
            return float(np.sum(residuals_func(params) ** 2))

        return chi_square


# Type hint for the  CMB chi2 function.
T_CMBLikelihoodTheoryFunc = t.Callable[[Params], np.ndarray]


@dataclass
class CMBLikelihood:
    """Groups likelihood functions for a specific model and dataset."""
    dataset: Dataset
    theory_func: T_CMBLikelihoodTheoryFunc = field(default=None)
    chi_square: T_LikelihoodFunc = field(init=False, default=None)

    def __post_init__(self):
        """"""
        self.chi_square = self._make_chi_square_func()

    def _make_chi_square_func(self):
        """Build the chi-square function."""
        y_cmb_func = self.theory_func

        def cmb_chi_square(params: Params):
            """Standard chi-square function."""
            y_cmb = y_cmb_func(params)
            covmat = COVMATCMB_3  # 4x4 matrix, including ns
            covmat_inv = np.linalg.inv(covmat)
            # base_TTlowP.covmat from Planck DE&MG 2015 Table 4
            return float(np.dot(y_cmb, np.dot(covmat_inv, y_cmb)))

        return cmb_chi_square


@dataclass
class Model(metaclass=ABCMeta):
    """Represents a cosmological model."""
    name: str
    params_cls: t.Type[Params]

    # Cosmological functions
    wz: T_CosmologyFunc = field(init=False, default=None, repr=False)
    f_dez: T_CosmologyFunc = field(init=False, default=None, repr=False)
    hubble_flat: T_CosmologyFunc = field(init=False, default=None, repr=False)
    ez_flat: T_CosmologyFunc = field(init=False, default=None, repr=False)
    rho_de: T_CosmologyFunc = field(init=False, default=None, repr=False)
    b_to_g: T_CosmologyFunc = field(init=False, default=None, repr=False)
    r_s: T_CosmologyFunc = field(init=False, default=None, repr=False)
    d_vz: T_CosmologyFunc = field(init=False, default=None, repr=False)
    r_bao: T_CosmologyFunc = field(init=False, default=None, repr=False)
    mu_sne: T_CosmologyFunc = field(init=False, default=None, repr=False)
    hz_hubble_flat: T_CosmologyFunc = \
        field(init=False, default=None, repr=False)
    d_ang: T_CosmologyFunc = field(init=False, default=None, repr=False)
    r_cmb: T_CMBCosmologyFunc = field(init=False, default=None, repr=False)
    theta_star: T_CMBCosmologyFunc = field(init=False, default=None, repr=False)
    l_a: T_CMBCosmologyFunc = field(init=False, default=None, repr=False)
    y_vec_cmb: T_CMBLikelihoodTheoryFunc = \
        field(init=False, default=None, repr=False)

    def __post_init__(self):
        """"""
        # Build functions.
        self.wz = self._make_wz_func()
        self.f_dez = self._make_f_dez_func()
        self.hubble_flat = self._make_hubble_flat_func()
        self.ez_flat = self._make_ez_flat_func()
        self.rho_de = self._make_rho_de_func()
        self.b_to_g = b_to_g
        self.r_s = self._make_r_s_func()
        self.d_vz = self._make_d_vz_func()
        self.r_bao = self._make_r_bao_func()
        self.mu_sne = self._make_mu_sne_func()
        self.hz_hubble_flat = self._make_hz_hubble_flat_func()
        self.d_ang = self._make_d_ang_func()
        self.r_cmb = self._make_r_cmb()
        self.theta_star = self._make_theta_star()
        self.l_a = self._make_l_a_func()
        self.y_vec_cmb = self._make_y_vec_cmb_func()

    def make_likelihood(self, dataset: Dataset):
        """Return a likelihood object using the given dataset."""
        theory_func = asdict(self)[dataset.cosmology_func]
        if dataset.name == "CMB":
            return CMBLikelihood(dataset, theory_func)
        return Likelihood(dataset, theory_func)

    @abstractmethod
    def _make_wz_func(self) -> T_CosmologyFunc:
        """"""
        pass

    @abstractmethod
    def _make_f_dez_func(self) -> T_CosmologyFunc:
        """"""
        pass

    def _make_hubble_flat_func(self):
        """"""
        f_dez = self.f_dez

        def hubble_flat(z: float, params: Params):
            """Hubble function in terms of OmegaMatter flat universe."""
            h = params.h
            omegabh2 = params.omegabh2
            omegach2 = params.omegach2
            omega_m = (omegabh2 + omegach2) / h ** 2
            hubble_func = H0P * h * sqrt(
                OMEGAR0 * (1 + z) ** 4 + omega_m * (1 + z) ** 3 +
                (1 - OMEGAR0 - omega_m) * f_dez(z, params))
            return hubble_func

        return hubble_flat

    def _make_ez_flat_func(self):
        """"""
        hubble_flat = self.hubble_flat

        def ez_flat(z: float, params: Params):
            """Normalized Hubble function E(z) = H(z)/H0 for the flat
            Universe.
            """
            h = params.h
            hz = hubble_flat(z, params)
            h0 = H0P * h
            return hz / h0

        return ez_flat

    def _make_rho_de_func(self):
        """"""
        f_dez = self.f_dez

        def rho_de(z: float, params: Params):
            """Volumetric energy density of the dark energy fluid component."""
            h = params.h
            omegabh2 = params.omegabh2
            omegach2 = params.omegach2
            omegam = (omegabh2 + omegach2) / h ** 2
            rho_de0 = (1 - omegam - OMEGAR0) * RHOCR0
            return rho_de0 * f_dez(z, params)

        return rho_de

    def _make_r_s_integrand(self):
        """"""
        hubble_flat = self.hubble_flat
        _btog = self.b_to_g

        def r_s_integrand(z: float, params: Params):
            """ flat universe """
            cs = 1 / sqrt(3 * (1 + _btog(z, params)))
            return cs / hubble_flat(z, params)

        return r_s_integrand

    def _make_r_s_func(self):
        """"""
        r_s_integrand = self._make_r_s_integrand()

        def r_s(zeval: float, params: Params):
            """Sound horizon at zeval either zdrag or zdec"""
            # noinspection PyTupleAssignmentBalance
            r_sound, error = integrate.quad(r_s_integrand, zeval, np.inf,
                                            epsabs=QUAD_EPS_ABS,
                                            args=(params,))
            return r_sound

        return r_s

    def _make_d_vz_integrand_func(self):
        """"""
        hubble_flat = self.hubble_flat

        def d_vz_integrand(zp: float, params: Params):
            return 1 / hubble_flat(zp, params)

        return d_vz_integrand

    def _make_d_vz_func(self):
        """"""
        hubble_flat = self.hubble_flat
        d_vz_integrand = self._make_d_vz_integrand_func()

        def d_vz(z: float, params: Params):
            """Dilation scale for rbao size """
            # noinspection PyTupleAssignmentBalance
            int2, error = integrate.quad(d_vz_integrand, 0, z,
                                         epsabs=QUAD_EPS_ABS,
                                         args=(params,))
            int3 = (z / hubble_flat(z, params)) ** (1. / 3)
            return int3 * int2 ** (2 / 3)

        return d_vz

    def _make_r_bao_func(self):
        """"""
        r_s = self.r_s
        d_vz = self.d_vz

        def r_bao(z: float, params: Params):
            """BAO scale at redshift z."""
            return r_s(ZDRAG, params) / d_vz(z, params)

        return r_bao

    def _make_d_ang_integrand_func(self):
        """"""
        hubble_flat = self.hubble_flat

        def d_ang_integrand(zp: float, params: Params):
            """Integrand for the angular diameter distance: dz/H(z)

            hubble_func = H0p * h * sqrt(
            OMEGAR0 * (1 + z) ** 4 + (omegabh2+omegach2)/h**2 * (1 + z) ** 3 +
            (1 - OMEGAR0 - ((omegabh2+omegach2)/h**2)) * f_DEz(z, w_params))
            """
            return 1 / hubble_flat(zp, params)

        return d_ang_integrand

    def _make_d_ang_func(self):
        """"""
        d_ang_integrand = self._make_d_ang_integrand_func()

        def d_ang(z: float, params: Params):
            """Angular Diameter distance:
            D_a(z) = c/(1+z)Int_0^z(dz'/H(z'))
            """
            # noinspection PyTupleAssignmentBalance
            int_val, error = integrate.quad(d_ang_integrand, 0, z,
                                            epsabs=QUAD_EPS_ABS,
                                            args=(params,))
            return 1 / (1 + z) * int_val

        return d_ang

    def _make_distance_sne_integrand_func(self):
        """SNeIa distances integrand."""
        hubble_flat = self.hubble_flat

        def distance_sne_integrand(z: float, params: Params):
            """"""
            return 1 / hubble_flat(z, params)

        return distance_sne_integrand

    def _make_distance_sne_func(self):
        """SNeIa distances definition."""
        distance_sne_integrand = self._make_distance_sne_integrand_func()

        def distance_sne(z: float, params: Params):
            """Luminosity distance for SNe data.

            (1+z) * Int_0^z dz E^-1(z; args) where E(z) = H(z)/H0
            """
            # noinspection PyTupleAssignmentBalance
            int_val, err = integrate.quad(distance_sne_integrand, 0, z,
                                          epsabs=QUAD_EPS_ABS,
                                          args=(params,))
            return (1 + z) * int_val

        return distance_sne

    def _make_mu_sne_func(self):
        """"""
        distance_sne = self._make_distance_sne_func()

        def mu_sne(z: float, params: Params):
            """Modulus distance for SNe data: 5*log10 of dist_lum(z)

            5*Log10(distance_SNe in Mpc) + 25
            """
            mu0 = 25
            d = distance_sne(z, params)
            return 5 * np.log10(d) + mu0

        return mu_sne

    def _make_hz_hubble_flat_func(self):
        """"""
        hubble_flat = self.hubble_flat

        def hz_hubble_flat(z: float, params: Params):
            """Hubble function in terms of OmegaMatter flat universe."""
            return 100 * hubble_flat(z, params) / H0P

        return hz_hubble_flat

    def _make_r_cmb(self):
        """"""
        d_ang = self.d_ang

        def r_cmb(params: Params):
            """R(z*) = np.sqrt(Omega_M*H0*H0) D_ang(z*)/c"""
            h = params.h
            omegabh2 = params.omegabh2
            omegach2 = params.omegach2
            omega_m = (omegabh2 + omegach2) / h ** 2
            # Important to keep H0p factor for units
            factor1 = np.sqrt(omega_m) * h * H0P
            d_ang_z_star = d_ang(ZDEC, params) * (1 + ZDEC)
            return factor1 * d_ang_z_star

        return r_cmb

    def _make_theta_star(self):
        """"""
        r_s = self.r_s
        d_vz_integrand = self._make_d_vz_integrand_func()

        def theta_star(params: Params):
            """Angular sound horizon at decoupling.

            Used to calculate l_A in Wang & Mukherjee (2007) 's matrix
            for the CMB compressed likelihood.
            :return: Theta(z_dec) = r_s(z_dec) / D_V(z_dec)
            """
            # noinspection PyTupleAssignmentBalance
            int_val, error = integrate.quad(d_vz_integrand, 0, ZDEC,
                                            epsabs=QUAD_EPS_ABS,
                                            args=(params,))
            theta_dec = r_s(ZDEC, params) / int_val
            return theta_dec

        return theta_star

    def _make_l_a_func(self):
        """"""
        theta_star = self.theta_star

        def l_a(params: Params):
            """Angular scale of the sound horizon at last scattering.
            l_A:= pi/Theta_*
            Used in Wang & Mukherjee (2007) 's matrix for the CMB compressed
            likelihood. See section 5.1.6 of Planck 2015 DE & MG paper
            :return: l_A = pi/Theta(z*)
            """
            t_s = theta_star(params)
            return np.pi / t_s

        return l_a

    def _make_y_vec_cmb_func(self):
        """"""
        r_cmb = self.r_cmb
        l_a = self.l_a

        def y_vec_cmb(params: Params):
            """Standard chi-square function."""
            omegabh2 = params.omegabh2
            y_cmb = np.array([
                1.7382 - r_cmb(params),
                301.63 - l_a(params),
                0.02262 - omegabh2,
                # 0.9741 - 0.97415
            ])
            return y_cmb

        return y_vec_cmb


@dataclass
class DatasetUnion(t.Iterable):
    ### T0DO: refactor this class to avoid phrasing 'Union'
    ### Idea: DatasetJoin ?
    """Represent the union of several datasets."""
    datasets: t.List[Dataset]

    def __post_init__(self):
        """Post initialization."""

        # Sort datasets by name.
        def _name(dataset: Dataset):
            return dataset.name

        self.datasets = sorted(self.datasets, key=_name)

    @classmethod
    def from_name(cls, name: str):
        """"""
        names = name.split(DATASET_UNION_LABEL_SEP)
        datasets = [get_dataset(label) for label in names]
        return cls(datasets)

    @property
    def name(self):
        return "+".join([dataset.name for dataset in self.datasets])

    @property
    def label(self):
        return DATASET_UNION_LABEL_SEP.join([
            dataset.label for dataset in self.datasets])

    @property
    def length(self):
        """Number of data items."""
        return sum([dataset.length for dataset in self.datasets])

    def __iter__(self):
        return iter(self.datasets)


# Mapping for the different EoS / models. Keep private/protected.
_EOS_MODELS: t.Dict[str, Model] = {}

# Mapping for the different datasets. Keep private/protected.
_DATASETS: t.Dict[str, Dataset] = {}

# Mapping for the registered dataset combinations. Keep private/protected.
_DATASET_UNIONS: t.Dict[str, DatasetUnion] = {}


def register_model(model: Model):
    """Register a new model."""
    if model.name in _EOS_MODELS:
        raise KeyError(f"model '{model.name}' has been already registered")
    _EOS_MODELS[model.name] = model


def get_model(name: str):
    """Return an existing model."""
    return _EOS_MODELS[name]


def registered_models():
    """Return a list with the registered models."""
    return list(_EOS_MODELS)


def register_dataset(dataset: Dataset):
    """Register a new dataset."""
    if dataset.name in _DATASETS:
        raise KeyError(f"dataset '{dataset.name}' has been already registered")
    _DATASETS[dataset.name] = dataset


def get_dataset(name: str):
    """Return an existing dataset."""
    return _DATASETS[name]


def registered_datasets():
    """Return a list with the registered datasets."""
    return list(_DATASETS)


def register_dataset_union(dataset_union: DatasetUnion):
    """Register a new dataset union."""
    if dataset_union.name in _DATASET_UNIONS:
        raise KeyError(f"dataset union '{dataset_union.name}' has been "
                       f"already registered")
    _DATASET_UNIONS[dataset_union.name] = dataset_union


def get_dataset_union(name: str):
    """Return an existing dataset union."""
    return _DATASET_UNIONS[name]


def registered_dataset_unions():
    """Return a list with the registered dataset unions."""
    return list(_DATASET_UNIONS)
