import os
import typing as t
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp

from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable, integrate

from cosmostat.constants_units import OMEGABH2, OMEGACH2, REDUCED_H0
from cosmostat.cosmology import Functions as BaseFunctions
from cosmostat.cosmology import Model
from cosmostat.cosmology import Params as ParamsBase
from cosmostat.cosmology import T_CosmologyFunc

cw_directory = os.path.dirname(os.path.abspath(__file__))

# ==========   Numerical integration quantities for the calculation ===========
QUAD_EPS_ABS = 1.49e-8


# ---------------------------------------------------------------------#
# This part of the script contains the definition of a new equation
# of state for DE component and its integral to enter Hubble function:
#
# Parametrization capable of model quintessence-like theories
# or Modified gravity models (crossing the phantom divide line or not)
#
# ****** Reminder: A = w0, n = w1, C = w2*****
#
# w(z) = -1 - w0*exp(-z)*(z**w1 - z - w2)
#
# which recovers w = -1 for z>>0
# And has w(z=0) =
#               if w1 == 1
#               -1 - w0*(1-w2) (if w1 = 1)
#               else
#               -1 + w0*w2
#
# ---------------------------------------------------------------------#
class Params(ParamsBase, t.NamedTuple):
    """"""

    w0: float
    w1: float
    w2: float
    h: float = REDUCED_H0
    omegabh2: float = OMEGABH2
    omegach2: float = OMEGACH2


@jit(nopython=True, cache=True)
def wz(z: float, params: Params):
    """
    w(z) = -1 - w0*exp(-z)*(z**w1 - z - w2)

    One parametrization to fit them all!

    Case 0. If w0 = 0 we recover LCDM with w(z) = -1 -- cosmological const.
    Case 1. If w1 = w2 = 1 we have w(z) = -1 + w0*exp(-n)*z -- exponential-like
    Case 2. If w2 = 0, we have w(z) = -1 -w0*exp(-z)(z**w1 -z) -- f(R) like
    Case 3. If w1 = 0 we get, by direct substitution w(z) = -1 -w0*Exp(-z)(1-z-w2)
                      which mimics a quintessence-like evolution
            In this case we redefined the parameters to avoid degeneracy among
            w0 & w2:
            w(z|w1 = 0) = -1 + exp(-z) (w0*z + w2') with w2' = w0(w2-1)
    Case 4. Fixing w1 to be in [0, 1], and w0, w2 Real numbers, we have the most
            general evolution with
            w(z) = -1 - A * Exp(-z) [z^n -z -C]

    The only fixed behaviour is the limit w(z>>0) --> -1

    :param w0: A: fixes the amplitude
    :param w1: n: power of the polynomial, fixes the number of roots and crosses on the x-axis
    :param w2: C: additive constant, scrolls the roots left/right along the x-axis
    :returns: w(z) for given value of z
    """
    w0 = params.w0
    w1 = params.w1
    w2 = params.w2
    if w0 == 0:
        return -1
    if w1 == 0:
        # return -1 - w0 * np.exp(-z) * (1 - z - w2) # direct substitution
        return -1 + exp(-z) * (
            w0 * z + w2
        )  # re-definition of eos as explained above
    if w2 == 0:
        return -1 - w0 * exp(-z) * (z ** w1 - z)
    return -1 - w0 * exp(-z) * (z ** w1 - z - w2)


@jit(nopython=True, cache=True)
def f_dez_integrand(zp: float, params: Params):
    return (1 + wz(zp, params)) / (1 + zp)


# Signature of the numba C function.
f_dez_integrand_cfunc_sig = types.double(
    types.int32, types.CPointer(types.double)
)


@cfunc(f_dez_integrand_cfunc_sig, cache=True)
def _f_dez_integrand_cfunc(num_params: int, params_pointer: int):
    """Numba C implementation of the equation of state integrand.

    :param num_params: Integer, it refers to the number of arguments of the
              integrand. Automatically guessed by numba.
    :param params_pointer: A C pointer to the data passed to the integrand,
                       including the values of the variable integrated.
    :return:
    """
    # Converts the data C pointer into a numpy array view.
    params_array = carray(params_pointer, num_params)
    # Extract data. First element is always the variable to be integrated.
    zp = params_array[0]  # red shift.
    w0 = params_array[1]
    w1 = params_array[2]
    w2 = params_array[3]
    h = params_array[4]
    omegabh2 = params_array[5]
    omegach2 = params_array[6]

    # w3 = params_array[4] #<--- now we have only 3 parameters
    params = Params(w0, w1, w2, h, omegabh2, omegach2)  # , w3
    return (1 + wz(zp, params)) / (1 + zp)


# We have to wrap the ``ctypes`` implementation of the C-function
# with a LowLevelCallable.
f_dez_integrand_cfunc = LowLevelCallable(_f_dez_integrand_cfunc.ctypes)


@lru_cache(maxsize=1024)
def f_dez_fast(z: float, params: Params) -> float:
    """Integral of DE eos for Hubble function using a
    LowLevelCallable.
    :param z:
    :param params:
    :return:
    """
    int_de: float  # Type annotation.
    # noinspection PyTupleAssignmentBalance
    int_de, error = integrate.quad(
        f_dez_integrand_cfunc,
        0,
        z,
        epsabs=QUAD_EPS_ABS,
        # Note that args is equal to params.
        #  This is necessary when the integrand
        #  is a numba function.
        args=params,
    )
    return exp(3 * int_de)


def f_dez(z: float, params: Params):
    """Integral of DE eos for Hubble function"""
    # noinspection PyTupleAssignmentBalance
    int_de, error = integrate.quad(
        f_dez_integrand,
        0,
        z,
        epsabs=QUAD_EPS_ABS,
        # Note that args is equal to (params,).
        #  This is the correct procedure when the
        #  integrand is not a numba function.
        args=(params,),
    )
    return exp(3 * int_de)


@dataclass
class Functions(BaseFunctions):
    """Gus model."""

    def _make_wz_func(self) -> T_CosmologyFunc:
        """"""
        return wz

    def _make_f_dez_func(self) -> T_CosmologyFunc:
        """"""
        return f_dez_fast


# Singleton with the model functions.
functions = Functions()

# Dictionary of the functions.
functions_dict = asdict(functions)

# Singleton with the model definition.
model = Model(name="ONE", params_cls=Params, functions=functions)
