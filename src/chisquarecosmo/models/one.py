import os
import typing as t
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp

import numpy as np
from chisquarecosmo.constants_units import OMEGABH2, OMEGACH2, REDUCED_H0
from chisquarecosmo.cosmology import (
    Functions as BaseFunctions, Model, Params as ParamsBase,
    T_CosmologyFunc
)
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable, integrate

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

    Case 1. If w1 = 0 we get a quintessence-like evolution
            In this case we
    """
    # w0: fija la amplitud
    # w1: potencia del polinomio (numero de raices, cortes)
    # w2: recorre el polinomio derecha/izquierda
    w0 = params.w0
    w1 = params.w1
    w2 = params.w2
    if w0 == 0:
        return -1
    if w1 == 0:
        # return -1 - w0 * np.exp(-z) * (1 - z - w2)
        return -1 + np.exp(-z) * (w0 * z + w2)
    if w2 == 0:
        return -1 - w0 * np.exp(-z) * (z ** w1 - z)
    return -1 - w0 * np.exp(-z) * (z ** w1 - z - w2)


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
    int_de, error = integrate.quad(f_dez_integrand_cfunc, 0, z,
                                   epsabs=QUAD_EPS_ABS,
                                   # Note that args is equal to params.
                                   #  This is necessary when the integrand
                                   #  is a numba function.
                                   args=params)
    return np.exp(3 * int_de)


def f_dez(z: float, params: Params):
    """Integral of DE eos for Hubble function"""
    # noinspection PyTupleAssignmentBalance
    int_de, error = integrate.quad(f_dez_integrand, 0, z,
                                   epsabs=QUAD_EPS_ABS,
                                   # Note that args is equal to (params,).
                                   #  This is the correct procedure when the
                                   #  integrand is not a numba function.
                                   args=(params,))
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
model = Model(name="ONE",
              params_cls=Params,
              functions=functions)
