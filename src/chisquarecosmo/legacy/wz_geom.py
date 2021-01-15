from functools import lru_cache
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable, integrate
import numpy as np
import os

cw_directory = os.path.dirname(os.path.abspath(__file__))



# ==========   Numerical integration quantities for the calculation ===========
QUAD_EPSABS = 1.49e-8
inf = np.inf
quad = integrate.quad


# ---------------------------------------------------------------------#
# This part of the script contains the definition of a new equation
# of state for DE component and its integral to enter Hubble function:
#
# w(z) = -1 + w0/(1 + w1*z^2) cos(w2 + z)
# w_geom(z) = -1 + w0/(1 + w1*z**w3) cos(w2 + z) <<<<
# which recovers w= -1 for z>>0
# and has w(z=0) = w0cos(w2)-1
# ---------------------------------------------------------------------#


@jit(nopython=True, cache=True)
def wz(z, w_params):
    """
    -1 + (w0 / (1 + (w1) * z ** (w3))) * cos(z + w2)
    EoS for parametrizing f(R) behaviour
    w(z) = -1
    :param w3:
    """
    w0, w1, w2, w3 = w_params

    if w0 == 0:
        return -1

    if w3 == 0:
        return -1 + w0 / (1 + w1) * np.cos(z + w2)

    if w1 == 0:
        return -1 + w0 * np.cos(z + w2)

    return -1 + (w0 / (1 + (w1) * z ** (w3))) * np.cos(z + w2)



@jit(nopython=True, cache=True)
def f_DEz_integrand(zp, w_params):
    return (1 + wz(zp, w_params)) / (1 + zp)

# Signature of the numba C function.
f_DEz_integrand_cf_sig = types.double(
    types.int32, types.CPointer(types.double)
)

@cfunc(f_DEz_integrand_cf_sig, cache=True)
def f_DEz_integrand_cf(n, params_in_):
    """Numba C implementation of the integrand of the
    equation of state.

    :param n: Integer, it refers to the number of arguments of the
              integrand. Automatically guessed by numba.
    :param params_in_: A C pointer to the data passed to the integrand,
                       including the values of the variable integrated.
    :return:
    """
    # Converts the data C pointer into a numpy array view.
    params_array = carray(params_in_, n)

    # Extract data. First element is always the variable to be integrated.
    zp = params_array[0]
    w0 = params_array[1]
    w1 = params_array[2]
    w2 = params_array[3]
    w3 = params_array[4]

    w_params = w0, w1, w2, w3
    return (1 + wz(zp, w_params)) / (1 + zp)


# We have to wrap the ``ctypes`` implementation of the C-function
# with a LowLevelCallable.
f_DEz_integrand_cf_cc = LowLevelCallable(f_DEz_integrand_cf.ctypes)

@lru_cache(maxsize=1024 * 1024)
def f_DEz_cc(z, w_params):
    """Integral of DE eos for Hubble function using a
    LowLevelCallable.
    :param z:
    :param w_params:
    :return:
    """
    w0, w1, w2, w3 = w_params
    #w1, w2 = w_params
    intDE, error = integrate.quad(f_DEz_integrand_cf_cc, 0, z,
                                  epsabs=QUAD_EPSABS,
                                  args=w_params)
    return np.exp(3 * intDE)

@lru_cache(maxsize=1024 * 1024)
def f_DEz_sp(z, w_params):
    """Integral of DE eos for Hubble function"""
    intDE, error = integrate.quad(f_DEz_integrand, 0, z,
                                  epsabs=QUAD_EPSABS,
                                  args=(w_params,))
    return np.exp(3 * intDE)

f_DEz = f_DEz_cc

