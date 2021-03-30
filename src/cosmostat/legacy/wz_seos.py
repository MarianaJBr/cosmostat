from functools import lru_cache
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable, integrate
import numpy as np



# ==========   Fixed quantities for the calculation =============

QUAD_EPSABS = 1.49e-8
inf = np.inf
quad = integrate.quad
exp = np.exp
# ----------------------



# ---------------------------------------------------------------------#
# This part of the script contains the definition of a Steep EoS
# for the DE component:
# w(z) = w0 + (wi-w0) (z/zt)^q/(1+(z/zt)^q)
# w(a) = w0 + wa * (1-a)^q/((a*zt)^q+(1-a)^q)
# where wa = wi-w0

@jit(nopython=True)
# @lru_cache(maxsize=1024 * 1024)
def wz(z, w_params):
    """This function parametrizes the eos for DE at low redshifts"""

    w0, wi, q, zt = w_params
    if zt == 0:
        return wi
    if q == 1:
        return w0 + (wi - w0) * ((z / zt) / (1 + (z / zt)))
    if w0 == wi:
        return w0

    return w0 + (wi - w0) * ((z / zt) ** q / (1 + (z / zt) ** q))


# @jit(nopython=True)
# @lru_cache(maxsize=1024 * 1024)
def wxa(avalue, w_params):
    """
    This function parametrizes the eos for DE at low redshifts
    as function of the scale factor
    """
    w0, wi, q, zt = w_params
    if zt == 0:
        return wi
    if avalue == 1:
        return w0
    return w0 + (wi - w0) * ((1 - avalue) ** q / ((avalue * zt) ** q + (1 - avalue) ** q))


@jit(nopython=True)
# @lru_cache(maxsize=1024 * 1024)
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
    wi = params_array[2]
    q = params_array[3]
    zt = params_array[4]

    w_params = w0, wi, q, zt
    return (1 + wz(zp, w_params)) / (1 + zp)


# We have to wrap the ``ctypes`` implementation of the C-function
# with a LowLevelCallable.

f_DEz_integrand_cf_cc = LowLevelCallable(f_DEz_integrand_cf.ctypes)


# @jit
@lru_cache(maxsize=1024 * 1024)
def f_DEz_cc(z, w_params):
    """Integral of DE eos for Hubble function using a
    LowLevelCallable.

    :param z:
    :param w_params:
    :return:
    """
    w0, wi, q, zt = w_params
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


# # @jit
# @lru_cache(maxsize=1024 * 1024)
# def f_DEz(z, w0=-1, wi=-1, q=1, zt=1):
#     """Integral of DE eos for Hubble function"""
#     intDE, error = integrate.quad(f_DEz_integrand, 0, z,
#                                   epsabs=QUAD_EPSABS,
#                                   args=(w0, wi, q, zt))
#     return exp(3 * intDE)


# TODO add a similar implementation for the integral in scale factor << DONE :)
def f_DElna_integrand(u, w_params):
    """Integrand in terms of u=lna variable"""
    w0, wi, q, zt = w_params
    return 1 / ((1 + ((zt * exp(u)) / (1 - exp(u))) ** q))


f_DElna_integrand_cf_sig = types.double(
    types.int32, types.CPointer(types.double)
)


@cfunc(f_DElna_integrand_cf_sig, cache=True)
def f_DElna_integrand_cf(n, params_in_):
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
    # Integration variable is lna = u
    u = params_array[0]
    w0 = params_array[1]
    wi = params_array[2]
    q = params_array[3]
    zt = params_array[4]

    w_params = w0, wi, q, zt

    return  1 / ((1 + ((zt * exp(u)) / (1 - exp(u))) ** q))

# We have to wrap the ``ctypes`` implementation of the C-function
# with a LowLevelCallable.
f_DElna_integrand_cf_cc = LowLevelCallable(f_DElna_integrand_cf.ctypes)


def f_DEa_sp(avalue, w_params):
    """
    integral of DE eos for Hubble function
    in terms of scale factor
    split in two terms: one analitically solved and a numerical integration
    """
    w0, wi, q, zt = w_params
    if avalue == 1:
        return 1
    intDEa, error = integrate.quad(f_DElna_integrand, 0, np.log(avalue),
                                   epsabs=QUAD_EPSABS,
                                   args=(q, zt))
    wa = wi - w0
    return 1 / (avalue ** (1 + w0)) * exp(-3 * wa * intDEa)


def f_DEa_cc(avalue, w_params):
    """
    integral of DE eos for Hubble function using a
    LowLevelCallable.

    in terms of scale factor
    split in two terms: one analitically solved and a numerical integration
    """
    w0, wi, q, zt = w_params
    if avalue == 1:
        return 1
    intDEa, error = integrate.quad(f_DElna_integrand_cf_cc, 0, np.log(avalue),
                                   epsabs=QUAD_EPSABS,
                                   args=(w_params,))
    wa = wi - w0
    return 1 / (avalue ** (1 + w0)) * exp(-3 * wa * intDEa)

f_DEa = f_DEa_cc
