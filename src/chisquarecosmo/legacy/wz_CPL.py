from functools import lru_cache
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable, integrate
import numpy as np

# ==========   Numerical integration quantities for the calculation ===========
QUAD_EPSABS = 1.49e-8
inf = np.inf
quad = integrate.quad


# ---------------------------------------------------------------------#
# This part of the script contains the definition of CPL equation
# of state for DE component and its integral to enter Hubble function:
#
# w(z) = w0 + wa * z/(1+z)
#      = w0 + (w1 - w0) * z /(1+z)
# with wa == w1 - w0
# ---------------------------------------------------------------------#


@jit(nopython=True, cache=True)
def wz(z, w_params):
    """
   w(z) = w0 + (w1 - w0) * z /(1+z)
    CPL
    """
    w0, w1 = w_params
    wa = w1 - w0

    if w0 == w1:
        return w0

    return w0 + wa * (z / (1 + z))




def f_DEzCPL(z, w_params):
    '''
    Analytcical integral for the CPL eos
    exp(3*Integral_o^z{[(1+w)/1+z]dz})  = exp(-3*wa*z/(1+z)) * (1+z)^{3(1+w0+wa)}
    :param z: redshift
    :param w_params: w0 and wa
    :return: exp(-3*wa*z/(1+z)) * (1+z)^{3(1+w0+wa)}
    '''
    w0, w1 = w_params
    wa = w1 - w0

    if np.abs(w1 - w0) < 1e-4:
        # wa = 0 and we keep the 2nd term only
        return (1 + z) ** (3 * (1 + w0))

    factor1 = np.exp(-3 * wa * z / (1 + z))
    factor2 = (1 + z) ** (3 * (1 + w0 + wa))

    return factor1 * factor2

f_DEz = f_DEzCPL

