import typing as t
from dataclasses import asdict, dataclass

import numpy as np
from numba import jit
from scipy import integrate

from cosmostat.constants_units import OMEGABH2, OMEGACH2, REDUCED_H0
from cosmostat.cosmology import Functions as BaseFunctions
from cosmostat.cosmology import Model
from cosmostat.cosmology import Params as ParamsBase
from cosmostat.cosmology import T_CosmologyFunc

# ==========   Numerical integration quantities for the calculation ===========
QUAD_EPS_ABS = 1.49e-8
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
class Params(ParamsBase, t.NamedTuple):
    """"""

    w0: float
    w1: float
    h: float = REDUCED_H0
    omegabh2: float = OMEGABH2
    omegach2: float = OMEGACH2


@jit(nopython=True, cache=True)
def wz(z: float, params: Params):
    """
    w(z) = w0 + (w1 - w0) * z /(1+z)
     CPL
    """
    w0 = params.w0
    w1 = params.w1
    wa = w1 - w0
    if w0 == w1:
        return w0
    return w0 + wa * (z / (1 + z))


def f_dez(z: float, params: Params):
    """Analytical integral for the CPL eos.

    exp(3*Integral_o^z{[(1+w)/1+z]dz}) = \
        exp(-3*wa*z/(1+z)) * (1+z)^{3(1+w0+wa)}
    :param z: redshift
    :param params: w0 and wa
    :return: exp(-3*wa*z/(1+z)) * (1+z)^{3(1+w0+wa)}
    """
    w0 = params.w0
    w1 = params.w1
    wa = w1 - w0
    if np.abs(w1 - w0) < 1e-4:
        # wa = 0 and we keep the 2nd term only
        return (1 + z) ** (3 * (1 + w0))
    factor1 = np.exp(-3 * wa * z / (1 + z))
    factor2 = (1 + z) ** (3 * (1 + w0 + wa))
    return factor1 * factor2


@dataclass
class Functions(BaseFunctions):
    """"""

    def _make_wz_func(self) -> T_CosmologyFunc:
        """"""
        return wz

    def _make_f_dez_func(self) -> T_CosmologyFunc:
        """"""
        return f_dez


# Singleton with the model functions.
functions = Functions()

# Dictionary of the functions.
functions_dict = asdict(functions)

# Singleton with the model definition.
model = Model(name="CPL", params_cls=Params, functions=functions)
