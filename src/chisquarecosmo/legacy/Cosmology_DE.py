# # # # # # # # ------------------------- # # #  # # # # # # # # # ## # #
# # # Mariana Jaber  2017(2018)(2019)
# # # This part of the program contains the basic cosmological parameters
# # # such as energy and physical densities for a Flat LCDM
# # # cosmology.
#     It is based on Planck 2015 Cosmological Parameters report
#     (arXiv:1502.01589) table 3 column 1:
#     "Planck + TT + lowP".
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # MAIN UPDATE 12.xii.19: BASE COSMOLOGY HAS BEEN CHANGED  from
# # # Planck TT + TE + EE + lowP to
# # # Planck TT + lowP
# -----------------------------------------
# # # 1502.01589 (Cosmological parameters)
# # # 1502.01590 (DE & MG)
# -----------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from functools import lru_cache
from math import sqrt

from chisquarecosmo.constants_units import *
from numba import jit
from scipy import integrate

# #we decide which equation of state to use:
# either geometric eos, Steep eos, CPL, BA, JJE, GGZ, gus
# from wz_BA import f_DEz
# from wz_CPL import f_DEz
# from wz_geom import f_DEz
# from wz_ggbz import f_DEz
from .wz_gus import f_DEz

# -----> Here:

# -----> Here:

# ==========   Numerical integration quantities for the calculation ===========
QUAD_EPSABS = 1.49e-8
inf = np.inf
quad = integrate.quad


# ------############ HUBBLE FUNCTION ################-----------#

# @jit
@lru_cache(maxsize=1024 * 1024)
def hubbleflat(z, w_params, cosmo_params):
    """
    Hubble function in terms of OmegaMatter
    flat universe
    """
    h, omegabh2, omegach2 = cosmo_params

    OmegaM = (omegabh2 + omegach2) / h ** 2
    # if OmegaM < 1:
    hubblefunc = H0P * h * sqrt(
        OMEGAR0 * (1 + z) ** 4 + OmegaM * (1 + z) ** 3 +
        (1 - OMEGAR0 - OmegaM) * f_DEz(z, w_params))
    # else:

    return hubblefunc


# @jit
@lru_cache(maxsize=1024 * 1024)
def Ezflat(z, w_params, cosmo_params):
    '''
    normalized Hubble function E(z) = H(z)/H0 for the flat Universe
    '''
    h, omegabh2, omegach2 = cosmo_params

    Hz = hubbleflat(z, w_params, cosmo_params)
    H0 = H0P * h

    return Hz / H0


# ---------------   Density for DE fluid  -------------------------#####

def rhode(z, w_params, cosmo_params):
    """
    volumentric energy density of the dark energy fluid component
    """
    h, omegabh2, omegach2 = cosmo_params

    OmegaM = (omegabh2 + omegach2) / h ** 2

    rhode0 = (1 - OmegaM - OMEGAR0) * RHOCR0
    return rhode0 * f_DEz(z, w_params)


# # # ------- Cosmic distances derived from the Hubble function-------####

# ===============        For BAO scale          =================
# ===============             BAO               =================

@jit(nopython=True)
def btog(z, h, omegabh2):
    """Baryon-to-photon ratio as function of z"""
    # return 3 * OMEGAB0 / (4 * OMEGAG0 * (1 + z))
    return 3 * (omegabh2 / h ** 2) / (4 * OMEGAG0 * (1 + z))


@lru_cache(maxsize=1024 * 1024)
def r_s_integrand(z, w_params, cosmo_params):
    """ flat universe """

    h, omegabh2, omegach2 = cosmo_params

    cs = 1 / sqrt(3 * (1 + btog(z, h, omegabh2)))

    return cs / hubbleflat(z, w_params, cosmo_params)


@lru_cache(maxsize=1024 * 1024)
def r_s(zeval, w_params, cosmo_params):
    """Sound horizon at zeval either zdrag or zdec"""
    # noinspection PyTupleAssignmentBalance
    r_sound, error = integrate.quad(r_s_integrand, zeval, inf,
                                    epsabs=QUAD_EPSABS,
                                    args=(w_params, cosmo_params)
                                    )
    return r_sound


def d_ang_integrand(zp, w_params, cosmo_params):
    """Integrand for the angular diameter distance: dz/H(z)"""
    """hubblefunc = H0p * h * sqrt(
        OMEGAR0 * (1 + z) ** 4 + (omegabh2+omegach2)/h**2 * (1 + z) ** 3 +
        (1 - OMEGAR0 - ((omegabh2+omegach2)/h**2)) * f_DEz(z, w_params))"""

    return 1 / hubbleflat(zp, w_params, cosmo_params)


def d_ang(z, w_params, cosmo_params):
    """Angular Diameter distance:
    D_a(z) = c/(1+z)Int_0^z(dz'/H(z'))
    """
    # noinspection PyTupleAssignmentBalance
    int, error = integrate.quad(d_ang_integrand, 0, z,
                                epsabs=QUAD_EPSABS,
                                args=(w_params, cosmo_params))
    return 1 / (1 + z) * int


@lru_cache(maxsize=1024 * 1024)
def d_vz_integrand(zp, w_params, cosmo_params):
    return 1 / hubbleflat(zp, w_params, cosmo_params)


@lru_cache(maxsize=1024 * 1024)
def d_vz(z, w_params, cosmo_params):
    """Dilation scale for rbao size """
    # noinspection PyTupleAssignmentBalance
    int2, error = integrate.quad(d_vz_integrand, 0, z,
                                 epsabs=QUAD_EPSABS,
                                 args=(w_params, cosmo_params))
    int3 = (z / hubbleflat(z, w_params, cosmo_params)) ** (1. / 3)
    return int3 * int2 ** (2 / 3)


def Rcmb(w_params, cosmo_params):
    """
    R(z*) = np.sqrt(Omega_M*H0*H0) D_ang(z*)/c
    """
    h, omegabh2, omegach2 = cosmo_params
    OmegaM = (omegabh2 + omegach2) / h ** 2

    factor1 = np.sqrt(
        OmegaM) * h * H0P  # important to keep H0p factor for units
    Dangzstar = d_ang(ZDEC, w_params, cosmo_params) * (1 + ZDEC)
    return factor1 * Dangzstar


@lru_cache(maxsize=1024 * 1024, typed=True)
def theta_star(w_params, cosmo_params):
    """ Angular sound horizon at decoupling. Used to calculate l_A in Wang &
    Mukherjee (2007) 's matrix  for the CMB compressed likelihood.
    :param w_params: equation of state parameters in the chosen eos w(z)
    :param cosmo_params: h, omega_b, omega_c
    :return: Thetha(z_dec) = r_s(z_dec) / D_V(z_dec)
    """
    # noinspection PyTupleAssignmentBalance
    int, error = integrate.quad(d_vz_integrand, 0, ZDEC,
                                epsabs=QUAD_EPSABS,
                                args=(w_params, cosmo_params))
    thethadec = r_s(ZDEC, w_params, cosmo_params) / int
    return thethadec


def l_A(w_params, cosmo_params):
    """Angular scale of the sound horizon at last scattering l_A:= pi/Theta_*
    Used in Wang & Mukherjee (2007) 's matrix for the CMB compressed likelihood.
    See section 5.1.6 of Planck 2015 DE & MG paper
    :param w_params: equation of state parameters in the chosen eos w(z)
    :param cosmo_params: h, omega_b, omega_c
    :return: l_A = pi/Theta(z*)
    """
    ts = theta_star(w_params, cosmo_params)
    return PI / ts


@lru_cache(maxsize=1024 * 1024, typed=True)
def rBAO(z, w_params, cosmo_params):
    """BAO scale at redshit z """
    rv = (
        r_s(ZDRAG, w_params, cosmo_params) /
        d_vz(z, w_params, cosmo_params))
    if isinstance(rv, complex):
        raise ValueError
    return rv


# =============== SNeIa distances definition    =================
# ===============           SNeIa               =================
def distance_SNe_integrand(z, w_params, cosmo_params):
    return 1 / hubbleflat(z, w_params, cosmo_params)


def distance_SNe(z, w_params, cosmo_params):
    '''
    Luminosity distance for SNe data
    :return: (1+z) * Int_0^z dz E^-1(z; args) where E(z) = H(z)/H0
    '''
    # noinspection PyTupleAssignmentBalance
    int, err = integrate.quad(distance_SNe_integrand, 0, z,
                              epsabs=QUAD_EPSABS,
                              args=(w_params, cosmo_params))
    return (1 + z) * int


def mu_SNe(z, w_params, cosmo_params):
    '''
     Modulus distance for SNe data: 5*log10 of dist_lum(z)
    :return: 5*Log10(distance_SNe in Mpc) + 25
    '''
    h, omegabh2, omegach2 = cosmo_params
    # OmegaM = (omegabh2 + omegach2) / h ** 2
    cosmo_params = h, omegabh2, omegach2
    d = distance_SNe(z, w_params, cosmo_params)
    mu0 = 25
    if d < 0:
        raise ValueError
    return 5 * np.log10(d) + mu0
