from functools import lru_cache
from math import pi
from multiprocessing.pool import Pool

from chisquarecosmo.constants_units import *
from chisquarecosmo.data.bao_data import DATA_BAO_7_NEW, ERRORS_RBAO_7_NEW
from chisquarecosmo.data.hz_data import DATA_HZ, ERROR_HZ, REDSHIFTS_HZ
# from Input.SNeIa_bdata import DATA_SNe, REDSHIFTS_SNe, ERROR_SNe
from chisquarecosmo.data.sneia_union2_1 import (
    DATA_SNE_SQR, ERROR_SNE, REDSHIFTS_SNE
)
from chisquarecosmo.legacy.Cosmology_DE import (
    Rcmb, hubbleflat, l_A, mu_SNe,
    rBAO
)


# from progressbar import Bar, ETA, Percentage, ProgressBar


# ===============        For BAO scale          =================
# ===============             BAO               =================

@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAOuncorr(w_params, cosmo_params):
    """ chi2 function for BAO data that has no correlations, i.e.
     diagonal covariance matrix: 6DF, DR7, SDSS(R), BOSS DR13, no WIGGLEZ, Ly-a"""
    chi_sum = 0
    for (zi, obs, err) in DATA_BAO_7_NEW:
        # for (zi, obs, err) in DATA_BAO_old9:
        chi_sum += ((obs - rBAO(zi, w_params, cosmo_params)) / err) ** 2
    return chi_sum


@lru_cache(maxsize=1024 * 1024, typed=True)
def LikeBAOuncorr(w_params, cosmo_params):
    """Likelihood associated to the uncorrelated BAO chi-squared function """
    m = len(ERRORS_RBAO_7_NEW)

    return (
        np.exp(-chi2BAOuncorr(w_params, cosmo_params) / 2) / ((2 * pi) ** (
        m / 2)
                                                              * (
                                                                  1 / np.prod(
                                                                  ERRORS_RBAO_7_NEW)) ** (
                                                                  1 / 2))
    )


@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAO(w_params, cosmo_params):
    """total chi squared function for BAO data"""
    return (
        # chi2BAOcorr(w0, w1, w2, h, OmegaM) +
        chi2BAOuncorr(w_params, cosmo_params)
    )


def loglikeBAO(w_params, cosmo_params):
    return - chi2BAO(w_params, cosmo_params) / 2


# ===============        For SNeIa          =================
# ===============        UNION 2.1          =================

def chi2SNe(w_params, cosmo_params):
    '''
        Chi2 function for SNe  data from Union 2.1 sample (uncorrelated data)

    :param w_params:
    :param cosmo_params:
    :return:
    '''
    chi_sum = 0
    for (zi, obs, err) in DATA_SNE_SQR:
        # Errors are already squared both for JLA and Union 2.1 SNe datasets
        chi_sum += ((obs - mu_SNe(zi, w_params, cosmo_params))) ** 2 / err
    return chi_sum


@lru_cache(maxsize=1024 * 1024, typed=True)
def LikelihoodSNe(w_params, cosmo_params):
    """Likelihood associated to the binned SNe data """
    m = len(REDSHIFTS_SNE)
    return (
        np.exp(-chi2SNe(w_params, cosmo_params) / 2) / ((2 * pi) ** (
        m / 2) * (1 / np.prod(ERROR_SNE)) ** (1 / 2))
        # TODO: check the exponent in np.prod(Error) factor
        # UPDATE 11dec19: no error found in SNe chi2 vals. Exponent should be ok

    )


def loglikeSNe(w_params, cosmo_params):
    return -chi2SNe(w_params, cosmo_params) / 2


# ===============        For H(z)                =================
# ===============    from cosmic clocks          =================

def chi2hz(w_params, cosmo_params):
    ''' chi2 function for H(z) data from cosmic clocks '''
    chi_sum = 0
    for (zi, obs, err) in DATA_HZ:
        chi_sum += ((obs - 100 * hubbleflat(zi, w_params,
                                            cosmo_params) / H0P) / err) ** 2

    return chi_sum


def loglikehz(w_params, cosmo_params):
    return - chi2hz(w_params, cosmo_params) / 2


def likelihoodHz(w_params, cosmo_params):
    '''
    likelihood for the values of H(z) from cosmic clocks data
    :param w_params:  w0, w1, w2, w3...
    :param cosmo_params:  omegam, h0 (omegach2, omegabh2)
    :return:
    '''
    m = len(REDSHIFTS_HZ)
    return (
        np.exp(-chi2hz(w_params, cosmo_params) / 2) / ((2 * pi) ** (
        m / 2) * (1 / np.prod(ERROR_HZ)) ** (1 / 2))

    )


# =============== chi2 and likelihood functions =================
# ===============           CMB                 =================

def chi2CMB(w_params, cosmo_params):
    '''
    Calculates the chi2  for R(z*), l_A and omega_b (no n_s)
    Following Wang & Mukherjee (2007)
    Updated with values from Planck 2015 DE & MG paper. Table 4
    :param w_params: w_i of DE parametrization being used
    :param cosmo_params: omegabh2, omegach2 and H0
    :return: chi2_CMB(w_params, cosmo_params)
    '''
    h, omegabh2, omegach2 = cosmo_params
    ycmb = np.array([
        1.7382 - Rcmb(w_params, cosmo_params),
        301.63 - l_A(w_params, cosmo_params),
        0.02262 - omegabh2
        # ,
        # 0.9741 - 0.97415  # added to include full matrix
    ])
    covmat = COVMATCMB_3  # 4x4 matrix, including ns
    covmatinv = np.linalg.inv(covmat)
    # base_TTlowP.covmat from Planck DE&MG 2015 Table 4
    return np.dot(ycmb, np.dot(covmatinv, ycmb))


def likelihoodCMB(w_params, cosmo_params):
    """
    Likelihood function for the CMB quantities associated to the
    chi squared function.
    """
    covmat = COVMATCMB_4
    detcovmat = np.linalg.det(covmat)
    return (
        np.exp(- chi2CMB(w_params, cosmo_params) / 2) /
        (2 * pi * np.sqrt(detcovmat))
    )


# =============== Combining datasets: chi2 & likelihood =================
# ===============       BAO + CMB            =================

@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAOCMB(w_params, cosmo_params):
    """ Sum of BAO +CMB chi2 functions
    :param w_params:
    :param cosmo_params:
    :return: chi2TOTAL
    """
    c2b = chi2BAO(w_params, cosmo_params)
    c2c = chi2CMB(w_params, cosmo_params)

    chi2_total = c2b + c2c
    return chi2_total


@lru_cache(maxsize=1024 * 1024, typed=True)
def LikeliBAOCMB(w_params, cosmo_params):
    '''
    Product of  the likelihoods
    '''

    l_bao = likeBAOHz(w_params, cosmo_params)
    l_cmb = likelihoodCMB(w_params, cosmo_params)

    like_tot = l_bao * l_cmb
    return like_tot


@lru_cache(maxsize=1024 * 1024, typed=True)
def loglikeBAOCMB(w_params, cosmo_params):
    '''
    log of combined Likelihood
    :param w_params:
    :param cosmo_params:
    :return: - chi2_total/2
    '''
    return -chi2BAOCMB(w_params, cosmo_params) / 2


# ===============           BAO + Hz                   =================
# ===============                       =================
### --------------- more combinations of chi2 -----------
# ===============                       =================

@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAOHz(w_params, cosmo_params):
    """sum of BAO + H(z) chi2 functions"""

    return (
        chi2BAO(w_params, cosmo_params) +
        chi2hz(w_params, cosmo_params)
    )


def likeBAOHz(w_params, cosmo_params):
    """product of BAO and Hz likelihoods"""

    return (
        likelihoodHz(w_params, cosmo_params) *
        LikeBAOuncorr(w_params, cosmo_params)
    )


def loglikeBAOHz(w_params, cosmo_params):
    return - chi2BAOHz(w_params, cosmo_params) / 2


# ===============       BAO + SNe + CC            =================
@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAOSNeHz(w_params, cosmo_params):
    """ Sum of BAO + CC + SNe chi2 functions
    :param w_params:
    :param cosmo_params:
    :return: chi2TOTAL
    """
    c2b = chi2BAO(w_params, cosmo_params)
    c2s = chi2SNe(w_params, cosmo_params)
    c2h = chi2hz(w_params, cosmo_params)
    chi2_total = c2b + c2s + c2h
    return chi2_total


@lru_cache(maxsize=1024 * 1024, typed=True)
def LikeliBAOSNeHz(w_params, cosmo_params):
    '''
    Product of all the likelihoods
    '''
    l_bao = likeBAOHz(w_params, cosmo_params)
    l_u21 = LikelihoodSNe(w_params, cosmo_params)
    l_hz = likelihoodHz(w_params)

    like_tot = l_bao * l_u21 * l_hz
    return like_tot


@lru_cache(maxsize=1024 * 1024, typed=True)
def loglikeBAOSNeHz(w_params, cosmo_params):
    '''
    log of total Likelihood
    :param w_params:
    :param cosmo_params:
    :return: - chi2_total/2
    '''
    return -chi2BAOSNeHz(w_params, cosmo_params) / 2


# ===============       BAO + SNe + CC + CMB            =================
@lru_cache(maxsize=1024 * 1024, typed=True)
def chi2BAOSNeHzCMB(w_params, cosmo_params):
    """ Sum of BAO + CC + CMB + SNe chi2 functions
    :param w_params:
    :param cosmo_params:
    :return: chi2TOTAL
    """
    c2s = chi2SNe(w_params, cosmo_params)
    c2c = chi2CMB(w_params, cosmo_params)
    c2b = chi2BAO(w_params, cosmo_params)
    c2h = chi2hz(w_params, cosmo_params)
    chi2_total = c2s + c2c + c2b + c2h
    return chi2_total


@lru_cache(maxsize=1024 * 1024, typed=True)
def LikeliBAOSNeHzCMB(w_params, cosmo_params):
    '''
    Product of all the likelihoods
    '''
    l_u21 = LikelihoodSNe(w_params, cosmo_params)
    l_bao = likeBAOHz(w_params, cosmo_params)
    l_cmb = likelihoodCMB(w_params, cosmo_params)
    l_hz = likelihoodHz(w_params)

    like_tot = l_u21 * l_bao * l_bao * l_hz
    return like_tot


@lru_cache(maxsize=1024 * 1024, typed=True)
def loglikeBAOSNeHzCMB(w_params, cosmo_params):
    '''
    log of total Likelihood
    :param w_params:
    :param cosmo_params:
    :return: - chi2_total/2
    '''
    return -chi2BAOSNeHzCMB(w_params, cosmo_params) / 2


# ===============   Parallel griding    =================
# ===============                       =================

def chi2BAOuncorr_kernel(data):
    w_params, cosmo_params = data
    chi = chi2BAOuncorr(w_params, cosmo_params)
    # print("Completed chi2BAOuncorr for", data)
    # print("Value:", chi)
    return chi


# parallel implementation of chi2 function either CMB, BAO or BAO-CMB

def parallel_chi2any(kernel, data_array, processes=None):
    """Evaluates the chi2BAO function over the data saved in `data_array`
    and distributes the workload among several independent python
    processes.
    """

    # Let's create a pool of processes to execute calculate chi2BAO in
    # parallel.

    assert hasattr(data_array, 'dtype')

    data_values = data_array.shape[0]

    # percent = Percentage()
    # eta = ETA()
    # bar = Bar(marker='■')
    # progress_bar = ProgressBar(max_value=data_values,
    #                            widgets=[
    #                                '[', percent, ']',
    #                                bar,
    #                                '(', eta, ')'
    #                            ])

    with Pool(processes=processes) as pool:
        # The data accepted by the map method must be an iterable, like
        # a list, a tuple or a numpy array. The function is applied over
        # each element of the iterable. The result is another list with
        # the values returned by the function after being evaluated.
        #
        # [a, b, c, ...]  -> [f(a), f(b), f(c), ...]
        #
        # Here we use the imap method, so we need to create a list to
        # gather the results.
        results = []
        pool_imap = pool.imap(kernel, data_array)
        progress = 0
        for result in pool_imap:
            results.append(result)
            progress += 1
            # progress_bar.update(progress)

    return np.array(results)

# pp module parallel python: 1306.0573

# =============== Combining datasets: chi2 & likelihood =================
# ===============           BAO + SNe                   =================


# def chi2BAOSNe(w_params, cosmo_params):
#     """sum of BAO + H0 chi2 functions"""
#
#     h, OmegaM, mu0 = cosmo_params
#     cosmoparams = h, OmegaM
#
#     return (
#             chi2BAO(w_params, cosmoparams) +
#             chi2SNe(w_params, cosmo_params)
#     )
#
#
# def LikeliBAOSNe(w_params, cosmo_params):
#     """product of all the likelihood"""
#
#     h, OmegaM, mu0 = cosmo_params
#     cosmoparams = h, OmegaM
#
#     return (
#             LikelihoodSNe(w_params, cosmo_params) *
#             # LikeBAOcorr(w0, w1, w2, h, omegach2) *
#             LikeBAOuncorr(w_params, cosmoparams)
#     )
