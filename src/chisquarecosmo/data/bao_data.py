import numpy as np

from chisquarecosmo.cosmology import Dataset

# Making use of all available data but WiggleZ and updating BOSS results
#  6dF, DR7, DR13-1, DR13-3, SDSS(R), LyA1 &  LyA2 removing WiggleZ and
#  replacing LOWZ and CMASS by the latest results from BOSS-DR13
REDSHIFTS_RBAO_7_NEW = np.array([0.106, 0.15, 0.38, 0.61, 0.35, 2.34, 2.36])
OBSERV_RBAO_7_NEW = np.array(
    [0.336, 0.2239, 0.1000, 0.0691, 0.113754, 0.0320, 0.0329])
ERRORS_RBAO_7_NEW = np.array(
    [0.015, 0.0084, 0.0011, 0.0007, 0.0021, 0.0013, 0.0009])
DATA_BAO_7_NEW = np.stack(
    (REDSHIFTS_RBAO_7_NEW, OBSERV_RBAO_7_NEW, ERRORS_RBAO_7_NEW), axis=1)

# Making use of all available data but WiggleZ
#  6dF, DR7, LOWZ, CMASS, SDSS(R), LyA1 &  LyA2 removing WiggleZ
REDSHIFTS_RBAO_7 = np.array([0.106, 0.15, 0.32, 0.57, 0.35, 2.34, 2.36])
OBSERV_RBAO_7 = np.array(
    [0.336, 0.2239, 0.1181, 0.0726, 0.113754, 0.0320, 0.0329])
ERRORS_RBAO_7 = np.array(
    [0.015, 0.0084, 0.0023, 0.0007, 0.0021, 0.0013, 0.0009])
DATA_BAO_7 = np.stack((REDSHIFTS_RBAO_7, OBSERV_RBAO_7, ERRORS_RBAO_7), axis=1)

REDSHIFTS_WIGGLEZ = np.array([0.44, 0.60, 0.73])
OBSERV_WIGGLEZ = np.array([0.0870, 0.0672, 0.0593])
ERRORS_WIGGLEZ = np.array(
    [[17.72, 6.9271, 0], [6.9271, 9.2720, 2.2243], [0, 2.2243, 4.1173]]) * 1e-6
INV_COVMAT_WIGGLEZ = np.linalg.inv(ERRORS_WIGGLEZ)

# Data suitable to be combined with the BAO correlated data points
#  6dF, DR7, LowZ, SDSS(R), LyA1, LyA2 (i.e. all but 0.57 and 0.70)
REDSHIFTS_RBAO = np.array([0.106, 0.15, 0.32, 0.35, 2.34, 2.36])
OBSERV_RBAO = np.array([0.336, 0.2239, 0.1181, 0.113754, 0.0320, 0.0329])
ERRORS_RBAO = np.array([0.015, 0.0084, 0.0023, 0.0021, 0.0013, 0.0009])
DATA_BAO = np.stack((REDSHIFTS_RBAO, OBSERV_RBAO, ERRORS_RBAO), axis=1)

bao_dataset = Dataset(name="BAO",
                      label="> BAO = 6dF,DR7,DR13-1,DR13-3,SDSS(R),LyA1,LyA2",
                      data=DATA_BAO_7_NEW,
                      cosmology_func="r_bao")
