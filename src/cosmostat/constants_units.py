# # # Copyright Mariana Jaber
#     TODO: add license statement
# -----------------------------------------
#   \\\\\     BASIC CONSTANTS        \\\\\\
# -----------------------------------------

import numpy as np

PI = np.pi
C = 299792458.0
HBAR = 6.62606896 / (2 * PI) * 1e-34
E = 1.6021773 * 1e-19  # Electron unit charge
G = 6.6738 * 1e-11  # Newton's gravity constant
MPC = 3.085678 * 1e22  # Mpc-to-meters
KBOLTZ = 1.3806504 * 1e-23

# Planck's mass & Reduced Planck's mass in eV
MPLANCK = np.sqrt(HBAR * C / (8 * PI * G)) * C ** 2 / E
MPLANCK_REDUCED = MPLANCK * np.sqrt(8 * PI * G)

# CMB Temperature in K
TG0 = 2.7255
NEFF = 3.13  # 3.046   # 3.13 (from eqn. 60a-60d, arXiv:1502.01589)
THETHADEC = 1.04105e-2
TAU = 0.079
ZDRAG = 1059.57
ZDEC = 1090.09

# Hubble parameter:
REDUCED_H0 = 0.6731
H0 = (REDUCED_H0 * 1e5) / C
H0P = (100 * 1e3) / 2.99792458e8

# Critical density
RHOCR0 = 3 * (REDUCED_H0 * MPLANCK * E * MPC * 10 ** 5 / (C ** 2 * HBAR)) ** 2

# Radiation density
RHOR0 = (
    (1 + NEFF * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0))
    * PI ** 2
    / 15.0
    * TG0 ** 4
    * (KBOLTZ * MPC / (C * HBAR)) ** 4
)

# ------------------------------
# \\\\  The default value of the physical densities \\\\\\\\
# Planck's Cosmological parameters report
# # # 1502.01589 (Cosmological parameters)
# # # 1502.01590 (DE & MG)
# physical densities for different species: baryon, rad, photons,cdm and
# DE component
OMEGACH2 = 0.1197
OMEGABH2 = 0.02222

OMEGAR0 = RHOR0 / RHOCR0
OMEGAG0 = 4.64512e-31 / 8.53823e-27

OMEGAB0 = OMEGABH2 / (REDUCED_H0 ** 2)
OMEGADE = 0.685
OMEGADEH2 = (REDUCED_H0 ** 2) * OMEGADE
# fractional matter density
OMEGAM0 = OMEGAB0 + OMEGACH2 / (REDUCED_H0 ** 2)

RHOM0 = OMEGAM0 * RHOCR0
RHODE0 = OMEGADE * RHOCR0

# --------------------------------------------------------------------#
#
# DMATCMB: is the normalized matrix for R, l_A , omega_b and n_s
# Table 4 arXiv:1502.01590
# Planck TT + lowP and marginalizing over A_L
# --------------------------------------------------------------------#
#
DMATCMB_AL_b = np.array(
    [[1.0, 0.64, -0.75], [0.64, 1.0, -0.55], [-0.75, -0.55, 1.0]]
)
ERRCMB_AL_b = np.array([0.0088, 0.15, 0.00029])

# Update feb 12 2020: add the full matrix  to include ns
# 4x4 matrix
DMATCMB_AL_4 = np.array(
    [
        [1.0, 0.64, -0.75, -0.89],
        [0.64, 1.0, -0.55, -0.57],
        [-0.75, -0.55, 1.0, 0.71],
        [-0.89, -0.57, 0.71, 1.0],
    ]
)
ERRCMB_AL_4 = np.array([0.0088, 0.15, 0.00029, 0.0072])

COVMATCMB_b = DMATCMB_AL_b * ERRCMB_AL_b * ERRCMB_AL_b[:, np.newaxis]
COVMATCMB_4 = DMATCMB_AL_4 * ERRCMB_AL_4 * ERRCMB_AL_4[:, np.newaxis]
