import os

import numpy as np
from cosmostat.cosmology import Dataset

# Load data.
cw_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(cw_directory, 'Hznewdata.txt')
Hz_file = np.loadtxt(filename, usecols=(0, 1, 2))

REDSHIFTS_HZ = Hz_file[:, 0]
OBS_HZ = Hz_file[:, 1]
ERROR_HZ = Hz_file[:, 2]

# DATA_SNe = np.stack(REDSHIFTS_SNe, OBS_SNe, ERROR_SNe, axis=1)
DATA_HZ = Hz_file

dataset = Dataset(name="Hz",
                  label="> H(z)  (28)",
                  data=DATA_HZ,
                  cosmology_func="hz_hubble_flat")
