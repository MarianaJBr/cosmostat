import os

import numpy as np

# import sys

cw_directory = os.path.dirname(os.path.abspath(__file__))
# print(cw_directory)


# cw_directory = os.getcwd()
# print(nb_directory)
# sys.path.insert(0, os.path.join(nb_directory, '..','python code'))
# sys.path.insert(0, os.path.join(nb_directory, '..','python code', 'exec'))
# sys.path.insert(0, os.path.join(nb_directory, '..','python code', 'chains'))


# file_path = os.path.dirname(os.path.abspath(__file__))
# print(file_path)

# sys.path.insert(0, os.path.join(file_path, '..'))
sne_type = ['JLA', 'Union']
# TODO: add the 2 datasets in one file and choose via an If loop.
# TODO: extend to chi2_likelihood script in the SNe routine
# TODO: if SNeType == JLA then use file such, else use the Union2.1
# TODO: if SNeType == JLA then use Error as is, else use Error**2
filename = os.path.join(cw_directory, 'jla_mub_complete.txt')
sne_file = np.loadtxt(filename, usecols=(1, 2, 3))

REDSHIFTS_SNE = sne_file[:, 0]
OBS_SNE = sne_file[:, 1]
ERROR_SNE = sne_file[:, 2]

# DATA_SNe = np.stack(REDSHIFTS_SNe, OBS_SNe, ERROR_SNe, axis=1)
DATA_SNE = sne_file
