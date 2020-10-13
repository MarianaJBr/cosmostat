# Type hints for likelihood functions.
import typing as t
from dataclasses import asdict, dataclass, field

import numpy as np
from chisquarecosmo.constants_units import COVMATCMB_3

from .cosmology import Dataset, Model, Params, T_CosmologyFunc

# Type hints for likelihood functions.
T_LikelihoodTheoryFunc = T_CosmologyFunc
T_LikelihoodFunc = t.Callable[[Params], float]


@dataclass
class Likelihood:
    """Group likelihood functions for a specific model and dataset."""
    model: Model
    dataset: Dataset

    # Protected attributes.
    chi_square: T_LikelihoodFunc = field(init=False, default=None)

    def __post_init__(self):
        """"""
        self.chi_square = self._make_chi_square_func()

    def _make_residuals_func(self):
        """Build the residuals function."""
        data = self.dataset.data
        functions_dict = asdict(self.model.functions)
        theory_func = functions_dict[self.dataset.cosmology_func]

        def residuals_func(params: Params):
            """Returns a vector/array with the residuals."""
            zi_data, obs_data, err_data = data.T
            th_func = np.array([theory_func(zi, params) for zi in zi_data])
            return (obs_data - th_func) / err_data

        return residuals_func

    def _make_chi_square_func(self):
        """Build the chi-square function."""
        if self.dataset.name == "CMB":
            return self._make_cmb_chi_square_func()
        return self._make_standard_chi_square_func()

    def _make_standard_chi_square_func(self):
        """Build the chi-square function."""
        residuals_func = self._make_residuals_func()

        def standard_chi_square(params: Params):
            """Standard chi-square function."""
            return float(np.sum(residuals_func(params) ** 2))

        return standard_chi_square

    def _make_y_vec_cmb_func(self):
        """"""
        r_cmb = self.model.functions.r_cmb
        l_a = self.model.functions.l_a

        def y_vec_cmb(params: Params):
            """Standard chi-square function."""
            omegabh2 = params.omegabh2
            y_cmb = np.array([
                1.7382 - r_cmb(params),
                301.63 - l_a(params),
                0.02262 - omegabh2,
                # 0.9741 - 0.97415
            ])
            return y_cmb

        return y_vec_cmb

    def _make_cmb_chi_square_func(self):
        """Build the chi-square function."""
        y_cmb_func = self._make_y_vec_cmb_func()

        def cmb_chi_square(params: Params):
            """Standard chi-square function."""
            y_cmb = y_cmb_func(params)
            covmat = COVMATCMB_3  # 4x4 matrix, including ns
            covmat_inv = np.linalg.inv(covmat)
            # base_TTlowP.covmat from Planck DE&MG 2015 Table 4
            return float(np.dot(y_cmb, np.dot(covmat_inv, y_cmb)))

        return cmb_chi_square
