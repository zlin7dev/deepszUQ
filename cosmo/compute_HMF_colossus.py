import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from colossus.cosmology import cosmology as cocosmology
from colossus.lss import mass_function

import cosmo


class HMFCalculator:
    def __init__(self, z_arr, M_arr, model):
        """Initialize (fixed) arrays."""
        self.z_arr = z_arr
        self.M_arr = M_arr
        self.model = model

    def compute_HMF(self, cosmology, z, k, Pk):
        # Set up colossus cosmology
        co_params = {'flat': True}
        for me,co in zip(['h', 'Omega_m', 'Omega_b', 'sigma8', 'ns'],
                         ['H0', 'Om0', 'Ob0', 'sigma8', 'ns']):
            co_params[co] = cosmology[me]
        co_params['H0']*= 100
        _ = cocosmology.setCosmology('', co_params)
        # Mass function (per unit volume)
        dNdlnM_noVol = np.array([mass_function.massFunction(self.M_arr, z, mdef='vir',
                                                            model=self.model, q_in='M', q_out='dndlnM')
                                 for z in self.z_arr])
        # Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:,None]
        # Return HMF
        return dNdlnM_noVol, dNdlnM

