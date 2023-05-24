import numpy as np
import os

from colossus.cosmology import cosmology as cocosmology
from colossus.lss import mass_function

from cosmosis.datablock import option_section

import compute_HMF_Despali16, compute_HMF_Castro22, compute_HMF_colossus

class EmptyClass:
    pass

def setup(options):
    # Proceed with actual setup
    tmp = options.get_double_array_1d(option_section, 'z_arr')
    z_arr = np.linspace(tmp[0], tmp[1], int(tmp[2]))
    tmp = options.get_double_array_1d(option_section, 'M_arr')
    M_arr = np.logspace(tmp[0], tmp[1], int(tmp[2]))
    fitting_function = options.get_string(option_section, 'fitting_function')
    if fitting_function=='Castro22':
        HMF_calculator = compute_HMF_Castro22.HMFCalculator(z_arr, M_arr)
    elif fitting_function=='Despali16':
        HMF_calculator = compute_HMF_Despali16.HMFCalculator(z_arr, M_arr)
    else:
        HMF_calculator = compute_HMF_colossus.HMFCalculator(z_arr, M_arr, fitting_function)
    return HMF_calculator

def execute(block, HMF_calculator):
    # Only need cosmo for E(z)-type stuff
    cosmology = {
            'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'Omega_nu': block.get_double('cosmological_parameters', 'Omega_nu'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'h': block.get_double('cosmological_parameters', 'hubble')/100.,
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8'),
            'ns': block.get_double('cosmological_parameters', 'n_s'),}
    # cdm+bar power spectrum (w/o neutrinos)
    z, k, Pk = block.get_grid('cdm_baryon_power_lin', 'z', 'k_h', 'p_k')
    # Compute the HMF
    dNdlnM_noVol, dNdlnM = HMF_calculator.compute_HMF(cosmology, z, k, Pk)
    # np.save('dNdlnM', [HMF_calculator.z_arr, HMF_calculator.M_arr, dNdlnM])
    # Put it into block
    block.put_grid('HMF', 'z_arr', HMF_calculator.z_arr, 'M_arr', HMF_calculator.M_arr, 'dNdlnM', dNdlnM)
    block.put_double_array_nd('HMF', 'dNdlnM_unitVol', dNdlnM_noVol)
    return 0

def cleanup(config):
    pass
