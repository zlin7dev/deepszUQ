import numpy as np
from colossus.cosmology import cosmology as cocosmology
from colossus.lss import mass_function

from cosmosis.datablock import option_section

import deep_proj
import pandas as pd

def _read_dataframe(fname='20220413_MF.csv'):
    dataframe = pd.read_csv(fname)
    assert 'redshift' in dataframe.columns and 'Mvir' in dataframe.columns
    if 'split' not in dataframe:
        import numpy as np
        dataframe = dataframe.iloc[np.random.RandomState(7).choice(dataframe.index, int(0.2 * len(dataframe)), replace=False)]
    else:
        dataframe = dataframe[dataframe['split'] == 'test']
    return dataframe.reset_index().reindex(columns=['redshift', 'Mvir'])

def setup(options):
    fname = options.get_string(option_section, 'dataframe')
    return _read_dataframe(fname)

def execute(block, setup_stuff):
    cosmology = {
                 'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_b': 0.044, #block.get_double('cosmological_parameters', 'Omega_b'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa'),
                 'h': block.get_double('cosmological_parameters', 'hubble')/100,
                 'sigma8': block.get_double('cosmological_parameters', 'sigma_8'),
                 'ns': block.get_double('cosmological_parameters', 'n_s'),
                 'mass_bias': block.get_double('cosmological_parameters', 'mass_bias'),
                 'Tcmb0' : 2.726,
    }
    M_err = 0
    co_params = {'flat': True}
    for me,co in zip(['h', 'Omega_m', 'Omega_b', 'sigma8', 'ns', 'Tcmb0'],
                     ['H0', 'Om0', 'Ob0', 'sigma8', 'ns', 'Tcmb0']):
        co_params[co] = cosmology[me]
    co_params['H0']*= 100

    _ = cocosmology.setCosmology('', co_params)

    M_arr = np.logspace(13.9,16,256)
    z_arr = np.linspace(.1,2,128)
    massfunc = np.array([mass_function.massFunction(M_arr, z, mdef='vir', model='despali16', q_in='M', q_out='dndlnM')
                         for z in z_arr])
    
    lnlike = deep_proj.lnlike(cosmology, M_arr, z_arr, massfunc, setup_stuff, M_err)
    block.put_double('likelihoods', 'abundance_LIKE', lnlike)

    return 0

def cleanup(config):
    pass
