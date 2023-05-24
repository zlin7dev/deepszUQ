import numpy as np
from math import sqrt as msqrt
import baccoemu
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from cosmosis.datablock import option_section


def setup(options):
    """Return redshift array and the emulator object"""
    emulator = baccoemu.Matter_powerspectrum()
    z_min_max = options.get_double_array_1d(option_section, 'z_min_max')
    N_z = options.get_int(option_section, 'N_z')
    z_arr = np.linspace(z_min_max[0], z_min_max[1], N_z)
    return z_arr, emulator


def execute(block, stuff):
    """Read cosmological parameters, run power spectrum emulator, and write to
    block."""
    # Setup
    z_arr, emulator = stuff
    # Call the emulator for P_{CDM+bar}(k)
    k, Pk = emulator.get_linear_pk(omega_matter=block.get_double('cosmological_parameters', 'Omega_m'),
                                   omega_baryon=block.get_double('cosmological_parameters', 'Omega_b'),
                                   hubble=block.get_double('cosmological_parameters', 'h0'),
                                   ns=block.get_double('cosmological_parameters', 'n_s'),
                                   w0=block.get_double('cosmological_parameters', 'w'),
                                   wa=block.get_double('cosmological_parameters', 'wa'),
                                   neutrino_mass=block.get_double('cosmological_parameters', 'Omnuh2')*94.,
                                   A_s=block.get_double('cosmological_parameters', 'A_s'),
                                   expfactor=1./(1.+z_arr),
                                   cold=True)
    # Compute sigma_8 for total matter
    k_, Pk_ = emulator.get_linear_pk(omega_matter=block.get_double('cosmological_parameters', 'Omega_m'),
                                     omega_baryon=block.get_double('cosmological_parameters', 'Omega_b'),
                                     hubble=block.get_double('cosmological_parameters', 'h0'),
                                     ns=block.get_double('cosmological_parameters', 'n_s'),
                                     w0=block.get_double('cosmological_parameters', 'w'),
                                     wa=block.get_double('cosmological_parameters', 'wa'),
                                     neutrino_mass=block.get_double('cosmological_parameters', 'Omnuh2')*94.,
                                     A_s=block.get_double('cosmological_parameters', 'A_s'),
                                     expfactor=1.,
                                     cold=False)
    kR = 8.*k_
    window = 3. * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
    integrand_sigma2 = Pk_ * window**2 * k_**3
    sigma8_squ = .5/np.pi**2 * np.trapz(integrand_sigma2, np.log(k_))
    # Write to block
    block.put_double('cosmological_parameters', 'sigma_8', msqrt(sigma8_squ))
    block.put_grid('cdm_baryon_power_lin', 'z', z_arr, 'k_h', k, 'p_k', Pk)
    # np.save('pk', (z_arr, k, Pk))
    # print('sigma_8', msqrt(sigma8_squ))

    return 0


def cleanup(config):
    pass
