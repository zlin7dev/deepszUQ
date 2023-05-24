import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate
import cosmo

DELTA_COLLAPSE = 1.68647

class HMFCalculator:
    def __init__(self, z_arr, M_arr):
        """Initialize Tinker parameter interpolation functions."""
        self.z_arr = z_arr
        self.M_arr = M_arr
        # All z, all cosmo, high mass
        self.a = .8199
        self.A = .3141


    def compute_HMF(self, cosmology, z, k, Pk):
        """Compute Despali HMF and apply redshift volume."""
        rho_m = (cosmology['Omega_m'] - cosmology['Omega_nu']) * cosmo.RHOCRIT
        Omega_m_z = cosmo.Omega_m_z(self.z_arr, cosmology)
        delta_coll = DELTA_COLLAPSE * Omega_m_z**.0055
        # Radius [M_arr]
        R = (3 * self.M_arr / (4 * np.pi * rho_m))**(1/3)
        # [M_arr, k]
        kR = k[None,:] * R[:,None]
        # Window functions [M_arr, k]
        window = 3 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        dwindow = 3/kR**4 * (3*kR*np.cos(kR) + ((kR**2 - 3)*np.sin(kR)))
        # Integrands [z_arr, M_arr, k]
        integrand_sigma2 = Pk[:,None,:] * window[None,:,:]**2 * k[None,None,:]**3
        integrand_dsigma2dM = Pk[:,None,:] * window[None,:,:] * dwindow[None,:,:] * k[None,None,:]**4
        # Sigma^2 and dsigma^2/dM [z_arr, M_arr]
        sigma2 = .5/np.pi**2 * np.trapz(integrand_sigma2, np.log(k), axis=-1)
        dsigma2dM = np.pi**-2 * R[None,:]/self.M_arr[None,:]/3 * np.trapz(integrand_dsigma2dM, np.log(k), axis=-1)
        sigma2_fine = np.exp(interp1d(z, np.log(sigma2), axis=0)(self.z_arr))
        dsigma2dM_fine = -np.exp(interp1d(z, np.log(-dsigma2dM), axis=0)(self.z_arr))
        # peak height
        nup = self.a * delta_coll[:,None]**2 / sigma2_fine
        # multiplicity function
        f = 2. * self.A * 2. * np.sqrt(nup/2/np.pi) * np.exp(-nup/2)
        dNdlnM_noVol = - f * rho_m * dsigma2dM_fine/2/sigma2_fine
        # np.save('HMF_Despali', [self.z_arr, self.M_arr, dNdlnM_noVol])
        ##### Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:,None]
        ##### Return HMF
        return dNdlnM_noVol, dNdlnM

