import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d
import scipy.integrate
from math import sqrt as msqrt
import cosmo

DELTA_COLLAPSE = 1.68647
sqrt_pi = msqrt(np.pi)
# Rockstar Best-Fit
a1, a2, az, p1, p2, q1, q2, qz = [0.7962216913460142, 0.1448873400535069, -0.06577417849389455,
                                  -0.5612250094549993, -0.47425161063000365,
                                  0.3688494639810085, -0.2803717198857928, 0.0251485668327307]

def ST(v, a=0.707, p=0.3, q=1.0):
    a_vsqu = a * v**2
    #if (-p + q/2)<0.:
    #    A = 99.9
    #else:
    A = (2.**(-0.5-p+q/2) * (2**p * gamma(q/2) + gamma(-p + q/2) ) / sqrt_pi)**-1.
    fv = A * (1. + a_vsqu**-p) * np.sqrt(2.*a_vsqu/np.pi) * np.exp(-a_vsqu/2) * a_vsqu**((q-1)/2)
    return fv




class HMFCalculator:
    def __init__(self, z_arr, M_arr):
        """Initialize Tinker parameter interpolation functions."""
        self.z_arr = z_arr
        self.M_arr = M_arr

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
        # sigma^2 and dsigma^2/dM coarse [z, M_arr]
        sigma2 = .5/np.pi**2 * np.trapz(integrand_sigma2, np.log(k), axis=-1)
        dsigma2dM = np.pi**-2 * R[None,:]/self.M_arr[None,:]/3 * np.trapz(integrand_dsigma2dM, np.log(k), axis=-1)
        # sigma^2 and dsigma^2/dM fine [z_arr, M_arr]
        sigma2_fine = np.exp(interp1d(z, np.log(sigma2), axis=0)(self.z_arr))
        dsigma2dM_fine = -np.exp(interp1d(z, np.log(-dsigma2dM), axis=0)(self.z_arr))
        # dlnsigma/dlnR = R/sigma dsigma/dM dM/dR = R/sigma dsigma^2/dM/(2sigma) 4pi R^2
        dlnsigma_dlnR = 2*np.pi*R[None,:]**3/np.sqrt(sigma2_fine) * dsigma2dM_fine
        # peak height
        nu = delta_coll[:,None] / np.sqrt(sigma2_fine)
        # multiplicity function
        a = (a1 + a2*(dlnsigma_dlnR+0.6125)**2 ) * Omega_m_z[:,None]**az
        p = p1 + p2*(dlnsigma_dlnR+0.5)
        q = (q1 + q2*(dlnsigma_dlnR+0.5)) * Omega_m_z[:,None]**qz
        f = ST(nu, a, p, q)
        # Mass function
        dNdlnM_noVol = - f * rho_m * dsigma2dM_fine/2/sigma2_fine

        ##### Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:,None]
        # np.save('HMF_Castro', (self.z_arr, self.M_arr, dNdlnM))

        ##### Return HMF
        return dNdlnM_noVol, dNdlnM

