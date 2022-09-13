from socket import IP_TOS
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter1d
#import pickle5 as pickle
#from . import cosmo
import cosmo
import datetime

_STEP = 0


def lnlike(cosmology, M_arr, z_arr, massfunc, df, M_err=0, test_ratio=0.2,survey_area=4 * np.pi / 8.):
    global _STEP
    M_arr*= cosmology['mass_bias']
    deltaV = cosmo.deltaV(z_arr, cosmology)
    dNdlnM = massfunc*deltaV[:,None] * survey_area * test_ratio
    #filter the halos
    df = df[(df['Mvir'] <= M_arr.max()) & (df['Mvir'] >= M_arr.min())]
    df = df[(df['redshift'] <= z_arr.max()) & (df['redshift'] >= z_arr.min())]
    if M_err>0:
        Nbin = M_err/np.log(M_arr[1]/M_arr[0])
        dNdlnM = gaussian_filter1d(dNdlnM, Nbin, axis=1, mode='constant')
    dNdM_interp = RectBivariateSpline(z_arr, M_arr, dNdlnM/M_arr)
    N_total = dNdM_interp.integral(.1, 2, 1e14, 1e16)
    halo_contrib = np.log(np.array([dNdM_interp(df['redshift'][i], df['Mvir'][i])
                                    for i in df.index]))
    lnlike = np.sum(halo_contrib) - N_total
    #lnlike /= len(df)
    _STEP += 1
    print(datetime.datetime.now(), f"At Step {_STEP}, lnlike={lnlike}")
    if _STEP >= 10000: raise Exception()
    return lnlike

