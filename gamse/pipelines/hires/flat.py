import numpy as np
import scipy.interpolate as intp

from ...utils.onedarray import iterative_savgol_filter

def smooth_flux(x, flux, nx):
    """Smooth flux
    """
    result = iterative_savgol_filter(flux,
                winlen=301, order=3, maxiter=10, lower_clip=3.0)
    ysmooth, yres, mask, std = result

    #f = intp.InterpolatedUnivariateSpline(x, ysmooth, k=3)
    return ysmooth
