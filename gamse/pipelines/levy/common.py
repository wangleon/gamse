import os
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from ...echelle.trace import TraceFigureCommon

def print_wrapper(string, item):
    """A wrapper for log printing for APF/Levy pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype = item['imgtype']
    obj     = item['object'].lower().strip()

    if imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif obj in ['thar']:
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'

    elif obj in ['iodine']:
        # iodine cell, use light magenta (95)
        return '\033[95m'+string.replace('\033[0m', '')+'\033[0m'

    elif obj in ['wideflat']:
        # flat, use light red (91)
        return '\033[91m'+string.replace('\033[0m', '')+'\033[0m'

    elif obj in ['dark']:
        # dark images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    else:
        return string

def correct_overscan(data, head):
    """Correct the overscan of CCD image.

    Args:
        data (:class:`numpy.dtype`): Input data image.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.dtype`) – Output image with overscan 
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
            * **overmean** (*float*) -- Mean value of overscan region.
    """
    if data.shape==(4608, 2080):
        overmean = data[:,2049:2088].mean(axis=1)
        oversmooth = savgol_filter(overmean, window_length=1201, polyorder=3)
        #coeff = np.polyfit(np.arange(overmean.size), overmean, deg=7)
        #oversmooth2 = np.polyval(coeff, np.arange(overmean.size))
        res = (overmean - oversmooth).std()
        #fig = plt.figure(dpi=150)
        #ax = fig.gca()
        #ax.plot(overmean)
        #ax.plot(oversmooth)
        #ax.plot(oversmooth2)
        #plt.show()
        #plt.close(fig)
        overdata = np.tile(oversmooth, (2048, 1)).T
        newdata = data[:,0:2048] - overdata
        overmean = overdata.mean()

        card_lst = []
        prefix = 'HIERARCH GAMSE OVERSCAN '


        # update fits header
        head[prefix+'CORRECTED'] = True
        head[prefix+'METHOD']    = 'smooth'
        head[prefix+'AXIS-1']    = '2049:2088'
        head[prefix+'AXIS-2']    = '0:4608'
        head[prefix+'MEAN']      = overmean

        return newdata, card_lst, overmean

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self, datashape, figsize=(12,6)):
        TraceFigureCommon.__init__(self, figsize=figsize, dpi=150)

        axh = 0.85
        axw = axh/figsize[0]*figsize[1]/datashape[0]*datashape[1]
        x1 = 0.06
        self.ax1 = self.add_axes([x1,0.07,axw,axh])

        hgap = 0.06
        x2 = x1 + axw + hgap
        self.ax2 = self.add_axes([x2, 0.50, 0.93-x2, 0.40])
        self.ax3 = self.add_axes([x2, 0.10, 0.93-x2, 0.40])
        self.ax4 = self.ax3.twinx()

    def close(self):
        plt.close(self)
