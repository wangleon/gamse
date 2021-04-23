import os
import re
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.ticker as tck
from matplotlib.figure import Figure

from ...echelle.imageproc import combine_images
from ...echelle.trace import TraceFigureCommon

def print_wrapper(string, item):
    """A wrapper for log printing for Keck/HIRES pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype = item['imgtype']
    obj     = item['object']

    if len(obj)>=4 and obj[0:4]=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif len(obj)>=8 and obj[0:8]=='flatlamp':
        # flat images, analyze nsat
        nsat_1 = item['nsat_1']
        nsat_2 = item['nsat_2']
        nsat_3 = item['nsat_3']
        q95_1  = item['q95_1']
        q95_2  = item['q95_2']
        q95_3  = item['q95_3']
        q_lst = [q95_1 if q95_1 < 6e4 else -1,
                 q95_2 if q95_2 < 6e4 else -1,
                 q95_3 if q95_3 < 6e4 else -1]

        maxccd = np.argmax(q_lst)

        if max(q_lst)<0:
            # all CCDs are saturated
            return string

        elif 'quartz1' in obj and maxccd == 0:
            # quartz1 for UV, use light magenta (95)
            return '\033[95m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 0:
            # blue flat, use light blue (94)
            return '\033[94m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 1:
            # green flat, use light green (92)
            return '\033[92m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 2:
            # red flat, use light red (91)
            return '\033[91m'+string.replace('\033[0m', '')+'\033[0m'

        else:
            # no idea
            return string

    elif len(obj)>=7 and obj[0:7]=='arclamp':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

def parse_3ccd_images(hdu_lst):
    """Parse the 3 CCD images.

    Args:
        hdu_lst (:class:`astropy.io.fits.HDUList`): Input HDU list.

    Returns:
        tuple: A tuple containing:

            * **data_lst** (*tuple*): A tuple of (Image1, Image2, Image3).
            * **mask_lst** (*tuple*): A tuple of (Mask1, Mask2, Mask3).

    """
    if len(hdu_lst) != 4:
        raise ValueError

    # get CCD Binning
    tmp = hdu_lst[0].header['CCDSUM'].split()
    binx, biny = int(tmp[0]), int(tmp[1])
    # get data sect rectanle
    dataset_lst = {(2, 1): ('[7:1030,1:4096]', (6, 1030), (0, 4096)),
                   (2, 2): ('[7:1030,1:2048]', (6, 1030), (0, 2048)),
                  }
    datasec, (x1, x2), (y1, y2) = dataset_lst[(binx, biny)]
    # get data section
    data_lst = [hdu_lst[i+1].data[y1:y2, x1:x2] for i in range(3)
                if hdu_lst[i+1].header['DATASEC']==datasec]

    # get saturated masks
    mask_sat1 = data_lst[0]==65535   # for UV CCD, saturated pixels are 65535.
    mask_sat2 = data_lst[1]==0       # for green & red CCDs, saturated pixels
    mask_sat3 = data_lst[2]==0       # are 0.
    # get bad pixel masks
    #mask_bad1 = np.zeros_like(mask_sat1, dtype=np.bool)
    #mask_bad2 = np.zeros_like(mask_sat1, dtype=np.bool)
    #mask_bad3 = np.zeros_like(mask_sat1, dtype=np.bool)
    mask_bad1 = get_badpixel_mask((binx, biny), ccd=1)
    mask_bad2 = get_badpixel_mask((binx, biny), ccd=2)
    mask_bad3 = get_badpixel_mask((binx, biny), ccd=3)
    # pack masks
    mask1 = np.int16(mask_sat1)*4 + np.int16(mask_bad1)*2
    mask2 = np.int16(mask_sat2)*4 + np.int16(mask_bad2)*2
    mask3 = np.int16(mask_sat3)*4 + np.int16(mask_bad3)*2

    mask_lst = (mask1, mask2, mask3)

    # fix saturated pixels in the green and red CCDs
    data_lst[1][mask_sat2] = 65535
    data_lst[2][mask_sat3] = 65535

    return (data_lst, mask_lst)

def get_badpixel_mask(binning, ccd=0):
    """Get bad pixel mask for HIRES CCDs.

    Args:
        binning (tuple): CCD binning (*bin_x*, *bin_y*).
        ccd (int): CCD number.

    Returns:
        mask (:class:`numpy.ndarray`): Mask Image.

    """
    # for only 1 CCD
    if ccd == 0:
        if binning == (1, 1):
            # all Flase
            mask = np.zeros((2048, 2048), dtype=np.bool)
            mask[:,    1127] = True
            mask[:375, 1128] = True
            mask[:,    2007] = True
            mask[:,    2008] = True
    # for 3 CCDs
    elif ccd == 1:
        # for Blue CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[3878:,   4]  = True
            mask[3008:, 219]  = True
            mask[4005:, 337]  = True
            mask[1466:, 411]  = True
            mask[1466:, 412]  = True
            mask[3486:, 969]  = True
            mask[:,     994:] = True
    elif ccd == 2:
        # for Green CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[3726:, 323] = True
            mask[3726:, 324] = True
    elif ccd == 3:
        # for Red CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[1489:2196, 449]  = True
            mask[:,         0:45] = True
    return np.int16(mask)

def mosaic_3ccd_images(data_lst, mask_lst):
    """Mosaic three images.

    Args:
        data_lst (list): List of image data.
        mask_lst (list): List of mask data.

    Returns:
        tuple:
    """
    data1, data2, data3 = data_lst
    mask1, mask2, mask3 = mask_lst
    gap_rg, gap_gb = 26, 20

    # mosaic image: allimage and allmask
    h3, w3 = data3.shape
    h2, w2 = data2.shape
    h1, w1 = data1.shape

    hh = h3 + gap_rg + h2 + gap_gb + h1
    allimage = np.ones((hh, w3), dtype=data1.dtype)
    allmask = np.zeros((hh, w3), dtype=np.int16)
    r1, g1, b1 = 0, h3+gap_rg, h3+gap_rg+h2+gap_gb
    r2, g2, b2 = r1+h3, g1+h2, b1+h1
    allimage[r1:r2] = data3
    allimage[g1:g2] = data2
    allimage[b1:b2] = data1
    allmask[r1:r2] = mask3
    allmask[g1:g2] = mask2
    allmask[b1:b2] = mask1
    # fill gap with gap pixels
    allmask[r2:g1] = 1
    allmask[g2:b1] = 1

    return allimage, allmask

class BiasFigure(Figure):

    def __init__(self, dpi=200, figsize=(12,8), data_lst=None):
        Figure.__init__(self, dpi=dpi, figsize=figsize)
        nccd = 3
        axh = 0.8

        self.ax_imag_lst = []
        self.ax_ycut_lst = []
        self.ax_hist_lst = []
        self.ax_cbar_lst = []
        for iccd in range(nccd):
            data = data_lst[iccd]
            ny, nx = data.shape
            axw = axh/figsize[0]*figsize[1]/ny*nx
            ax_imag = self.add_axes([0.1+iccd*0.20, 0.07, axw, axh])
            ax_ycut = ax_imag.twiny()
            ax_hist = self.add_axes([0.7, 0.07+(nccd-iccd-1)*0.28+0.03, 0.25, 0.2])
            ax_cbar = self.add_axes([0.7, 0.07+(nccd-iccd-1)*0.28, 0.25, 0.02])

            self.ax_imag_lst.append(ax_imag)
            self.ax_ycut_lst.append(ax_ycut)
            self.ax_hist_lst.append(ax_hist)
            self.ax_cbar_lst.append(ax_cbar)

            vmin = np.percentile(data, 1)
            vmax = np.percentile(data, 99)
            cax = ax_imag.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
            self.colorbar(cax, cax=ax_cbar, orientation='horizontal')
            ax_imag.set_xlabel('X (pixel)')
            if iccd==0:
                ax_imag.set_ylabel('Y (pixel)')

            ax_imag.set_xlim(0, nx-1)
            ax_imag.set_ylim(0, ny-1)
            ax_imag.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax_imag.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax_imag.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax_imag.yaxis.set_minor_locator(tck.MultipleLocator(100))
            ax_imag.set_title('CCD {}'.format(iccd+1))
            #ax_ycut.set_ylim(0, ny-1)

            # plot histogram
            bins = np.linspace(vmin, vmax, 50)
            color = ('C0', 'C2', 'C3')[iccd]
            ax_hist.hist(data.flatten(), bins=bins, color=color,
                        label='CCD {}'.format(iccd+1))
            ax_hist.legend(loc='upper right')
            ax_hist.set_xticklabels([])
            ax_hist.set_xlim(vmin, vmax)
            ax_cbar.set_xlim(vmin, vmax)

        self.suptitle('Bias')


class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(13,6), dpi=300)
        self.ax1 = self.add_axes([0.06,0.07,0.49,0.86])
        self.ax2 = self.add_axes([0.60,0.55,0.35,0.352])
        self.ax3 = self.add_axes([0.60,0.13,0.35,0.352])
        self.ax4 = self.ax3.twinx()
