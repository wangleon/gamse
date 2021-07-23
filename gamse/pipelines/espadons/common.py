import os
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ...echelle.background import BackgroundFigureCommon
from ...echelle.wlcalib import get_calib_from_header
from ...utils.regression import get_clip_mean
from ...utils.onedarray import iterative_savgol_filter
from ...utils.download import get_file

def correct_overscan(data, header):
    """Correct overscan.

    Args:
        data ():
        header ():
    Returns:

    """
    ny, nx = data.shape

    # get mask
    satmask = data >= 65535
    mask = np.int16(satmask)*4

    winlen = 501

    mean1 = data[:,0:20].mean(axis=1)
    mean1_ext = np.zeros((mean1.size+2*winlen),dtype=mean1.dtype)
    mean1_ext[winlen:winlen+mean1.size] = mean1
    mean1_ext[0:winlen] = mean1[0:winlen][::-1]
    mean1_ext[mean1.size+winlen:] = mean1[-winlen:][::-1]
    ovr1,_,_,_ = iterative_savgol_filter(mean1_ext,
                    winlen=winlen, order=3, upper_clip=3)
    ovr1 = ovr1[winlen:winlen+mean1.size]

    mean2 = data[:,nx-20:nx].mean(axis=1)
    mean2_ext = np.zeros((mean2.size+2*winlen),dtype=mean1.dtype)
    mean2_ext[winlen:winlen+mean2.size] = mean2
    mean2_ext[0:winlen] = mean2[0:winlen][::-1]
    mean2_ext[mean2.size+winlen:] = mean2[-winlen:][::-1]
    ovr2,_,_,_ = iterative_savgol_filter(mean2_ext,
                    winlen=winlen, order=3, upper_clip=3)
    ovr2 = ovr2[winlen:winlen+mean1.size]



    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mean1, lw=0.6, alpha=0.5)
    ax1.plot(ovr1, lw=0.6)

    ax2.plot(mean2, lw=0.6, alpha=0.6)
    ax2.plot(ovr2, lw=0.6)

    ax1.set_xlim(0,ny-1)
    ax2.set_xlim(0,ny-1)
    fig.savefig('{}_ovr.png'.format(fileid))
    plt.close(fig)
    '''

    scidata1 = data[:,20:nx//2]
    scidata2 = data[:,nx//2:nx-20]

    ovrimage1 = np.repeat([ovr1], scidata1.shape[1], axis=0).T
    ovrimage2 = np.repeat([ovr2], scidata2.shape[1], axis=0).T

    ovrdata = np.zeros((ny, nx-40), dtype=np.float64)
    ny1, nx1 = ovrdata.shape
    ovrdata[:, 0:nx1//2]   = scidata1 - ovrimage1
    ovrdata[:, nx1//2:nx1] = scidata2 - ovrimage2

    mask = mask[:,20:nx-20]
    return ovrdata, mask


def norm_profile(x, y):
    """Normalize the decker profile.

    Args:
        x ():
        y ():

    Returns:
        
    """

    x1, x2 = x[0], x[-1]
    y1, y2 = y[0], y[-1]
    background = (x-x1)/(x2-x1)*(y2-y1)+y1
    newy = y - background

    #v0, yp1, yp2, p1, p2 = find_local_peak(xnodes, ynodes)
    v0, p1, yp1, p2, yp2 = find_local_peak(x, newy)
    newx = x - v0
    Amean = (yp1+yp2)/2

    param = (v0, p1, p2, Amean, background.mean())

    if Amean < 1e-3:
        return None
    
    return newx, newy/Amean, param


def find_local_peak(x, y):
    n = x.size
    f = intp.InterpolatedUnivariateSpline(x, y, k=3)
    # find central valley. index=v0
    x0 = n/2
    i0 = int(round(x0))
    i1, i2 = i0-3, i0+3
    func = intp.InterpolatedUnivariateSpline(np.arange(i1, i2), y[i1:i2],
                k=3, ext=3)
    result = opt.minimize(func, x0)
    v0 = result.x[0] + x[0]
    yv0 = func(result.x[0])

    # find p1
    x0 = v0 - x[0] - 7
    i0 = int(round(x0))
    i1 = max(i0-3, 0)
    i2 = i1 + 6
    func = intp.InterpolatedUnivariateSpline(np.arange(i1, i2), -y[i1:i2],
                k=3, ext=3)
    result = opt.minimize(func, x0)
    p1 = result.x[0] + x[0]
    yp1 = -func(result.x[0])

    # find p2
    x0 = v0 - x[0] + 7
    i0 = int(round(x0))
    i2 = min(i0+3, y.size)
    i1 = i2 - 6
    func = intp.InterpolatedUnivariateSpline(np.arange(i1, i2), -y[i1:i2],
                k=3, ext=3)
    result = opt.minimize(func, x0)
    p2 = result.x[0] + x[0]
    yp2 = -func(result.x[0])

    '''
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y)
    ax.axvline(v0, ls='--')
    ax.axvline(p1, ls='--')
    plt.show()
    '''

    return v0, p1, yp1, p2, yp2

def get_mean_profile(xnodes, ynodes, p1, p2, step):
    xlst, ylst = [], []
    for xmid in np.arange(p1, p2+1e-5, step):
        x1, x2 = xmid-step/2, xmid+step/2
        mask = (xnodes > x1)*(xnodes < x2)
        mask = mask * (ynodes>0)
        if mask.sum()<=3:
            xlst.append(xmid)
            ylst.append(0.0)
        else:
            ymean, _, _ = get_clip_mean(ynodes[mask], high=3, low=3, maxiter=5)
            xlst.append(xmid)
            ylst.append(ymean)
    xlst = np.array(xlst)
    ylst = np.array(ylst)
    return xlst, ylst

def print_wrapper(string, item):
    """A wrapper for log printing for Xinglong216HRS pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    obstype = item['obstype']

    if obstype=='BIAS':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif obstype=='OBJECT':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif obstype=='COMPARISON':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

def select_calib_from_database(index_file, dateobs):
    """Select wavelength calibration file in database.

    Args:
        index_file (str): Index file of saved calibration files.
        dateobs (str): .

    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
              spectra.
            * **calib** (dict): Previous calibration results.
    """

    calibtable = Table.read(index_file, format='ascii.fixed_width_two_line')

    input_date = dateutil.parser.parse(dateobs)

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable['obsdate']]
    irow = np.abs(timediff).argmin()
    row = calibtable[irow]
    fileid = row['fileid']      # selected fileid
    md5 = row['md5']

    message = 'Select {} from database index as ThAr reference'.format(fileid)
    logger.info(message)

    filepath = os.path.join('espadons', 'wlcalib_{}.fits'.format(fileid))

    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    calib = get_calib_from_header(head)

    return spec, calib




class BackgroundFigure(BackgroundFigureCommon):
    def __init__(self, data=None, background=None, dpi=300, figsize=(6, 5),
            title=None, figname=None, zscale=('log', 'linear'), contour=True):

        ny, nx = data.shape
        BackgroundFigureCommon.__init__(self, figsize=figsize, dpi=dpi)
        width = 0.3
        # size[0]*width = size[1]*height for (1:1)
        # size[0]*width/(size[1]*height) = nx/ny
        # height = size[0]*width/size[1]/nx*ny
        height = width*figsize[0]/figsize[1]/nx*ny

        self.ax1  = self.add_axes([0.10, 0.1, width, height])
        self.ax2  = self.add_axes([0.55, 0.1, width, height])
        self.ax1c = self.add_axes([0.10+width+0.01, 0.1, 0.015, height])
        self.ax2c = self.add_axes([0.55+width+0.01, 0.1, 0.015, height])

        if data is not None and background is not None:
            self.plot_background(data, background,
                            zscale=zscale, contour=contour)
        if title is not None:
            self.suptitle(title)
        if figname is not None:
            self.savefig(figname)

    def plot_background(self, data, background, scale=(5, 99),
            zscale=('log', 'linear'), contour=True):
        """Plot the image data with background and the subtracted background
        light.

        Args:
            data (:class:`numpy.ndarray`): Image data to be background
                subtracted.
            background (:class:`numpy.ndarray`): Background light as a 2D array.
        """
        # find the minimum and maximum value of plotting

        if zscale[0] == 'linear':
            vmin = np.percentile(data, scale[0])
            vmax = np.percentile(data, scale[1])
            cax1 = self.ax1.imshow(data, cmap='gray', vmin=vmin, vmax=vmax,
                        origin='lower')
            # set colorbar
            cbar1 = self.colorbar(cax1, cax=self.ax1c)
        elif zscale[0] == 'log':
            m = data <= 0
            plotdata1 = np.zeros_like(data, dtype=np.float32)
            plotdata1[m] = 0.1
            plotdata1[~m] = np.log10(data[~m])
            vmin = np.percentile(plotdata1[~m], scale[0])
            vmax = np.percentile(plotdata1[~m], scale[1])
            cax1 = self.ax1.imshow(plotdata1, cmap='gray', vmin=vmin, vmax=vmax,
                        origin='lower')
            # set colorbar
            tick_lst = np.arange(int(np.ceil(vmin)), int(np.ceil(vmax)))
            ticklabel_lst = ['$10^{}$'.format(i) for i in tick_lst]
            cbar1 = self.colorbar(cax1, cax=self.ax1c, ticks=tick_lst)
            cbar1.ax.set_yticklabels(ticklabel_lst)
        else:
            print('Unknown zscale:', zscale)


        if zscale[1] == 'linear':
            vmin = background.min()
            vmax = background.max()
            cax2 = self.ax2.imshow(background, cmap='viridis',
                    vmin=vmin, vmax=vmax, origin='lower')
            # set colorbar
            cbar2 = self.colorbar(cax2, cax=self.ax2c)
        elif zscale[1] == 'log':
            m = background <= 0
            plotdata2 = np.zeros_like(background, dtype=np.float32)
            plotdata2[m] = 0.1
            plotdata2[~m] = np.log10(background[~m])
            vmin = max(0.1, background[~m].min())
            vmax = plotdata2[~m].max()
            cax2 = self.ax2.imshow(plotdata2, cmap='viridis',
                    vmin=vmin, vmax=vmax, origin='lower')
            # plot contour in background panel
            if contour:
                cs = self.ax2.contour(plotdata2, colors='r', linewidths=0.5)
                self.ax2.clabel(cs, inline=1, fontsize=7, use_clabeltext=True)
            # set colorbar
            tick_lst = np.arange(int(np.ceil(vmin)), int(np.ceil(vmax)))
            ticklabel_lst = ['$10^{}$'.format(i) for i in tick_lst]
            cbar2 = self.colorbar(cax2, cax=self.ax2c, ticks=tick_lst)
            cbar2.ax.set_yticklabels(ticklabel_lst)
        else:
            print('Unknown zscale:', zscale)

        # set labels and ticks
        self.ax1.set_ylabel('Y (pixel)', fontsize=8)
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X (pixel)', fontsize=8)
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(8)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(8)
        for cbar in [cbar1, cbar2]:
            for tick in cbar.ax.get_yaxis().get_major_ticks():
                tick.label2.set_fontsize(8)

class SpatialProfileFigure(Figure):
    """Figure to plot the cross-dispersion profiles.

    """
    def __init__(self,
            nrow = 2,
            ncol = 5,
            figsize = (12,8),
            dpi = 200,
            ):

        # create figure
        Figure.__init__(self, figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self)

        # add axes
        _w = 0.16
        _h = 0.40
        for irow in range(nrow):
            for icol in range(ncol):
                _x = 0.05 + icol*0.19
                _y = 0.05 + (nrow-1-irow)*0.45

                ax = self.add_axes([_x, _y, _w, _h])

    def close(self):
        plt.close(self)
