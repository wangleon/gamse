import dateutil.parser

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt

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
                for t in calibtable[mask]['obsdate']]
    irow = np.abs(timediff).argmin()
    row = calibtable[mask][irow]
    fileid = row['fileid']      # selected fileid
    md5 = row['md5']

    message = 'Select {} from database index as ThAr reference'.format(fileid)
    logger.info(message)

    filepath = os.path.join('espadons', 'wlcalib_{}.fits'.format(fileid))
    filepath = os.path.join(os.path.expanduser('~'), '.gamse', 'espadons')

    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    f = fits.open(filepath)
    head = f[0].header
    spec = f[1].data
    f.close()

    calib = get_calib_from_header(head)

    return spec, calib
