import os
import re
import sys
import math
import datetime
import dateutil.parser
import itertools
import logging
logger = logging.getLogger(__name__)

import numpy as np
import numpy.polynomial as poly
import astropy.io.fits as fits
import scipy.interpolate as intp
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.figure import Figure

from ..utils.regression2d import polyfit2d, polyval2d
from ..utils.onedarray    import pairwise
from .trace import load_aperture_set_from_header

# Data format for identified line table
identlinetype = np.dtype({
    'names':  ['aperture','order','pixel','wavelength','amplitude',
                'fwhm','q','mask','residual','method'],
    'formats':[np.int16, np.int16, np.float32, np.float64, np.float32,
                np.float32, np.float32, np.int16, np.float64, 'S1'],
    })

def fit_wavelength(identlist, npixel, xorder, yorder, maxiter, clipping,
        fit_filter=None):
    """Fit the wavelength using 2-D polynomial.

    Args:
        identlist (dict): Dict of identification lines for different apertures.
        npixel (int): Number of pixels for each order.
        xorder (int): Order of polynomial along X direction.
        yorder (int): Order of polynomial along Y direction.
        maxiter (int): Maximim number of iterations in the polynomial
            fitting.
        clipping (float): Threshold of sigma-clipping.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        tuple: A tuple containing:

            * **coeff** (:class:`numpy.ndarray`) -- Coefficients array.
            * **std** (*float*) -- Standard deviation.
            * **k** (*int*) -- *k* in the relationship between aperture
              numbers and diffraction orders: `order = k*aperture + offset`.
            * **offset** (*int*) -- *offset* in the relationship between
              aperture numbers and diffraction orders: `order = k*aperture +
              offset`.
            * **nuse** (*int*) -- Number of lines used in the fitting.
            * **ntot** (*int*) -- Number of lines found.

    See also:
        :func:`get_wavelength`
    """
    # find physical order
    k, offset = find_order(identlist, npixel)

    # parse the fit_filter=None
    if fit_filter is None:
        fit_filter = lambda item: True

    # convert indent_line_lst into fitting inputs
    fit_p_lst = []  # normalized pixel
    fit_o_lst = []  # diffraction order
    fit_w_lst = []  # order*wavelength
    fit_m_lst = []  # initial mask
    # the following list is used to find the position (aperture, no)
    # of each line
    lineid_lst = []
    for aperture, list1 in sorted(identlist.items()):
        order = k*aperture + offset
        #norm_order = 50./order
        #norm_order = order/50.
        list1['order'][:] = order
        for iline, item in enumerate(list1):
            norm_pixel = item['pixel']*2/(npixel-1) - 1
            fit_p_lst.append(norm_pixel)
            fit_o_lst.append(order)
            #fit_o_lst.append(norm_order)
            #fit_w_lst.append(item['wavelength'])
            fit_w_lst.append(item['wavelength']*order)
            fit_m_lst.append(fit_filter(item))
            lineid_lst.append((aperture, iline))
    fit_p_lst = np.array(fit_p_lst)
    fit_o_lst = np.array(fit_o_lst)
    fit_w_lst = np.array(fit_w_lst)
    fit_m_lst = np.array(fit_m_lst)

    mask = fit_m_lst

    for nite in range(maxiter):
        coeff = polyfit2d(fit_p_lst[mask], fit_o_lst[mask], fit_w_lst[mask],
                          xorder=xorder, yorder=yorder)
        res_lst = fit_w_lst - polyval2d(fit_p_lst, fit_o_lst, coeff)
        res_lst = res_lst/fit_o_lst

        mean = res_lst[mask].mean(dtype=np.float64)
        std  = res_lst[mask].std(dtype=np.float64)
        m1 = res_lst > mean - clipping*std
        m2 = res_lst < mean + clipping*std
        new_mask = m1*m2*mask
        if new_mask.sum() == mask.sum():
            break
        else:
            mask = new_mask

    # convert mask back to ident_line_lst
    for lineid, ma, res in zip(lineid_lst, mask, res_lst):
        aperture, iline = lineid
        identlist[aperture][iline]['mask']     = ma
        identlist[aperture][iline]['residual'] = res

    # number of lines and used lines
    nuse = mask.sum()
    ntot = fit_w_lst.size
    return coeff, std, k, offset, nuse, ntot

def get_wavelength(coeff, npixel, pixel, order):
    """Get wavelength.

    Args:
        coeff (:class:`numpy.ndarray`): 2-D Coefficient array.
        npixel (int): Number of pixels along the main dispersion direction.
        pixel (*int* or :class:`numpy.ndarray`): Pixel coordinates.
        order (*int* or :class:`numpy.ndarray`): Diffraction order number.
            Must have the same length as **pixel**.

    Returns:
        float or :class:`numpy.ndarray`: Wavelength solution of the given pixels.

    See also:
        :func:`fit_wavelength`
    """
    # convert aperture to order
    norm_pixel = pixel*2./(npixel-1) - 1
    #norm_order  = 50./order
    #norm_order  = order/50.
    return polyval2d(norm_pixel, order, coeff)/order

def guess_wavelength(x, aperture, identlist, linelist, param):
    """Guess wavelength according to the identified lines.
    First, try to guess the wavelength from the identified lines in the same
    order (aperture) by fitting polynomials.
    If failed, find the rough wavelength the global wavelength solution.
    Finally, pick up the closet wavelength from the wavelength standards.

    Args:
        x (float): Pixel coordinate.
        aperture (int): Aperture number.
        identlist (dict): Dict of identified lines for different apertures.
        linelist (list): List of wavelength standards.
        param (dict): Parameters of the :class:`CalibWindow`.

    Returns:
        float: Guessed wavelength. If failed, return *None*.
    """
    rough_wl = None

    # guess wavelength from the identified lines in this order
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size >= 2:
            fit_order = min(list1.size-1, 2)
            local_coeff = np.polyfit(list1['pixel'], list1['wavelength'],
                            deg=fit_order)
            rough_wl = np.polyval(local_coeff, x)

    # guess wavelength from global wavelength solution
    if rough_wl is None and param['coeff'].size > 0:
        npixel = param['npixel']
        order = aperture*param['k'] + param['offset']
        rough_wl = get_wavelength(param['coeff'], param['npixel'], x, order)

    if rough_wl is None:
        return None
    else:
        # now find the nearest wavelength in linelist
        wave_list = np.array([line[0] for line in linelist])
        iguess = np.abs(wave_list-rough_wl).argmin()
        guess_wl = wave_list[iguess]
        return guess_wl

def is_identified(wavelength, identlist, aperture):
    """Check if the input wavelength has already been identified.

    Args:
        wavelength (float): Wavelength of the input line.
        identlist (dict): Dict of identified lines.
        aperture (int): Aperture number.

    Returns:
        bool: *True* if **wavelength** and **aperture** in **identlist**.
    """
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size==0:
            # has no line in this aperture
            return False
        diff = np.abs(list1['wavelength'] - wavelength)
        if diff.min()<1e-3:
            return True
        else:
            return False
    else:
        return False

def find_order(identlist, npixel):
    """Find the linear relation between the aperture numbers and diffraction
    orders.
    The relationship is `order = k*aperture + offset`.
    Longer wavelength has lower order number.

    Args:
        identlist (dict): Dict of identified lines.
        npixel (int): Number of pixels along the main dispersion direction.

    Returns:
        tuple: A tuple containing:

            * **k** (*int*) -- Coefficient in the relationship
              `order = k*aperture + offset`.
            * **offset** (*int*) -- Coefficient in the relationship
              `order = k*aperture + offset`.
    """
    aper_lst, wlc_lst = [], []
    for aperture, list1 in sorted(identlist.items()):
        if list1.size<3:
            continue
        less_half = (list1['pixel'] < npixel/2).sum()>0
        more_half = (list1['pixel'] > npixel/2).sum()>0
        #less_half, more_half = False, False
        #for pix, wav in zip(list1['pixel'], list1['wavelength']):
        #    if pix < npixel/2.:
        #        less_half = True
        #    elif pix >= npixel/2.:
        #        more_half = True
        if less_half and more_half:
            if list1['pixel'].size>2:
                deg = 2
            else:
                deg = 1
            c = np.polyfit(list1['pixel'], list1['wavelength'], deg=deg)
            wlc = np.polyval(c, npixel/2.)
            aper_lst.append(aperture)
            wlc_lst.append(wlc)
    aper_lst = np.array(aper_lst)
    wlc_lst  = np.array(wlc_lst)
    if wlc_lst[0] > wlc_lst[-1]:
        k = 1
    else:
        k = -1

    offset_lst = np.arange(-500, 500)
    eva_lst = []
    for offset in offset_lst:
        const = (k*aper_lst + offset)*wlc_lst
        diffconst = np.diff(const)
        eva = (diffconst**2).sum()
        eva_lst.append(eva)
    eva_lst = np.array(eva_lst)
    offset = offset_lst[eva_lst.argmin()]

    return k, offset

def save_ident(identlist, coeff, filename):
    """Write the ident line list and coefficients into an ASCII file.
    The existing informations in the ASCII file will not be affected.
    Only the input channel will be overwritten.

    Args:
        identlist (dict): Dict of identified lines.
        coeff (:class:`numpy.ndarray`): Coefficient array.
        result (dict): A dict containing identification results.
        filename (str): Name of the ASCII file.

    See also:
        :func:`load_ident`
    """
    outfile = open(filename, 'w')

    # write identified lines
    fmtstr = ('LINE {:03d} {:10.4f} {:10.4f} {:10.4f} {:12.5e} {:9.5f}'
             '{:1d} {:+10.6f} {:1s}')
    for aper, list1 in sorted(identlist.items()):
        for row in list1:
            pix    = row['pixel']
            wav    = row['wavelength']
            amp    = row['amplitude']
            fwhm   = row['fwhm']
            mask   = int(row['mask'])
            res    = row['residual']
            method = row['method'].decode('ascii')
            outfile.write(fmtstr.format(aper, pix, wav, amp, fwhm,
                mask, res, method)+os.linesep)

    # write coefficients
    for irow in range(coeff.shape[0]):
        string = ' '.join(['{:18.10e}'.format(v) for v in coeff[irow]])
        outfile.write('COEFF {}'.format(string)+os.linesep)

    outfile.close()


def load_ident(filename):
    """Load identified line list from an ASCII file.

    Args:
        filename (str): Name of the identification file.

    Returns:
        tuple: A tuple containing:

            * **identlist** (*dict*) -- Identified lines for all orders.
            * **coeff** (:class:`numpy.ndarray`) -- Coefficients of wavelengths.

    See also:
        :func:`save_ident`
    """
    identlist = {}
    coeff = []

    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#$%^@!':
            continue
        g = row.split()

        key = g[0]
        if key == 'LINE':
            aperture    = int(g[1])
            pixel       = float(g[2])
            wavelength  = float(g[3])
            amplitude   = float(g[4])
            fwhm        = float(g[5])
            mask        = bool(g[6])
            residual    = float(g[7])
            method      = g[8].strip()

            item = np.array((aperture, 0, pixel, wavelength, amplitdue, fwhm,
                    0., mask, residual, method),dtype=identlinetype)

            if aperture not in identlist:
                identlist[aperture] = []
            identlist[aperture].append(item)

        elif key == 'COEFF':
            coeff.append([float(v) for v in g[2:]])

        else:
            pass

    infile.close()

    # convert list of every order to numpy structured array
    for aperture, list1 in identlist.items():
        identlist[aperture] = np.array(list1, dtype=identlinetype)

    # convert coeff to numpy array
    coeff = np.array(coeff)

    return identlist, coeff

def gaussian(A,center,fwhm,x):
    sigma = fwhm/2.35482
    return A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc(p,x,y):
    return y - gaussian(p[0],p[1],p[2],x)

def gaussian_bkg(A,center,fwhm,bkg,x):
    sigma = fwhm/2.35482
    return bkg + A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc2(p,x,y):
    return y - gaussian_bkg(p[0],p[1],p[2],p[3],x)

def find_local_peak(flux, x, width):
    """Find the central pixel of an emission line.

    Args:
        flux (:class:`numpy.ndarray`): Flux array.
        x (int): The approximate coordinate of the peak pixel.
        width (int): Window of profile fitting.

    Returns:
        tuple: A tuple containing:

            * **i1** (*int*) -- Index of the left side.
            * **i2** (*int*) -- Index of the right side.
            * **p1** (*list*) -- List of fitting parameters.
            * **std** (*float*) -- Standard deviation of the fitting.
    """
    width = int(round(width))
    if width%2 != 1:
        width += 1
    half = int((width-1)/2)

    i = int(round(x))

    # find the peak in a narrow range

    i1, i2 = max(0, i-half), min(flux.size, i+half+1)

    if i2 - i1 <= 4:
        # 4 is the number of free parameters in fitting function
        return None

    # find the peak position
    imax = flux[i1:i2].argmax() + i1
    xdata = np.arange(i1,i2)
    ydata = flux[i1:i2]
    # determine the initial parameters for gaussian fitting + background
    p0 = [ydata.max()-ydata.min(), imax, 3., ydata.min()]
    # least square fitting
    #p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))
    p1, cov, info, mesg, ier = opt.leastsq(errfunc2, p0[:],
                                    args=(xdata,ydata), full_output=True)

    res_lst = errfunc2(p1, xdata, ydata)

    if res_lst.size-len(p0)-1 == 0:
        return None

    std = math.sqrt((res_lst**2).sum()/(res_lst.size-len(p0)-1))

    return i1, i2, p1, std

def recenter(flux, center):
    """Relocate the profile center of the lines.

    Args:
        flux (:class:`numpy.ndarray`): Flux array.
        center (float): Center of the line.

    Returns:
        float: The new center of the line profile.
    """
    y1, y2 = int(center)-3, int(center)+4
    ydata = flux[y1:y2]
    xdata = np.arange(y1,y2)
    p0 = [ydata.min(), ydata.max()-ydata.min(), ydata.argmax()+y1, 2.5]
    p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))
    return p1[2]

def search_linelist(linelistname):
    """Search the line list file and load the list.

    Args:
        linelistname (str): Name of the line list file.

    Returns:
        *string*: Path to the line list file
    """

    # first, seach $LINELIST in current working directory
    if os.path.exists(linelistname):
        return linelistname

    # seach $LINELIST.dat in current working directory
    newname = linelistname+'.dat'
    if os.path.exists(newname):
        return newname

    # seach $LINELIST in data path of edrs
    data_path = os.path.join(os.path.dirname(__file__),
                '../data/linelist/')
    newname = os.path.join(data_path, linelistname)
    if os.path.exists(newname):
        return newname

    # seach $LINELIST.dat in data path of edrs
    newname = os.path.join(data_path, linelistname+'.dat')
    if os.path.exists(newname):
        return newname

    # seach GAMSE_DATA path
    gamse_data = os.getenv('GAMSE_DATA')
    if len(gamse_data)>0:
        data_path = os.path.join(gamse_data, 'linelist')
        newname = os.path.join(data_path, linelistname+'.dat')
        if os.path.exists(newname):
            return newname

    return None

def load_linelist(filename):
    """Load standard wavelength line list from a given file.

    Args:
        filename (str): Name of the wavelength standard list file.

    Returns:
        *list*: A list containing (wavelength, species).
    """
    linelist = []
    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#%!@':
            continue
        g = row.split()
        wl = float(g[0])
        if len(g)>1:
            species = g[1]
        else:
            species = ''
        linelist.append((wl, species))
    infile.close()
    return linelist

def find_shift_ccf(f1, f2, shift0=0.0):
    """Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.ndarray`): Flux array.
        f2 (:class:`numpy.ndarray`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.

    Returns:
        float: Relative shift between the two flux arrays.
    """
    x = np.arange(f1.size)
    interf = intp.InterpolatedUnivariateSpline(x, f1, k=3)
    func = lambda shift: -(interf(x - shift)*f2).sum(dtype=np.float64)
    res = opt.minimize(func, shift0, method='Powell')
    return res['x']

def find_shift_ccf2(f1, f2, shift0=0.0):
    """Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.ndarray`): Flux array.
        f2 (:class:`numpy.ndarray`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.

    Returns:
        float: Relative shift between the two flux arrays.
    """
    n = f1.size
    def aaa(shift):
        shift = int(np.round(shift))
        s1 = f1[max(0,shift):min(n,n+shift)]
        s2 = f2[max(0,-shift):min(n,n-shift)]
        c1 = math.sqrt((s1**2).sum())
        c2 = math.sqrt((s2**2).sum())
        return -np.correlate(s1, s2)/c1/c2
    res = opt.minimize(aaa, shift0, method='Powell')
    return res['x']


def get_simple_ccf(flux1, flux2, shift_lst):
    """Get cross-correlation function of two fluxes with the given relative
    shift.

    Args:
        flux1 (:class:`numpy.ndarray`): Input flux array.
        flux2 (:class:`numpy.ndarray`): Input flux array.
        shift_lst (:class:`numpy.ndarray`): List of pixel shifts.

    Returns:
        :class:`numpy.ndarray`: Cross-correlation function
    """

    n = flux1.size
    ccf_lst = []
    for shift in shift_lst:
        segment1 = flux1[max(0,shift):min(n,n+shift)]
        segment2 = flux2[max(0,-shift):min(n,n-shift)]
        c1 = math.sqrt((segment1**2).sum())
        c2 = math.sqrt((segment2**2).sum())
        corr = np.correlate(segment1, segment2)/c1/c2
        ccf_lst.append(corr)
    return np.array(ccf_lst)


def find_pixel_drift(spec1, spec2,
        aperture_koffset=(1, 0), pixel_koffset=(1, 0.0)
    ):
    """Find the drift between two spectra. The apertures of the two spectra must
    be aligned.

    The **aperture_offset** is defined as:

        aperture1 + aperture_offset = aperture2

    Args:
        spec1 (:class:`numpy.dtype`): Spectra array.
        spec2 (:class:`numpy.dtype`): Spectra array.
        offset (float): Approximate relative shift between the two spectra
            arrays.

    Returns:
        float: Calculated relative shift between the two spectra arrays.
    """

    aperture_k, aperture_offset = aperture_koffset
    pixel_k, pixel_offset = pixel_koffset

    shift_lst = []
    for item1 in spec1:
        aperture1 = item1['aperture']
        aperture2 = aperture_k*aperture1 + aperture_offset
        m = spec2['aperture'] == aperture2
        if m.sum()==1:
            item2 = spec2[m][0]
            flux1 = item1['flux']
            flux2 = item2['flux']

            #shift = find_shift_ccf(flux1, flux2)
            #shift = find_shift_ccf_pixel(flux1, flux2, 100)
            shift = find_shift_ccf(flux1[::pixel_k], flux2, shift0=pixel_offset)

            shift_lst.append(shift)

    drift = np.median(np.array(shift_lst))
    return drift

class CalibFigure(Figure):
    """Figure class for wavelength calibration.

    Args:
        width (int): Width of figure.
        height (int): Height of figure.
        dpi (int): DPI of figure.
        filename (str): Filename of input spectra.
        channel (str): Channel name of input spectra.
    """

    def __init__(self, width, height, dpi, title):
        """Constuctor of :class:`CalibFigure`.
        """
        # set figsize and dpi
        figsize = (width/dpi, height/dpi)
        super(CalibFigure, self).__init__(figsize=figsize, dpi=dpi)

        # set background color as light gray
        self.patch.set_facecolor('#d9d9d9')

        # add axes
        self._ax1 = self.add_axes([0.07, 0.07,0.52,0.87])
        self._ax2 = self.add_axes([0.655,0.07,0.32,0.40])
        self._ax3 = self.add_axes([0.655,0.54,0.32,0.40])

        # add title
        self.suptitle(title, fontsize=15)

        #draw the aperture number to the corner of ax1
        bbox = self._ax1.get_position()
        self._ax1._aperture_text = self.text(bbox.x0 + 0.05, bbox.y1-0.1,
                                  '', fontsize=15)

        # draw residual and number of identified lines in ax2
        bbox = self._ax3.get_position()
        self._ax3._residual_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=13)

        # draw fitting parameters in ax3
        bbox = self._ax2.get_position()
        self._ax2._fitpar_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=13)

    def plot_solution(self, identlist, aperture_lst, plot_ax1=False,  **kwargs):
        """Plot the wavelength solution.

        Args:
            identlist (dict): Dict of identified lines.
            aperture_lst (list): List of apertures to be plotted.
            plot_ax1 (bool): Whether to plot the first axes.
            coeff (:class:`numpy.ndarray`): Coefficient array.
            k (int): `k` value in the relationship `order = k*aperture +
                offset`.
            offset (int): `offset` value in the relationship `order =
                k*aperture + offset`.
            npixel (int): Number of pixels along the main dispersion
                direction.
            std (float): Standard deviation of wavelength fitting.
            nuse (int): Number of lines actually used in the wavelength
                fitting.
            ntot (int): Number of lines identified.
        """
        coeff    = kwargs.pop('coeff')
        k        = kwargs.pop('k')
        offset   = kwargs.pop('offset')
        npixel   = kwargs.pop('npixel')
        std      = kwargs.pop('std')
        nuse     = kwargs.pop('nuse')
        ntot     = kwargs.pop('ntot')
        xorder   = kwargs.pop('xorder')
        yorder   = kwargs.pop('yorder')
        clipping = kwargs.pop('clipping')
        maxiter  = kwargs.pop('maxiter')

        label_size = 13  # fontsize for x, y labels
        tick_size  = 12  # fontsize for x, y ticks

        #wave_scale = 'linear'
        wave_scale = 'reciprocal'

        #colors = 'rgbcmyk'

        self._ax2.cla()
        self._ax3.cla()

        if plot_ax1:
            self._ax1.cla()
            x = np.linspace(0, npixel-1, 100, dtype=np.float64)

            # find the maximum and minimum wavelength
            wl_min, wl_max = 1e9,0
            allwave_lst = {}
            for aperture in aperture_lst:
                order = k*aperture + offset
                wave = get_wavelength(coeff, npixel, x, np.repeat(order, x.size))
                allwave_lst[aperture] = wave
                wl_max = max(wl_max, wave.max())
                wl_min = min(wl_min, wave.min())
            # plot maximum and minimum wavelength, to determine the display
            # range of this axes, and the tick positions
            self._ax1.plot([0, 0],[wl_min, wl_max], color='none')
            yticks = self._ax1.get_yticks()
            self._ax1.cla()


        for aperture in aperture_lst:
            order = k*aperture + offset
            color = 'C{}'.format(order%10)

            # plot pixel vs. wavelength
            if plot_ax1:
                wave = allwave_lst[aperture]
                if wave_scale=='reciprocal':
                    self._ax1.plot(x, 1/wave,
                            color=color, ls='-', alpha=0.8, lw=0.8)
                else:
                    self._ax1.plot(x, wave,
                            color=color, ls='-', alpha=0.8, lw=0.8)

            # plot identified lines
            if aperture in identlist:
                list1 = identlist[aperture]
                pix_lst = list1['pixel']
                wav_lst = list1['wavelength']
                mask    = list1['mask'].astype(bool)
                res_lst = list1['residual']

                if plot_ax1:
                    if wave_scale=='reciprocal':
                        self._ax1.scatter(pix_lst[mask],  1/wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], 1/wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)
                    else:
                        self._ax1.scatter(pix_lst[mask],  wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)

                repeat_aper_lst = np.repeat(aperture, pix_lst.size)
                self._ax2.scatter(repeat_aper_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax2.scatter(repeat_aper_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, ec=color)
                self._ax3.scatter(pix_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax3.scatter(pix_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, ec=color)

        # refresh texts in the residual panels
        text = 'R.M.S. = {:.5f}, N = {}/{}'.format(std, nuse, ntot)
        self._ax3._residual_text.set_text(text)
        text = u'Xorder = {}, Yorder = {}, clipping = \xb1{:g}, Niter = {}'.format(
                xorder, yorder, clipping, maxiter)
        self._ax2._fitpar_text.set_text(text)

        # adjust layout for ax1
        if plot_ax1:
            self._ax1.set_xlim(0, npixel-1)
            if wave_scale == 'reciprocal':
                _y11, _y22 = self._ax1.get_ylim()
                newtick_lst, newticklabel_lst = [], []
                for tick in yticks:
                    if _y11 < 1/tick < _y22:
                        newtick_lst.append(1/tick)
                        newticklabel_lst.append(tick)
                self._ax1.set_yticks(newtick_lst)
                self._ax1.set_yticklabels(newticklabel_lst)
                self._ax1.set_ylim(_y22, _y11)
            self._ax1.set_xlabel('Pixel', fontsize=label_size)
            self._ax1.set_ylabel(u'\u03bb (\xc5)', fontsize=label_size)
            self._ax1.grid(True, ls=':', color='gray', alpha=1, lw=0.5)
            self._ax1.set_axisbelow(True)
            self._ax1._aperture_text.set_text('')
            for tick in self._ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)
            for tick in self._ax1.yaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)

        # adjust axis layout for ax2 (residual on aperture space)
        self._ax2.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax2.axhline(y=i*std, color='k', ls=':', lw=0.5)
        x1, x2 = self._ax2.get_xlim()
        x1 = max(x1,aperture_lst.min())
        x2 = min(x2,aperture_lst.max())
        self._ax2.set_xlim(x1, x2)
        self._ax2.set_ylim(-6*std, 6*std)
        self._ax2.set_xlabel('Aperture', fontsize=label_size)
        self._ax2.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)

        ## adjust axis layout for ax3 (residual on pixel space)
        self._ax3.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax3.axhline(y=i*std, color='k', ls=':', lw=0.5)
        self._ax3.set_xlim(0, npixel-1)
        self._ax3.set_ylim(-6*std, 6*std)
        self._ax3.set_xlabel('Pixel', fontsize=label_size)
        self._ax3.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax3.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax3.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)


def select_calib_from_database(path, time_key, current_time,
        filepattern='wlcalib.\S*.fits$'):
    """Select a previous calibration result in database.

    Args:
        path (str): Path to search for the calibration files.
        time_key (str): Name of the key in the FITS header.
        current_time (str): Time string of the file to be calibrated.
        filepattern (str):

    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
                spectra.
            * **calib** (dict): Previous calibration results.
    """
    if not os.path.exists(path):
        return None, None
    filename_lst = []
    datetime_lst = []
    for fname in os.listdir(path):
        if re.match(filepattern, fname) is not None:
            filename = os.path.join(path, fname)
            head = fits.getheader(filename)
            dt = dateutil.parser.parse(head[time_key])
            filename_lst.append(filename)
            datetime_lst.append(dt)
    if len(filename_lst)==0:
        return None, None

    # select the FITS file with the shortest interval in time.
    input_datetime = dateutil.parser.parse(current_time)
    deltat_lst = np.array([abs((input_datetime - dt).total_seconds())
                            for dt in datetime_lst])
    imin = deltat_lst.argmin()
    sel_filename = filename_lst[imin]

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(sel_filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    calib = get_calib_from_header(head)

    # aperset seems unnecessary
    #aperset = load_aperture_set_from_header(head, channel=channel)
    #return spec, calib, aperset
    return spec, calib

def wlcalib(*args, **kwargs):
    recalib(*args, **kwargs)

def recalib(spec, figfilename, title, ref_spec, linelist, ref_calib,
        aperture_koffset=(1, 0), pixel_koffset=(1, None),
        xorder=None, yorder=None, maxiter=None, clipping=None, window_size=None,
        q_threshold=None, direction=None, fit_filter=None
        ):
    """Re-calibrate the wavelength of an input spectra file using another
    spectra as the reference.

    Args:
        spec (:class:`numpy.dtype`): The spectral data array to be wavelength
            calibrated.
        figfilename (str): Filename of the output wavelength figure.
        title (str): A title to display in the calib figure.
        ref_spec (:class:`numpy.dtype`): Reference spectra.
        linelist (str): Name of wavelength standard file.
        coeff (:class:`numpy.ndarray`): Coefficients of the reference wavelength.
        npixel (int): Number of pixels along the main-dispersion direction.
        k (int): -1 or 1, depending on the relationship `order = k*aperture
            + offset`.
        offset (int): coefficient in the relationship `order = k*aperture +
            offset`.
        window_size (int): Size of the window in pixel to search for the
            lines.
        xorder (int): Order of polynomial along X axis.
        yorder (int): Order of polynomial along Y axis.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        q_threshold (float): Minimum *Q*-factor of the spectral lines to be
            accepted in the wavelength fitting.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.ndarray`) -- Coefficient array.
            * **npixel** (*int*) -- Number of pixels along the main
              dispersion direction.
            * **k** (*int*) -- Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*int*) -- Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*) -- Standard deviation of wavelength fitting in Å.
            * **nuse** (*int*) -- Number of lines used in the wavelength
              fitting.
            * **ntot** (*int*) -- Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*) -- Dict of identified lines.
            * **window_size** (*int*) -- Length of window in searching the
              line centers.
            * **xorder** (*int*) -- Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*int*) -- Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*int*) -- Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*) -- Clipping value of the wavelength fitting.
            * **q_threshold** (*float*) -- Minimum *Q*-factor of the spectral
              lines to be accepted in the wavelength fitting.

    See also:
        :func:`wlcalib`
    """

    aperture_k, aperture_offset = aperture_koffset
    pixel_k, pixel_offset       = pixel_koffset

    # unpack ref_calib
    k           = ref_calib['k']
    offset      = ref_calib['offset']
    coeff       = ref_calib['coeff']
    npixel      = ref_calib['npixel']
    xorder      = (xorder, ref_calib['xorder'])[xorder is None]
    yorder      = (yorder, ref_calib['yorder'])[yorder is None]
    maxiter     = (maxiter,  ref_calib['maxiter'])[maxiter is None]
    clipping    = (clipping, ref_calib['clipping'])[clipping is None]
    window_size = (window_size, ref_calib['window_size'])[window_size is None]
    q_threshold = (q_threshold, ref_calib['q_threshold'])[q_threshold is None]

    #if pixel_offset is None:
    if False:
        # find initial shift with cross-corelation functions
        pixel_offset = find_pixel_drift(ref_spec, spec,
                        aperture_koffset = aperture_koffset,
                        pixel_koffset   = pixel_koffset)
        print('calculated shift = ', pixel_offset)

    #message = '{} channel {} shift = {:+8.6f} pixel'.format(
    #            os.path.basename(filename), channel, shift
    #            )
    #print(message)

    # initialize the identlist
    identlist = {}

    # load the wavelengths
    linefilename = search_linelist(linelist)
    if linefilename is None:
        print('Error: Cannot find linelist file: %s'%linelist)
        exit()
    line_list = load_linelist(linefilename)

    x = np.arange(npixel)[::pixel_k] + pixel_k*pixel_offset

    for row in spec:
        # variable alias
        aperture = row['aperture']
        flux     = row['flux']
        # obtain a rough wavelength array according to the input
        # aperture_koffset and pixel_koffset
        old_aperture = (aperture - aperture_offset)/aperture_k

        # now convert the old aperture number to echelle order number (m)
        order = k*old_aperture + offset
        wl = get_wavelength(coeff, npixel, x, np.repeat(order, npixel))
        w1 = min(wl[0], wl[-1])
        w2 = max(wl[0], wl[-1])

        has_insert = False
        for line in line_list:
            if line[0] < w1:
                continue
            if line[0] > w2:
                break

            # wavelength in the range of this order
            # find the nearest pixel to the calibration line
            diff = np.abs(wl - line[0])
            i = diff.argmin()

            result = find_local_peak(flux, i, window_size)
            if result is None:
                continue
            i1, i2, param, std = result

            keep = auto_line_fitting_filter(param, i1, i2)
            if not keep:
                continue

            # unpack the fitted parameters
            amplitude = param[0]
            center    = param[1]
            fwhm      = param[2]
            back      = param[3]

            # q = A/std is a proxy of signal-to-noise ratio.
            q = amplitude/std
            if q < q_threshold:
                continue

            if aperture not in identlist:
                identlist[aperture] = np.array([], dtype=identlinetype)

            # pack the line data
            item = np.array((aperture, order, center, line[0], amplitude, fwhm,
                            q, True, 0.0, 'a'), dtype=identlinetype)

            identlist[aperture] = np.append(identlist[aperture], item)
            has_insert = True

        if has_insert:
            identlist[aperture] = np.sort(identlist[aperture], order='pixel')

    new_coeff, new_std, new_k, new_offset, new_nuse, new_ntot = fit_wavelength(
        identlist = identlist,
        npixel    = npixel,
        xorder    = xorder,
        yorder    = yorder,
        maxiter   = maxiter,
        clipping  = clipping,
        fit_filter= fit_filter,
        )

    fig_width  = 2500
    fig_height = 1500
    fig_dpi    = 150

    fig = CalibFigure(width  = fig_width,
                      height = fig_height,
                      dpi    = fig_dpi,
                      title  = title,
                      )
    #canvas = FigureCanvasAgg(fig)

    fig.plot_solution(identlist,
                      aperture_lst = spec['aperture'],
                      plot_ax1     = True,
                      coeff        = new_coeff,
                      k            = new_k,
                      offset       = new_offset,
                      npixel       = npixel,
                      std          = new_std,
                      nuse         = new_nuse,
                      ntot         = new_ntot,
                      xorder       = xorder,
                      yorder       = yorder,
                      clipping     = clipping,
                      maxiter      = maxiter,
                      )
    fig.savefig(figfilename)
    plt.close(fig)

    # refresh the direction code
    new_direction = direction[0] + {1:'r', -1:'b'}[new_k] + '-+'[wl[0] < wl[-1]]
    # compare the new direction to the input direction. if not consistent,
    # print a warning
    if direction[1]!='?' and direction[1]!=new_direction[1]:
        print('Warning: Direction code 1 refreshed:',
                direction[1], new_direction[1])
    if direction[2]!='?' and direction[2]!=new_direction[2]:
        print('Warning: Direction code 2 refreshed:',
                direction[2], new_direction[2])


    # pack calibration results
    return {
            'coeff':       new_coeff,
            'npixel':      npixel,
            'k':           new_k,
            'offset':      new_offset,
            'std':         new_std,
            'nuse':        new_nuse,
            'ntot':        new_ntot,
            'identlist':   identlist,
            'window_size': window_size,
            'xorder':      xorder,
            'yorder':      yorder,
            'maxiter':     maxiter,
            'clipping':    clipping,
            'q_threshold': q_threshold,
            'direction':   new_direction,
            }

def find_caliblamp_offset(spec1, spec2, colname1='flux', colname2='flux',
        aperture_k=None, pixel_k=None, pixel_range=(-30, 30), mode='normal'):
    """Find the offset between two spectra.

    The aperture offset is defined as:

    of the same echelle order, `aperture1` in spec1 is marked as
    `k*aperture1 + offset` in spec2.

    Args:
        spec1 (:class:`numpy.dtype`): Input spectra as a numpy structrued array.
        spec2 (:class:`numpy.dtype`): Input spectra as a numpy structrued array.
        colname1 (str): Name of flux column in **spec1**.
        colname2 (str): Name of flux column in **spec2**.
        aperture_k (int): Aperture direction code (1 or -1) between **spec1**
            and **spec2**.
        pixel_k (int): Pixel direction code (1 or -1) between **spec1** and
            **spec2**.
        pixel_range (int or tuple): Pixel range of the CCF.
        mode (str):

    Returns:
        tuple: A tuple containing:

            * **offset** (*int*): Aperture offset between the two spectra.
            * **shift** (*float*): Pixel shift between the two spectra.
    """

    if isinstance(pixel_range, int) or isinstance(pixel_range, float):
        if pixel_range <=0:
            print('Error: pixel_range must be positive')
            raise ValueError
        pixel_range = int(pixel_range)
        pixel_shift_lst = np.arange(-pixel_range, pixel_range)
    elif isinstance(pixel_range, list) or isinstance(pixel_range, tuple):
        if len(pixel_range)<2:
            print('Error: pixel_range must have length of 2')
            raise ValueError
        if pixel_range[0] >= pixel_range[1]:
            print('Error: pixel_range error')
            raise ValueError
        pixel_shift_lst = np.arange(pixel_range[0], pixel_range[1])
    else:
        pass

    if mode=='debug':
        dbgpath = 'debug'
        if not os.path.exists(dbgpath):
            os.mkdir(dbgpath)
        plot_ccf     = True
        plot_scatter = True
        figname_ccf     = os.path.join(dbgpath,
                                'lamp_ccf_{:+2d}_{:+03d}.png')
        figname_scatter = os.path.join(dbgpath,
                                'lamp_ccf_scatter.png')
    else:
        plot_ccf     = False
        plot_scatter = False

    mean_lst    = {(1, 1):[], (1, -1):[], (-1, 1):[], (-1, -1):[]}
    scatter_lst = {(1, 1):[], (1, -1):[], (-1, 1):[], (-1, -1):[]}
    all_scatter_lst = []
    all_mean_lst    = []
    scatter_id_lst = []

    aper1_lst = spec1['aperture']
    aper2_lst = spec2['aperture']
    min_aper1 = aper1_lst.min()
    max_aper1 = aper1_lst.max()
    min_aper2 = aper2_lst.min()
    max_aper2 = aper2_lst.max()

    # determine the maxium absolute offsets between the orders of the two
    # spectra
    maxoff = min(max(aper1_lst.size, aper2_lst.size)//2, 10)
    aperture_offset_lst = np.arange(-maxoff, maxoff)

    def get_aper2(aper1, k, offset):
        if k == 1:
            # (aper2 - min_aper2) = (aper1 - min_aper1) + offset
            # in this case, real_offset = offset - min_aper1 + min_aper2
            aper2 = (aper1 - min_aper1) + offset + min_aper2
        elif k == -1:
            # (aper2 - min_aper2) = -(aper1 - max_aper1) + offset
            # in this cose, real_offset = offset + max_aper1 + min_aper2
            aper2 = -aper1 + max_aper1 + offset + min_aper2
        else:
            raise ValueError
        return aper2

    # aperture_k =  1: same cross-order direction;
    #              -1: reverse cross-order direction.
    if aperture_k is None:
        search_aperture_k_lst = [1, -1]
    elif aperture_k in [1, -1]:
        search_aperture_k_lst = [aperture_k]
    else:
        print('Warning: Unknown aperture_k:', aperture_k)
        raise ValueError

    # pixel_k =  1: same main-dispersion direction;
    #           -1: reverse main-dispersion direction.
    if pixel_k is None:
        search_pixel_k_lst = [1, -1]
    elif pixel_k in [1, -1]:
        search_pixel_k_lst = [pixel_k]
    else:
        print('Warning: Unknown pixel_k:', pixel_k)
        raise ValueError


    for aperture_k in search_aperture_k_lst:
        for aperture_offset in aperture_offset_lst:
            calc_pixel_shift_lst = {1: [], -1: []}
            if plot_ccf:
                fig2 = plt.figure(figsize=(10,8), dpi=150)
                axes2 = { 1: fig2.add_subplot(211),
                         -1: fig2.add_subplot(212),
                         }
            for row1 in spec1:
                aperture1 = row1['aperture']
                aperture2 = get_aper2(aperture1, aperture_k, aperture_offset)
                m = spec2['aperture'] == aperture2
                if m.sum()==0:
                    continue
                row2 = spec2[m][0]
                flux1 = row1[colname1]
                flux2 = row2[colname2]
                for pixel_k in search_pixel_k_lst:
                    '''
                    if aperture_k == -1 and pixel_k == -1:
                        fig1 = plt.figure(dpi=150)
                        ax1 = fig1.gca()
                        ax1.plot(flux1[::pixel_k], 'C0')
                        ax1.plot(flux2, 'C1')
                        ax1.set_title('Aper1 = %d, Aper2 = %d (%d, %d, %d)'%(
                            aperture1, aperture2, aperture_k, aperture_offset,
                            pixel_k))
                        fig1.savefig('check_%d_%d_%d_%02d_%02d_.png'%(
                            aperture_k, aperture_offset, pixel_k, aperture1,
                            aperture2))
                        plt.close(fig1)
                    '''

                    ccf_lst = get_simple_ccf(flux1[::pixel_k], flux2,
                                             pixel_shift_lst)
                    # find the pixel shift
                    calc_shift = pixel_shift_lst[ccf_lst.argmax()]
                    # pack the pixel shift into a list
                    calc_pixel_shift_lst[pixel_k].append(calc_shift)

                    if plot_ccf:
                        axes2[pixel_k].plot(pixel_shift_lst, ccf_lst, alpha=0.4)
                    # pixel direction loop ends here
                # order-by-order loop ends here

            # adjust the ccf figure and save
            if plot_ccf:
                for ax in axes2.values():
                    ax.set_xlim(pixel_shift_lst[0], pixel_shift_lst[-1])
                fig2.savefig(figname_ccf.format(aperture_k, aperture_offset))
                plt.close(fig2)

            # convert calc_pixel_shift_lst to numpy array
            pixel_shift_mean = {1: None, -1: None}
            pixel_shift_std  = {1: None, -1: None}
            for pixel_k in search_pixel_k_lst:
                tmp = np.array(calc_pixel_shift_lst[pixel_k])

                mean = tmp.mean()
                std  = tmp.std()

                mean_lst[(aperture_k, pixel_k)].append(mean)
                scatter_lst[(aperture_k, pixel_k)].append(std)

                # used to search the global minimum shift scatter along all the
                # (aperture_k, aperture_offset, pixel_k) space
                all_mean_lst.append(mean)
                all_scatter_lst.append(std)
                scatter_id_lst.append((aperture_k, aperture_offset, pixel_k))

    # direction loop ends here

    # plot the scatters of peaks and save it as a figure file
    if plot_scatter:
        fig3 = plt.figure(dpi=150, figsize=(8,6))
        ax3 = fig3.gca()
        for key, scatters in scatter_lst.items():
            aperture_k, pixel_k = key
            if len(scatters)==0:
                continue
            ax3.plot(aperture_offset_lst, scatters,
                        color = {1:'C0', -1:'C1'}[aperture_k],
                        ls    = {1:'-',  -1:'--'}[pixel_k],
                        label = 'Aperture k = {}, Pixel k = {}'.format(
                            aperture_k, pixel_k))
        ax3.set_xlabel('Aperture Offset')
        ax3.set_ylabel('Scatter (pixel)')
        ax3.legend(loc='lower right')
        fig3.savefig(figname_scatter)
        plt.close(fig3)

    imin = np.argmin(all_scatter_lst)
    scatter_id = scatter_id_lst[imin]
    result_aperture_k      = scatter_id[0]
    result_aperture_offset = scatter_id[1]
    result_pixel_k         = scatter_id[2]
    result_pixel_offset    = all_mean_lst[imin]

    # convert aperture_offset to real aperture_offset
    real_aperture_offset = {
             1: result_aperture_offset - min_aper1 + min_aper2,
            -1: result_aperture_offset + max_aper1 + min_aper2,
            }[result_aperture_k]
    return (result_aperture_k, real_aperture_offset,
            result_pixel_k,    result_pixel_offset)


def save_calibrated_thar(head, spec, calib, channel):
    """Save the wavelength calibrated ThAr spectra.

    Args:
        head (:class:`astropy.io.fits.Header`):
        spec (:class:`numpy.dtype`):
        calib (tuple):
        channel (str):
    """
    k      = calib['k']
    offset = calib['offset']
    xorder = calib['xorder']
    yorder = calib['yorder']
    coeff  = calib['coeff']

    if channel is None:
        leading_str = 'HIERARCH GAMSE WLCALIB'
    else:
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
    head[leading_str+' K']      = k
    head[leading_str+' OFFSET'] = offset
    head[leading_str+' XORDER'] = xorder
    head[leading_str+' YORDER'] = yorder

    # write the coefficients
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

    head[leading_str+' MAXITER']       = calib['maxiter']
    head[leading_str+' STDDEV']        = calib['std']
    head[leading_str+' WINDOW_SIZE']   = calib['window_size']
    head[leading_str+' SNR_THRESHOLD'] = calib['snr_threshold']
    head[leading_str+' CLIPPING']      = calib['clipping']
    head[leading_str+' NTOT']          = calib['ntot']
    head[leading_str+' NUSE']          = calib['nuse']
    head[leading_str+' NPIXEL']        = calib['npixel']

    file_identlist = []

    # pack the identfied line list
    for aperture, list1 in calib['identlist'].items():
        for row in list1:
            file_identlist.append(row)

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(spec)
    lst = [pri_hdu, tbl_hdu1]
    file_identlist = np.array(file_identlist, dtype=list1.dtype)
    tbl_hdu2 = fits.BinTableHDU(file_identlist)
    lst.append(tbl_hdu2)
    hdu_lst  = fits.HDUList(lst)

    return hdu_lst

def reference_wl_new(spec, calib, head, channel, include_identlist):
    k      = calib['k']
    offset = calib['offset']
    xorder = calib['xorder']
    yorder = calib['yorder']
    coeff  = calib['coeff']

    for row in spec:
       aperture = row['aperture']
       npixel   = row['points']
       order = aperture*k + offset
       wavelength = get_wavelength(coeff, npixel,
                        np.arange(npixel),
                        np.repeat(order, npixel))
       row['order']      = order
       row['wavelength'] = wavelength

    if channel is None:
        leading_str = 'HIERARCH GAMSE WLCALIB'
    else:
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
    head[leading_str+' K']      = k
    head[leading_str+' OFFSET'] = offset
    head[leading_str+' XORDER'] = xorder
    head[leading_str+' YORDER'] = yorder

    # write the coefficients
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

    head[leading_str+' NPIXEL']        = calib['npixel']
    head[leading_str+' WINDOW_SIZE']   = calib['window_size']
    head[leading_str+' MAXITER']       = calib['maxiter']
    head[leading_str+' CLIPPING']      = calib['clipping']
    head[leading_str+' SNR_THRESHOLD'] = calib['snr_threshold']
    head[leading_str+' NTOT']          = calib['ntot']
    head[leading_str+' NUSE']          = calib['nuse']
    head[leading_str+' STDDEV']        = calib['std']

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(spec)
    hdu_lst = [pri_hdu, tbl_hdu1]

    if include_identlist:
        file_identlist = []

        # pack the identfied line list
        for aperture, list1 in calib['identlist'].items():
            for row in list1:
                file_identlist.append(row)

        file_identlist = np.array(file_identlist, dtype=list1.dtype)
        tbl_hdu2 = fits.BinTableHDU(file_identlist)
        hdu_lst.append(tbl_hdu2)

    return fits.HDUList(hdu_lst)

def get_calib_weight_lst(calib_lst, obsdate, exptime):
    """Get weight according to the time interval.

    Args:
        calib_lst (list): A list of calib dicts.
        obsdate (str):
        exptime (float): Exposure time in seconds.

    Returns:
        list: A list of floats as the weights.
    """
    input_datetime = dateutil.parser.parse(obsdate) \
                        + datetime.timedelta(seconds=exptime/2)
    datetime_lst = [dateutil.parser.parse(calib['date-obs']) \
                        + datetime.timedelta(seconds=calib['exptime']/2)
                        for calib in calib_lst]

    dt_lst = [(dt - input_datetime).total_seconds() for dt in datetime_lst]
    dt_lst = np.array(dt_lst)

    if len(dt_lst)==1:
        # only one reference in datetime_lst
        weight_lst = [1.0]
    elif (dt_lst<0).sum()==0:
        # all elements in dt_lst > 0. means all references are after the input
        # datetime. then use the first reference
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        weight_lst[0] = 1.0
    elif (dt_lst>0).sum()==0:
        # all elements in dt_lst < 0. means all references are before the input
        # datetime. then use the last reference
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        weight_lst[-1] = 1.0
    else:
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        i = np.searchsorted(dt_lst, 0.0)
        w1 = -dt_lst[i-1]
        w2 = dt_lst[i]
        weight_lst[i-1] = w2/(w1+w2)
        weight_lst[i]   = w1/(w1+w2)

    return weight_lst

def combine_calib(calib_lst, weight_lst):
    """Combine a list of wavelength calibration results.

    Args:
        calib_lst (list):
        weight_lst (list):

    Return:
        dict: The combined wavelength claibration result
    """

    k      = calib_lst[0]['k']
    offset = calib_lst[0]['offset']
    xorder = calib_lst[0]['xorder']
    yorder = calib_lst[0]['yorder']
    npixel = calib_lst[0]['npixel']

    for calib in calib_lst:
        if     calib['k']      != k \
            or calib['offset'] != offset \
            or calib['xorder'] != xorder \
            or calib['yorder'] != yorder \
            or calib['npixel'] != npixel:
            print('Error: calib list is not self-consistent')
            raise ValueError


    # calculate the weighted average coefficients
    coeff = np.zeros_like(calib_lst[0]['coeff'], dtype=np.float64)
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        for calib, weight in zip(calib_lst, weight_lst):
            coeff[j, i] += calib['coeff'][j, i]*weight

    return {'k': k, 'offset': offset, 'xorder': xorder, 'yorder': yorder,
            'npixel': npixel, 'coeff': coeff}


def get_calib_from_header(header):
    """Get calib from FITS header.

    Args:
        header (:class:`astropy.io.fits.Header`): FITS header.

    Returns:
        tuple: A tuple containing calib results.
    """

    prefix = 'HIERARCH GAMSE WLCALIB '

    xorder = header[prefix+'XORDER']
    yorder = header[prefix+'YORDER']

    coeff = np.zeros((yorder+1, xorder+1))
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        coeff[j,i] = header[prefix+'COEFF {:d} {:d}'.format(j, i)]

    calib = {
              'coeff':         coeff,
              'npixel':        header[prefix+'NPIXEL'],
              'k':             header[prefix+'K'],
              'offset':        header[prefix+'OFFSET'],
              'std':           header[prefix+'STDDEV'],
              'nuse':          header[prefix+'NUSE'],
              'ntot':          header[prefix+'NTOT'],
#             'identlist':     calibwindow.identlist,
              'window_size':   header[prefix+'WINDOW_SIZE'],
              'xorder':        xorder,
              'yorder':        yorder,
              'maxiter':       header[prefix+'MAXITER'],
              'clipping':      header[prefix+'CLIPPING'],
              'q_threshold':   header[prefix+'Q_THRESHOLD'],
              'direction':     header[prefix+'DIRECTION'],
            }
    return calib


def auto_line_fitting_filter(param, i1, i2):
    """A filter function for fitting of a single calibration line.

    Args:
        param ():
        i1 (int):
        i2 (int):

    Return:
        bool:
    """
    if param[0] <= 0.:
        # line amplitdue too small
        return False
    if param[1] < i1 or param[1] > i2:
        # line center not in the fitting range (i1, i2)
        return False
    if param[2] > 50. or param[2] < 1.0:
        # line too broad or too narrow
        return False
    if param[3] < -0.5*param[0]:
        # background too low
        return False
    return True

def reference_self_wavelength(spec, calib):
    """Calculate the wavelengths for an one dimensional spectra.

    Args:
        spec (:class:`numpy.dtype`):
        calib (tuple):

    Returns:
        tuple: A tuple containing:
    """

    # calculate the wavelength for each aperture
    for row in spec:
        aperture = row['aperture']
        npoints  = len(row['wavelength'])
        order = aperture*calib['k'] + calib['offset']
        wavelength = get_wavelength(calib['coeff'], calib['npixel'],
                    np.arange(npoints), np.repeat(order, npoints))
        row['order']      = order
        row['wavelength'] = wavelength

    card_lst = []
    card_lst.append(('K',      calib['k']))
    card_lst.append(('OFFSET', calib['offset']))
    card_lst.append(('XORDER', calib['xorder']))
    card_lst.append(('YORDER', calib['yorder']))
    card_lst.append(('NPIXEL', calib['npixel']))

    # write the coefficients to fits header
    for j, i in itertools.product(range(calib['yorder']+1),
                                  range(calib['xorder']+1)):
        key   = 'COEFF {:d} {:d}'.format(j, i)
        value = calib['coeff'][j,i]
        card_lst.append((key, value))

    # write other information to fits header
    card_lst.append(('WINDOW_SIZE', calib['window_size']))
    card_lst.append(('MAXITER',     calib['maxiter']))
    card_lst.append(('CLIPPING',    calib['clipping']))
    card_lst.append(('Q_THRESHOLD', calib['q_threshold']))
    card_lst.append(('NTOT',        calib['ntot']))
    card_lst.append(('NUSE',        calib['nuse']))
    card_lst.append(('STDDEV',      calib['std']))
    card_lst.append(('DIRECTION' ,  calib['direction']))

    # pack the identfied line list
    identlist = []
    for aperture, list1 in calib['identlist'].items():
        for row in list1:
            identlist.append(row)
    identlist = np.array(identlist, dtype=list1.dtype)

    return spec, card_lst, identlist


def combine_fiber_spec(spec_lst):
    """Combine one-dimensional spectra of different fibers.

    Args:
        spec_lst (dict): A dict containing the one-dimensional spectra for all
            fibers.

    Returns:
        numpy.dtype: The combined one-dimensional spectra
    """
    spec1 = list(spec_lst.values())[0]
    newdescr = [descr for descr in spec1.dtype.descr]
    # add a new column
    newdescr.insert(0, ('fiber', 'S1'))

    newspec = []
    for fiber, spec in sorted(spec_lst.items()):
        for row in spec:
            item = list(row)
            item.insert(0, fiber)
            newspec.append(tuple(item))
    newspec = np.array(newspec, dtype=newdescr)

    return newspec

def combine_fiber_cards(card_lst):
    """Combine header cards of different fibers.

    Args:
        card_lst (dict): FITS header cards of different fibers.

    Returns:
        list: List of header cards.
    """
    newcard_lst = []
    for fiber, cards in sorted(card_lst.items()):
        for card in cards:
            key = 'FIBER {} {}'.format(fiber, card[0])
            value = card[1]
            newcard_lst.append((key, value))
    return newcard_lst

def combine_fiber_identlist(identlist_lst):
    """Combine the identified line list of different fibers.

    Args:
        identlist_lst (dict): Identified line lists of different fibers.

    Returns:
        numpy.dtype
    """
    identlist1 = list(identlist_lst.values())[0]
    newdescr = [descr for descr in identlist1.dtype.descr]
    # add a new column
    newdescr.insert(0, ('fiber', 'S1'))

    newidentlist = []
    for fiber, identlist in sorted(identlist_lst.items()):
        for row in identlist:
            item = list(row)
            item.insert(0, fiber)
            newidentlist.append(tuple(item))
    newidentlist = np.array(newidentlist, dtype=newdescr)

    return newidentlist

def reference_spec_wavelength(spec, calib_lst, weight_lst):
    """Calculate the wavelength of a spectrum with given calibration list and
    weights.

    Args:
        spec (class:`numpy.dtype`):
        calib_lst (list):
        weight_lst (list):

    Returns:
        tuple:

    See also:
        :func:`reference_pixel_wavelength`
    """
    combined_calib = combine_calib(calib_lst, weight_lst)

    k      = combined_calib['k']
    offset = combined_calib['offset']
    xorder = combined_calib['xorder']
    yorder = combined_calib['yorder']
    npixel = combined_calib['npixel']
    coeff  = combined_calib['coeff']

    # calculate the wavelength for each aperture
    for row in spec:
        aperture = row['aperture']
        npoints  = len(row['wavelength'])
        order = aperture*k + offset
        wavelength = get_wavelength(coeff, npixel,
                        np.arange(npoints), np.repeat(order, npoints))
        row['order']      = order
        row['wavelength'] = wavelength

    card_lst = []
    #prefix = 'HIERARCH GAMSE WLCALIB'
    #if fiber is not None:
    #    prefix = prefix + ' FIBER {}'.format(fiber)
    card_lst.append(('K', k))
    card_lst.append(('OFFSET', offset))
    card_lst.append(('XORDER', xorder))
    card_lst.append(('YORDER', yorder))
    card_lst.append(('NPIXEL', npixel))

    # write the coefficients to fits header
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        key   = 'COEFF {:d} {:d}'.format(j, i)
        value = coeff[j,i]
        card_lst.append((key, value))

    # write information for every reference
    for icalib, (calib, weight) in enumerate(zip(calib_lst, weight_lst)):
        prefix = 'REFERENCE {:d}'.format(icalib+1)
        card_lst.append((prefix+' FILEID',   calib['fileid']))
        card_lst.append((prefix+' DATE-OBS', calib['date-obs']))
        card_lst.append((prefix+' EXPTIME',  calib['exptime']))
        card_lst.append((prefix+' WEIGHT',   weight))
        card_lst.append((prefix+' NTOT',     calib['ntot']))
        card_lst.append((prefix+' NUSE',     calib['nuse']))
        card_lst.append((prefix+' STDDEV',   calib['std']))

    return spec, card_lst

def reference_pixel_wavelength(pixels, apertures, calib, weight=None):
    """Calculate the wavelength of a list of pixels with given calibration list
    and weights.

    Args:
        pixels (*list* or class:`numpy.ndarray`):
        apertures (*list* or class:`numpy.ndarray`):
        calib (*list* or *dict*):
        weight (*list* or class:`numpy.ndarray`):

    Returns:
        tuple:

    See also:
        :func:`reference_spec_wavelength`
    """
    pixels    = np.array(pixels)
    apertures = np.array(apertures)

    if isinstance(calib, dict):
        # calib is a single calib results
        used_calib = calib
    elif isinstance(calib, list):
        # when calib is a list of calib objects, combine them by weights
        if not hasattr(weight, '__len__'):
            print('Input weight {} has no length'.format(weight))
            raise ValueError
        elif len(calib) != len(weight):
            print('Different lengths of calib and weight ({}/{})'.format(
                len(calib), len(weight)))
            raise ValueError
        else:
            used_calib = combine_calib(calib, weight)
    else:
        print('Error: unknown datatype of calib: {}'.format(type(calib)))
        raise ValueError

    k      = used_calib['k']
    offset = used_calib['offset']
    xorder = used_calib['xorder']
    yorder = used_calib['yorder']
    npixel = used_calib['npixel']
    coeff  = used_calib['coeff']

    orders = apertures*k + offset
    wavelengths = get_wavelength(coeff, npixel, pixels, orders)
    return orders, wavelengths

def reference_wl(infilename, outfilename, regfilename, frameid, calib_lst):
    """Reference the wavelength and write the wavelength solution to the FITS
    file.

    Args:
        infilename (str): Filename of input spectra.
        outfilename (str): Filename of output spectra.
        regfilename (str): Filename of output region file for SAO-DS9.
        frameid (int): FrameID of the input spectra. The frameid is used to
            find the proper calibration solution in **calib_lst**.
        calib_lst (dict): A dict with key of frameids, and values of calibration
            solutions for different channels.

    See also:
        :func:`wlcalib`
    """
    data, head = fits.getdata(infilename, header=True)

    npoints = data['points'].max()

    newdescr = [descr for descr in data.dtype.descr]
    # add new columns
    newdescr.append(('order',np.int16))
    newdescr.append(('wavelength','>f8',(npoints,)))

    newspec = []

    # prepare for self reference. means one channel is ThAr
    file_identlist = []

    # find unique channels in the input spectra
    channel_lst = np.unique(data['channel'])

    # open region file and write headers
    regfile = open(regfilename, 'w')
    regfile.write('# Region file format: DS9 version 4.1'+os.linesep)
    regfile.write('global dashlist=8 3 width=1 font="helvetica 10 normal roman" ')
    regfile.write('select=1 highlite=1 dash=0 fixed=1 edit=0 move=0 delete=0 include=1 source=1'+os.linesep)

    # find aperture locations
    aperture_coeffs = get_aperture_coeffs_in_header(head)

    # loop all channels
    for channel in sorted(channel_lst):

        # filter the spectra in current channel
        mask = (data['channel'] == channel)
        if mask.sum() == 0:
            continue
        spec = data[mask]

        # check if the current frameid & channel are in calib_lst
        if frameid in calib_lst and channel in calib_lst[frameid]:
            self_reference = True
            calib = calib_lst[frameid][channel]
        else:
            self_reference = False
            # find the closet ThAr
            refcalib_lst = []
            if frameid <= min(calib_lst):
                calib = calib_lst[min(calib_lst)][channel]
                refcalib_lst.append(calib)
            elif frameid >= max(calib_lst):
                calib = calib_lst[max(calib_lst)][channel]
                refcalib_lst.append(calib)
            else:
                for direction in [-1, +1]:
                    _frameid = frameid
                    while(True):
                        _frameid += direction
                        if _frameid in calib_lst and channel in calib_lst[_frameid]:
                            calib = calib_lst[_frameid][channel]
                            refcalib_lst.append(calib)
                            #print(item.frameid, 'append',channel, frameid)
                            break
                        elif _frameid <= min(calib_lst) or _frameid >= max(calib_lst):
                            break
                        else:
                            continue

        # get variable shortcuts.
        # in principle, these parameters in refcalib_lst should have the same
        # values. so just use the last calib solution
        k      = calib['k']
        offset = calib['offset']
        xorder = calib['xorder']
        yorder = calib['yorder']

        if self_reference:
            coeff = calib['coeff']
        else:
            # calculate the average coefficients
            coeff_lst = np.array([_calib['coeff'] for _calib in refcalib_lst])
            coeff = coeff_lst.mean(axis=0, dtype=np.float64)

        # write important parameters into the FITS header
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
        head[leading_str+' K']      = k
        head[leading_str+' OFFSET'] = offset
        head[leading_str+' XORDER'] = xorder
        head[leading_str+' YORDER'] = yorder

        # write the coefficients
        for j, i in itertools.product(range(yorder+1), range(xorder+1)):
            head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

        # if the input spectra is a wavelength standard frame (e.g. ThAr), write
        # calibration solutions into FITS header
        if self_reference:
            head[leading_str+' MAXITER']    = calib['maxiter']
            head[leading_str+' STDDEV']     = calib['std']
            head[leading_str+' WINDOWSIZE'] = calib['window_size']
            head[leading_str+' NTOT']       = calib['ntot']
            head[leading_str+' NUSE']       = calib['nuse']
            head[leading_str+' NPIXEL']     = calib['npixel']

            # pack the identfied line list
            for aperture, list1 in calib['identlist'].items():
                for row in list1:
                    file_identlist.append(row)

        for row in spec:
            aperture = row['aperture']
            npixel   = len(row['wavelength'])
            order = aperture*k + offset
            wl = get_wavelength(coeff, npixel, np.arange(npixel), np.repeat(order, npixel))

            # add wavelength into FITS table
            item = list(row)
            item.append(order)
            item.append(wl)
            newspec.append(tuple(item))

            # write wavlength information into regfile
            if (channel, aperture) in aperture_coeffs:
                coeffs = aperture_coeffs[(channel, aperture)]
                position = poly.Chebyshev(coef=coeffs, domain=[0, npixel-1])
                color = {'A': 'red', 'B': 'green'}[channel]

                # write text in the left edge
                x = -6
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={A%d, O%d} color=%s'
                text = string%(x+1, y+1, aperture, order, color)
                regfile.write(text+os.linesep)
                print('-------'+text)

                # write text in the right edge
                x = npixel-1+6
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={A%d, O%d} color=%s'
                text = string%(x+1, y+1, aperture, order, color)
                regfile.write(text+os.linesep)

                # write text in the center
                x = npixel/2.
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={Channel %s, Aperture %3d, Order %3d} color=%s'
                text = string%(x+1, y+1+5, channel, aperture, order, color)
                regfile.write(text+os.linesep)

                # draw lines
                x = np.linspace(0, npixel-1, 50)
                y = position(x)
                for (x1,x2), (y1, y2) in zip(pairwise(x), pairwise(y)):
                    string = 'line(%7.2f,%7.2f,%7.2f,%7.2f) # color=%s'
                    text = string%(x1+1, y1+1, x2+1, y2+1, color)
                    regfile.write(text+os.linesep)

                # draw ticks at integer wavelengths
                pix = np.arange(npixel)
                if wl[0] > wl[-1]:
                    wl  = wl[::-1]
                    pix = pix[::-1]
                f = intp.InterpolatedUnivariateSpline(wl, pix, k=3)
                w1 = wl.min()
                w2 = wl.max()
                for w in np.arange(int(math.ceil(w1)), int(math.floor(w2))+1):
                    x = f(w)
                    y = position(x)
                    if w%10==0:
                        ticklen = 3
                        string = '# text(%7.2f, %7.2f) text={%4d} color=%s'
                        text = string%(x+1+20, y+1+5, w, color)
                        regfile.write(text+os.linesep)
                    else:
                        ticklen = 1
                    string = 'line(%7.2f, %7.2f, %7.2f, %7.2f) # color=%s wl=%d'
                    text = string%(x+1+20, y+1, x+1+20, y+1+ticklen, color, w)
                    regfile.write(text+os.linesep)

                # draw identified lines in region file
                if self_reference and aperture in calib['identlist']:
                    list1 = calib['identlist'][aperture]
                    for row in list1:
                        x = row['pixel']
                        y = position(x)
                        ps = ('x', 'circle')[row['mask']]
                        string = 'point(%7.2f, %7.2f) # point=%s color=%s wl=%9.4f'
                        text = string%(x+1, y+1, ps, color, row['wavelength'])
                        regfile.write(text+os.linesep)

    newspec = np.array(newspec, dtype=newdescr)

    regfile.close()

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(newspec)
    lst = [pri_hdu, tbl_hdu1]

    if len(file_identlist)>0:
        #file_identlist = np.array(file_identlist, dtype=identlinetype)
        file_identlist = np.array(file_identlist, dtype=list1.dtype)
        tbl_hdu2 = fits.BinTableHDU(file_identlist)
        lst.append(tbl_hdu2)
    hdu_lst  = fits.HDUList(lst)

    if os.path.exists(outfilename):
        os.remove(outfilename)
    hdu_lst.writeto(outfilename)

def get_aperture_coeffs_in_header(head):
    """Get coefficients of each aperture from the FITS header.

    Args:
        head (:class:`astropy.io.fits.Header`): Header of FITS file.

    Returns:
        *dict*: A dict containing coefficients for each aperture and each channel.
    """

    coeffs = {}
    for key, value in head.items():
        exp = '^GAMSE TRACE CHANNEL [A-Z] APERTURE \d+ COEFF \d+$'
        if re.match(exp, key) is not None:
            g = key.split()
            channel  = g[3]
            aperture = int(g[5])
            icoeff   = int(g[7])
            if (channel, aperture) not in coeffs:
                coeffs[(channel, aperture)] = []
            if len(coeffs[(channel, aperture)]) == icoeff:
                coeffs[(channel, aperture)].append(value)
    return coeffs


def select_calib_auto(calib_lst, rms_threshold=1e9, group_contiguous=True,
        time_diff=120):
    """Select calib as references from a list of calib objects.

    Args:
        calib_lst (dict): A dict of calib dicts.
        rms_threshold (float): Threshold of fitting RMS.
        group_contiguous (bool): Whether to group contiguous exposures.
        time_diff (float): Time difference of continuous exposures in minutes.
    Return:
        list: A list of calib dicts.
    See Also:
        :func:`select_calib_manu`
    """

    if group_contiguous:
        calib_groups = []

        # initialize previous
        prev_id = -99
        prev_time = datetime.datetime(1900, 1, 1, 0, 0, 0)

        for frameid, calib in sorted(calib_lst.items()):
            if calib['std'] > rms_threshold:
                continue

            # if this logitem is in calib_lst, it must be put into the
            # calib_groups, either in an existing group, or a new group
            this_id = frameid
            this_time = dateutil.parser.parse(calib['date-obs']) + \
                        datetime.timedelta(seconds=calib['exptime']/2)
            delta_time = (this_time - prev_time)

            if this_id == prev_id + 1 \
                and delta_time.total_seconds() < time_diff*60:
                # put it in an existing group
                calib_groups[-1].append(calib)
            else:
                # in a new group
                calib_groups.append([calib])

            prev_id = this_id
            prev_time = this_time

        # find the best calib in each group
        ref_calib_lst    = []
        for group in calib_groups:
            # find the minimum RMS within each group
            std_lst = np.array([calib['std'] for calib in group])
            imin = std_lst.argmin()
            ref_calib_lst.append(group[imin])

    else:
        # if not group continuous exposures, just pick up calibs with RMS less
        # than rms threshold
        ref_calib_lst = [calib for frameid, calib in sorted(calib_lst.items())
                            if calib['std'] < rms_threshold]

    return ref_calib_lst


def select_calib_manu(calib_lst, promotion,
        error_message='Warning: "{}" is not a valid calib object',
        ):
    """Select a calib dict manually.

    Args:
        calib_lst (dict): A dict of calib dicts.
        promotion (str): A promotion string.
        error_message (str): Message to be shown if user input is not found.
    Return:
        list: A list of calib dicts.

    See Also:
        :func:`select_calib_auto`
    """
    # print promotion and read input frameid list
    while(True):
        string = input(promotion)
        ref_calib_lst = []
        succ = True
        for s in string.split(','):
            s = s.strip()
            if len(s)>0 and s.isdigit():
                frameid = int(s)
                if frameid in calib_lst:
                    # user input is found in calib_lst
                    calib = calib_lst[frameid]
                    ref_calib_lst.append(calib)
                else:
                    # user input is not found in calib_lst
                    print(error_message.format(s))
                    succ = False
                    break

            else:
                print(error_messag.format(s))
                succ = False
                break
        if succ:
            return ref_calib_lst
        else:
            continue

class FWHMMapFigure(Figure):
    """Figure class for plotting FWHMs over the whole CCD.

    """
    def __init__(self, identlist, apertureset, fwhm_range=None):
        """Constuctor of :class:`FWHMMapFigure`.
        """
        # find shape of map
        aper0 = list(apertureset.keys())[0]
        aperloc = apertureset[aper0]
        shape = aperloc.shape

        figsize = (12, 6)
        super(FWHMMapFigure, self).__init__(figsize=figsize, dpi=150)
        _w = shape[1]/figsize[0]
        _h = shape[0]/figsize[1]
        _zoom = 0.82/_h
        _w *= _zoom
        _h *= _zoom
        self.ax1 = self.add_axes([0.09, 0.1, _w,   _h])
        self.ax2 = self.add_axes([0.58, 0.1, 0.38, 0.02])
        self.ax3 = self.add_axes([0.58, 0.6, 0.38, 0.32])
        self.ax4 = self.add_axes([0.58, 0.2, 0.38, 0.32])

        x_lst = []
        y_lst = []
        fwhm_lst = []
        aper_fwhm_lst = {}
        for row in identlist:
            if not row['mask']:
                continue
            aper   = row['aperture']
            fwhm   = row['fwhm']
            center = row['pixel']
            aperloc = apertureset[aper]
            ypos = aperloc.position(center)

            x_lst.append(center)
            y_lst.append(ypos)
            fwhm_lst.append(fwhm)

            if aper not in aper_fwhm_lst:
                aper_fwhm_lst[aper] = [[], []]
            aper_fwhm_lst[aper][0].append(center)
            aper_fwhm_lst[aper][1].append(fwhm)

        if fwhm_range is None:
            fwhm1, fwhm2 = min(fwhm_lst), max(fwhm_lst)
        else:
            fwhm1, fwhm2 = fwhm_range

        # ax1: plot FWHM as scatters
        cax = self.ax1.scatter(x_lst, y_lst, s=20, c=fwhm_lst, alpha=0.8,
                vmin=fwhm1, vmax=fwhm2, lw=0)
        self.cbar = self.colorbar(cax, cax=self.ax2, orientation='horizontal')
        self.cbar.set_label('FWHM')
        self.ax1.set_xlim(0, shape[1]-1)
        self.ax1.set_ylim(0, shape[0]-1)
        self.ax1.set_xlabel('X (pixel)')
        self.ax1.set_ylabel('Y (pixel)')

        # ax3: plot x vs. fwhm for different apertures
        for aper, (_x_lst, _fwhm_lst) in aper_fwhm_lst.items():
            self.ax3.plot(_x_lst, _fwhm_lst, 'o-',
                    c='C0', ms=2, alpha=0.6, lw=0.5)
        self.ax3.set_xlim(0, shape[1]-1)
        self.ax3.set_ylim(fwhm1, fwhm2)
        self.ax3.set_xlabel('X (pixel)')
        self.ax3.set_ylabel('FWHM')

        # ax4: plot FWHM histograms
        self.ax4.hist(fwhm_lst, bins=np.arange(fwhm1, fwhm2, 0.1))
        self.ax4.set_xlim(fwhm1, fwhm2)
        self.ax4.set_ylabel('N')

    def close(self):
        plt.close(self)
