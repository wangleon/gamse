import os
import re
import shutil
import datetime
import logging
logger = logging.getLogger(__name__)
import configparser
import dateutil.parser

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates as mdates

from ..echelle.imageproc import (combine_images, array_to_table,
                                 fix_pixels)
from ..echelle.trace import find_apertures, load_aperture_set, TraceFigureCommon
from ..echelle.flat  import get_fiber_flat, mosaic_flat_auto, mosaic_images
from ..echelle.extract import extract_aperset
from ..echelle.wlcalib import (wlcalib, recalib, get_calib_from_header,
                               get_time_weight, find_caliblamp_offset,
                               reference_wavelength,
                               reference_self_wavelength,
                               #self_reference_singlefiber,
                               #wl_reference,
                               )
from ..echelle.background import find_background
from ..utils.onedarray import get_local_minima
from ..utils.regression import iterative_polyfit
from ..utils.obslog import parse_num_seq, read_obslog
from ..utils.misc import extract_date
from .common import plot_background_aspect1, FormattedInfo
from .reduction          import Reduction

def get_badpixel_mask(shape, bins):
    """Get the mask of bad pixels and columns.

    Args:
        shape (tuple): Shape of image.
        bins (tuple): CCD bins.

    Returns:
        :class:`numpy.ndarray`: 2D binary mask, where bad pixels are marked with
            *True*, others *False*.

    The bad pixels are found *empirically*.
        
    """
    mask = np.zeros(shape, dtype=np.bool)
    if bins == (1, 1) and shape == (4136, 4096):
        h, w = shape

        mask[349:352, 627:630] = True
        mask[349:h//2, 628]    = True

        mask[1604:h//2, 2452] = True

        mask[280:284,3701]   = True
        mask[274:h//2, 3702] = True
        mask[272:h//2, 3703] = True
        mask[274:282, 3704]  = True

        mask[1720:1722, 3532:3535] = True
        mask[1720, 3535]           = True
        mask[1722, 3532]           = True
        mask[1720:h//2,3533]       = True

        mask[347:349, 4082:4084] = True
        mask[347:h//2,4083]      = True

        mask[h//2:2631, 1909] = True
    else:
        print('No bad pixel information for this CCD size.')
        raise ValueError
    return mask

def get_mask(data, head):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.

    The shape of output mask is determined by the keywords in the input FITS
    header. The numbers of columns and rows are given by::
     
        N (columns) = head['NAXIS1'] - head['COVER']

        N (rows)    = head['NAXIS2'] - head['ROVER']

    where *head* is the input FITS header. 

    """

    saturation_adu = 65535

    # determine shape of output image (also the shape of science region)
    y1 = head.get('CRVAL2', 0)
    rover = head.get('ROVER', 0)
    y2 = y1 + head['NAXIS2'] - rover
    x1 = head.get('CRVAL1', 0)
    cover = head.get('COVER', 64)
    x2 = x1 + head['NAXIS1'] - cover
    newshape = (y2-y1, x2-x1)

    # find the saturation mask
    mask_sat = (data[y1:y2, x1:x2] >= saturation_adu)
    # get bad pixel mask
    rbin = head.get('RBIN', 1)
    cbin = head.get('CBIN', 1)
    bins = (rbin, cbin)
    mask_bad = get_badpixel_mask(newshape, bins=bins)

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask


def correct_overscan(data, head, mask=None):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.
        mask (:class:`numpy.ndarray`): Input image mask.
    
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
            * **overmean** (*float) – Mean value of overscan pixels.
    """
    # define the cosmic ray fixing function
    def fix_cr(data):
        m = data.mean(dtype=np.float64)
        s = data.std(dtype=np.float64)
        _mask = data > m + 3.*s
        if _mask.sum()>0:
            x = np.arange(data.size)
            f = InterpolatedUnivariateSpline(x[~_mask], data[~_mask], k=3)
            return f(x)
        else:
            return data

    h, w = data.shape
    cover = head.get('COVER', 64)
    x1, x2 = w-cover, w

    # find the overscan level along the y-axis
    ovr_lst1 = data[0:h//2,x1+2:x2].mean(dtype=np.float64, axis=1)
    ovr_lst2 = data[h//2:h,x1+2:x2].mean(dtype=np.float64, axis=1)

    ovr_lst1_fix = fix_cr(ovr_lst1)
    ovr_lst2_fix = fix_cr(ovr_lst2)

    # apply the sav-gol fitler to the mean of overscan
    winlength = 301
    polyorder = 3
    ovrsmooth1 = savgol_filter(ovr_lst1_fix,
                    window_length=winlength, polyorder=polyorder)
    ovrsmooth2 = savgol_filter(ovr_lst2_fix,
                    window_length=winlength, polyorder=polyorder)

    # determine shape of output image (also the shape of science region)
    y1 = head.get('CRVAL2', 0)
    rover = head.get('ROVER', 0)
    y2 = y1 + head['NAXIS2'] - rover
    ymid = (y1 + y2)//2
    x1 = head.get('CRVAL1', 0)
    cover = head.get('COVER', 64)
    x2 = x1 + head['NAXIS1'] - cover
    newshape = (y2-y1, x2-x1)

    # subtract overscan
    new_data = np.zeros(newshape, dtype=np.float64)
    ovrdata1 = np.repeat([ovrsmooth1],x2-x1,axis=0).T
    ovrdata2 = np.repeat([ovrsmooth2],x2-x1,axis=0).T
    new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
    new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2
    overmean = (ovrsmooth1.mean() + ovrsmooth2.mean())/2.

    if mask is not None:
        # fix bad pixels
        bad_mask = (mask&2 > 0)
        new_data = fix_pixels(new_data, bad_mask, 'x', 'linear')

    card_lst = []
    prefix = 'HIERARCH GAMSE OVERSCAN '
    card_lst.append((prefix + 'CORRECTED', True))
    card_lst.append((prefix + 'METHOD',    'smooth:savgol'))
    card_lst.append((prefix + 'WINLEN',    winlength))
    card_lst.append((prefix + 'POLYORDER', polyorder))
    #card_lst.append((prefix+' AXIS-1',    '{}:{}'.format(x1, x2)))
    #card_lst.append((prefix+' AXIS-2',    '{}:{}'.format()))

    return new_data, card_lst, overmean


def parse_bias_frames(logtable, config, pinfo):
    """Parse the bias images and return the bias as an array.

    Args:
        logtable ():
        config ():
        pinfo ():

    Returns:
        bias:
        bias_card_lst:

    """

    rawdata = config['data']['rawdata']
    section = config['reduce.bias']
    bias_file = section['bias_file']

    bias_data_lst = []
    bias_fileid_lst = []

    for logitem in logtable:
        if logitem['object'].strip().lower()=='bias':
            filename = os.path.join(rawdata, logitem['fileid']+'.fits')
            data, head = fits.getdata(filename, header=True)
            mask = get_mask(data, head)
            data, card_lst, overmean = correct_overscan(data, head, mask)

            # print info
            if len(bias_data_lst) == 0:
                print('* Combine Bias Images: {}'.format(bias_file))
                print(' '*2 + pinfo.get_title())
                print(' '*2 + pinfo.get_separator())
            print(' '*2 + pinfo.get_format().format(logitem, overmean))

            bias_data_lst.append(data)
            bias_fileid_lst.append(logitem['fileid'])

    # get number of bias images
    n_bias = len(bias_data_lst)

    prefix = 'HIERARCH GAMSE BIAS '
    bias_card_lst = []
    bias_card_lst.append((prefix + 'NFILE', n_bias))

    if n_bias == 0:
        # there is no bias frames
        bias = None
    else:
        for i, fileid in enumerate(bias_fileid_lst):
            bias_card_lst.append((prefix + 'FILEID {:02d}'.format(i), fileid))

        # there is bias frames
        print(' '*2 + pinfo.get_separator())

        # combine bias images
        bias_data_lst = np.array(bias_data_lst)

        bias = combine_images(bias_data_lst,
                mode       = 'mean',
                upper_clip = section.getfloat('cosmic_clip'),
                maxiter    = section.getint('maxiter'),
                mask       = (None, 'max')[n_bias>=3],
                )

        ############## bias smooth ##################
        if section.getboolean('smooth'):
            # bias needs to be smoothed
            smooth_method = section.get('smooth_method')

            h, w = bias.shape
            if smooth_method in ['gauss','gaussian']:
                # perform 2D gaussian smoothing
                smooth_sigma = section.getint('smooth_sigma')
                smooth_mode  = section.get('smooth_mode')
                bias_smooth = np.zeros_like(bias, dtype=np.float64)
                bias_smooth[0:h//2, :] = gaussian_filter(bias[0:h//2, :],
                                            sigma = smooth_sigma,
                                            mode  = smooth_mode)
                bias_smooth[h//2:h, :] = gaussian_filter(bias[h//2:h, :],
                                            sigma = smooth_sigma,
                                            mode  = smooth_mode)

                # write information to FITS header
                bias_card_lst.append((prefix + 'SMOOTH CORRECTED',  True))
                bias_card_lst.append((prefix + 'SMOOTH METHOD', 'GAUSSIAN'))
                bias_card_lst.append((prefix + 'SMOOTH SIGMA',  smooth_sigma))
                bias_card_lst.append((prefix + 'SMOOTH MODE',   smooth_mode))
            else:
                print('Unknown smooth method: ', smooth_method)
                pass

            bias = bias_smooth
        else:
            # bias not smoothed
            bias_card_lst.append((prefix + 'SMOOTH CORRECTED', False))

        # create new FITS Header for bias
        head = fits.Header()
        for card in bias_card_lst:
            head.append(card)
        fits.writeto(bias_file, bias, header=head, overwrite=True)

        message = 'Bias image written to "{}"'.format(bias_file)
        logger.info(message)
        print(message)

    return bias, bias_card_lst


def select_calib_from_database(database_path, dateobs):
    """Select wavelength calibration file in database.

    Args:
        path (str): Path to search for the calibration files.
        dateobs (str): .
    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
                spectra.
            * **calib** (dict): Previous calibration results.
    """
    
    indexfile = os.path.join(database_path, 'index.dat')
    calibtable = read_obslog(indexfile)

    input_date = dateutil.parser.parse(dateobs)
    if input_date > datetime.datetime(2019, 1, 1):
        mask = calibtable['obsdate'] > datetime.datetime(2019, 1, 1)
    else:                         
        mask = calibtable['obsdate'] < datetime.datetime(2019, 1, 1)
    
    fileid = calibtable[mask][-1]['fileid']

    filename = 'wlcalib.{}.fits'.format(fileid)
    filepath = os.path.join(database_path, filename)
         
    # load spec, calib, and aperset from selected FITS file
    f = fits.open(filepath)
    head = f[0].header
    spec = f[1].data
    f.close()

    calib = get_calib_from_header(head)

    return spec, calib


def smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *A* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    
    """

    has_fringe_lst = []
    aperpar = np.array([np.nan]*w)
    xpiece_lst     = np.array([np.nan]*newx_lst.size)
    ypiece_res_lst = np.array([np.nan]*newx_lst.size)
    mask_rej_lst   = np.array([np.nan]*newx_lst.size)
    allx = np.arange(w)
    # the dtype of xpiece_lst and ypiece_lst is np.float64

    # first try, scan every segment. find fringe by checking the local maximum
    # points after smoothing. Meanwhile, save the smoothing results in case the
    # smoothing will be used afterwards.
    for group in group_lst:
        i1, i2 = group[0], group[-1]
        p1, p2 = newx_lst[i1], newx_lst[i2]
        m = fitmask[group]
        xpiece = newx_lst[group]
        ypiece = ypara[group]
        # now fill the NaN values in ypiece
        if (~m).sum() > 0:
            f = InterpolatedUnivariateSpline(xpiece[m], ypiece[m], k=3)
            ypiece = f(xpiece)
        # now xpiece and ypiece are ready

        _m = np.ones_like(ypiece, dtype=np.bool)
        for ite in range(3):
            f = InterpolatedUnivariateSpline(xpiece[_m], ypiece[_m], k=3)
            ypiece2 = f(xpiece)
            win_len = (11, 21)[ypiece2.size>23]
            ysmooth = savgol_filter(ypiece2, window_length=win_len, polyorder=3)
            res = ypiece - ysmooth
            std = res.std()
            _new_m = np.abs(res) < 3*std

            # prevent extrapolation at the boundaries
            if _new_m.size > 3:
                _new_m[0:3] = True
                _new_m[-3:] = True
            _new_m = _m*_new_m

            if _new_m.sum() == _m.sum():
                break
            _m = _new_m
        # now xpiece, ypiece, ypiece2, ysmooth, res, and _m have the same
        # lengths and meanings on their positions of elements

        f = InterpolatedUnivariateSpline(xpiece, ysmooth, k=3)
        _x = np.arange(p1, p2+1)

        aperpar[_x] = f(_x)
        xpiece_lst[group] = xpiece
        ypiece_res_lst[group] = res
        mask_rej_lst[group] = ~_m

        # find out if this order is affected by fringes, by checking the
        # distribution of local maximum points
        imax, ymax = get_local_minima(-ysmooth, window=5)
        if len(imax) > 0:
            x = xpiece[imax]
        else:
            x = []
        # determine how many pixels in each bin.
        # if w=4000, then 500 pix. if w=2000, then 250 pix.
        npixbin = w//8
        bins = np.linspace(p1, p2, int(p2-p1)/npixbin+2)
        hist, _ = np.histogram(x, bins)

        n_nonzerobins = np.nonzero(hist)[0].size
        n_zerobins = hist.size - n_nonzerobins

        if p2-p1<w/8 or n_zerobins<=1 or \
            n_zerobins<n_nonzerobins or n_nonzerobins>=3:
            # there's fringe
            has_fringe = True
        else:
            # no fringe
            has_fringe = False
        has_fringe_lst.append(has_fringe)

    # use global polynomial fitting if this order is affected by fringe and the
    # following conditions are satisified
    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2 and \
        has_fringe_lst.count(True) == len(has_fringe_lst):
        # fit polynomial over the whole order

        # prepare xpiece and y piece
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # fit with poly
        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
            upper_clip=3)
        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst = ~_m
    else:
        # scan again
        # fit polynomial for every segment
        for group, has_fringe in zip(group_lst, has_fringe_lst):
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            if has_fringe:
                deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]
            else:
                deg = 7
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, np.log(ypiece), deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)
            ypiece_fit = np.exp(ypiece_fit)
            ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.exp(np.polyval(coeff, ii/w))
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *k* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    """

    allx = np.arange(w)

    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*w)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/w)
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *c* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_bkg`
    """
    return smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, w)

def smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *bkg* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
    """

    allx = np.arange(w)

    # fit for bkg
    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*w)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 7)[xspan>w/2]

            scale = ('linear','log')[(ypiece<=0).sum()==0]
            if scale=='log':
                ypiece = np.log(ypiece)

            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            if scale=='log':
                ypiece = np.exp(ypiece)
                ypiece_fit = np.exp(ypiece_fit)
                ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/w)
            if scale=='log':
                aperpar[ii] = np.exp(aperpar[ii])
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.43,0.86])
        self.ax2 = self.add_axes([0.52,0.50,0.43,0.40])
        self.ax3 = self.add_axes([0.52,0.10,0.43,0.40])
        self.ax4 = self.ax3.twinx()

def reduce():
    """2D to 1D pipeline for the High Resolution spectrograph on Xinglong 2.16m
    telescope.
    """

    # find obs log
    logname_lst = [fname for fname in os.listdir(os.curdir)
                        if fname[-7:]=='.obslog']
    if len(logname_lst)==0:
        print('No observation log found')
        exit()
    elif len(logname_lst)>1:
        print('Multiple observation log found:')
        for logname in sorted(logname_lst):
            print('  '+logname)
    else:
        pass

    # read obs log
    logtable = read_obslog(logname_lst[0])

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )

    # find local config file
    for fname in os.listdir(os.curdir):
        if fname[0:14]=='Xinglong216HRS' and fname[-4:]=='.cfg':
            config.read(fname)
            print('Load Congfile File: {}'.format(fname))
            break

    section = config['data']
    fibermode = section.get('fibermode')

    if fibermode == 'single':
        reduce_singlefiber(logtable, config)
    elif fibermode == 'double':
        reduce_multifiber(logtable, config)

def reduce_singlefiber(logtable, config):
    """Reduce the single fiber data of Xinglong 2.16m HRS.

    Args:
        logtable ():
        config ():

    """

    # extract keywords from config file
    section     = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')

    section     = config['reduce']
    midproc     = section.get('midproc')
    onedspec    = section.get('onedspec')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):   os.mkdir(report)
    if not os.path.exists(onedspec): os.mkdir(onedspec)
    if not os.path.exists(midproc):  os.mkdir(midproc)

    # initialize printing infomation
    pinfo1 = FormattedInfo(all_columns, ['frameid', 'fileid', 'imgtype',
                'object', 'exptime', 'obsdate', 'nsat', 'q95'])
    pinfo2 = pinfo1.add_columns([('overscan', 'float', '{:^8s}', '{1:8.2f}')])

    ############################# parse bias ###################################

    bias_file = config['reduce.bias']['bias_file']
    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        bias, head = fits.getdata(bias_file, header=True)
        message = 'Load bias from image: {}'.format(bias_file)
        logger.info(message)
        print(message)
        bias_card_lst = []
    else:
        bias, bias_card_lst = parse_bias_frames(logtable, config, pinfo2)

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for single fiber
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}

    for logitem in logtable:
        objname = logitem['object'].lower().strip()
        # above only valid for single fiber

        mobj = re.match('^flat[\s\S]*', objname)
        if mobj is not None:
            # the object name of the channel matches "flat ???"

            # find a proper name for this flat
            if objname=='flat':
                # no special names given, use "flat_A_15"
                flatname = '{:g}'.format(logitem['exptime'])
            else:
                # flatname is given. replace space with "_"
                # remove "flat" before the objectname. e.g.,
                # "Flat Red" becomes "Red" 
                char = objname[4:].strip()
                flatname = char.replace(' ','_')

            # add flatname to flat_groups
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem)

    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {}
    flat_norm_lst = {}
    flat_mask_lst = {}
    aperset_lst   = {}
    flat_info_lst = {}

    # first combine the flats
    for flatname, item_lst in flat_groups.items():
        nflat = len(item_lst)       # number of flat fieldings

        # single-fiber
        flat_filename    = os.path.join(midproc,
                            'flat_{}.fits'.format(flatname))
        aperset_filename = os.path.join(midproc,
                            'trace_flat_{}.trc'.format(flatname))
        aperset_regname  = os.path.join(midproc,
                            'trace_flat_{}.reg'.format(flatname))
        trace_figname = os.path.join(report,
                        'trace_flat_{}.{}'.format(flatname, fig_format))

        # get flat_data and mask_array
        if mode=='debug' and os.path.exists(flat_filename) \
            and os.path.exists(aperset_filename):
            # read flat data and mask array
            hdu_lst = fits.open(flat_filename)
            flat_data  = hdu_lst[0].data
            exptime    = hdu_lst[0].header[exptime_key]
            mask_array = hdu_lst[1].data
            hdu_lst.close()
            aperset = load_aperture_set(aperset_filename)
        else:
            # if the above conditions are not satisfied, comine each flat
            data_lst = []
            head_lst = []
            exptime_lst = []

            print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
            print(' '*2 + pinfo2.get_separator())
            print(' '*2 + pinfo2.get_title())
            print(' '*2 + pinfo2.get_separator())

            for i_item, item in enumerate(item_lst):
                # read each individual flat frame
                filename = os.path.join(rawdata, item['fileid']+'.fits')
                data, head = fits.getdata(filename, header=True)
                exptime_lst.append(head[exptime_key])
                mask = get_mask(data, head)
                sat_mask = (mask&4>0)
                bad_mask = (mask&2>0)
                if i_item == 0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += sat_mask

                # correct overscan for flat
                data, card_lst, overmean = correct_overscan(data, head, mask)
                for key, value in card_lst:
                    head.append((key, value))

                # correct bias for flat, if has bias
                if bias is not None:
                    data = data - bias
                    message = 'Bias corrected'
                else:
                    message = 'No bias. skipped bias correction'
                logger.info(message)
                print(message)

                # print info
                string = pinfo2.get_format().format(item, overmean)
                print(' '*2 + print_wrapper(string, logitem))

                data_lst.append(data)

            print(' '*2 + pinfo2.get_separator())

            if nflat == 1:
                flat_data = data_lst[0]
            else:
                data_lst = np.array(data_lst)
                flat_data = combine_images(data_lst,
                                mode       = 'mean',
                                upper_clip = 10,
                                maxiter    = 5,
                                mask       = (None, 'max')[nflat>3],
                                )

            # get mean exposure time and write it to header
            head = fits.Header()
            exptime = np.array(exptime_lst).mean()
            head[exptime_key] = exptime

            # find saturation mask
            sat_mask = allmask > nflat/2.
            mask_array = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

            # pack results and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data, head),
                        fits.ImageHDU(mask_array),
                        ])
            hdu_lst.writeto(flat_filename, overwrite=True)

            # now flt_data and mask_array are prepared

            # create the trace figure
            tracefig = TraceFigure()

            section = config['reduce.trace']
            aperset = find_apertures(flat_data, mask_array,
                        scan_step  = section.getint('scan_step'),
                        minimum    = section.getfloat('minimum'),
                        separation = section.get('separation'),
                        align_deg  = section.getint('align_deg'),
                        filling    = section.getfloat('filling'),
                        degree     = section.getint('degree'),
                        display    = section.getboolean('display'),
                        fig        = tracefig,
                        )

            # save the trace figure
            tracefig.adjust_positions()
            tracefig.suptitle('Trace for {}'.format(flat_filename), fontsize=15)
            tracefig.savefig(trace_figname)

            aperset.save_txt(aperset_filename)
            aperset.save_reg(aperset_regname)

        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_norm_lst[flatname] = flat_data/exptime
        flat_mask_lst[flatname] = mask_array
        aperset_lst[flatname]   = aperset
        flat_info_lst[flatname] = {'exptime': exptime}

    ########################### Get flat fielding ##############################
    flatmap_lst = {}

    for flatname in sorted(flat_groups.keys()):

        flat_filename = os.path.join(midproc,
                    'flat_{}.fits'.format(flatname))

        hdu_lst = fits.open(flat_filename, mode='update')
        if len(hdu_lst)>=3:
            # sensitivity map already exists in fits file
            flatmap = hdu_lst[2].data
            hdu_lst.close()
        else:
            # do flat fielding
            print('*** Start parsing flat fielding: %s ***'%flatname)
            fig_aperpar = {
                'debug': os.path.join(report,
                    'flat_aperpar_{}_%03d.{}'.format(flatname, fig_format)),
                'normal': None,
                }[mode]

            fig_slit = os.path.join(report,
                            'slit_{}.{}'.format(flatname, fig_format))

            section = config['reduce.flat']

            flatmap = get_fiber_flat(
                        data            = flat_data_lst[flatname],
                        mask            = flat_mask_lst[flatname],
                        apertureset     = aperset_lst[flatname],
                        slit_step       = section.getint('slit_step'),
                        nflat           = len(flat_groups[flatname]),
                        q_threshold     = section.getfloat('q_threshold'),
                        smooth_A_func   = smooth_aperpar_A,
                        smooth_k_func   = smooth_aperpar_k,
                        smooth_c_func   = smooth_aperpar_c,
                        smooth_bkg_func = smooth_aperpar_bkg,
                        fig_aperpar     = fig_aperpar,
                        fig_overlap     = None,
                        fig_slit        = fig_slit,
                        slit_file       = None,
                        )
        
            # append the sensitity map to fits file
            hdu_lst.append(fits.ImageHDU(flatmap))
            # write back to the original file
            hdu_lst.flush()

        # append the flatmap
        flatmap_lst[flatname] = flatmap

        # continue to the next colored flat

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    if len(flat_groups) == 1:
        # there's only ONE "color" of flat
        flatname = list(flat_groups)[0]

        # copy the flat fits
        oriname = 'flat_{}.fits'.format(flatname)
        shutil.copyfile(os.path.join(midproc, oriname), flat_file)

        '''
        shutil.copyfile(os.path.join(midproc, 'trace_{}.trc'.format(flatname)),
                        trac_file)
        shutil.copyfile(os.path.join(midproc, 'trace_{}.reg'.format(flatname)),
                        treg_file)
        '''

        flat_map = flatmap_lst[flatname]

        # no need to aperset mosaic
        master_aperset = list(aperset_lst.values())[0]
    else:
        # mosaic apertures
        section = config['reduce.flat']
        # determine the mosaic order
        name_lst = sorted(flat_info_lst,
                    key = lambda x: flat_info_lst.get(x)['exptime'])

        master_aperset = mosaic_flat_auto(
                aperture_set_lst = aperset_lst,
                max_count        = section.getfloat('mosaic_maxcount'),
                name_lst         = name_lst,
                )
        # mosaic original flat images
        flat_data = mosaic_images(flat_data_lst, master_aperset)
        # mosaic flat mask images
        mask_data = mosaic_images(flat_mask_lst, master_aperset)
        # mosaic sensitivity map
        flat_map = mosaic_images(flatmap_lst, master_aperset)
        # mosaic exptime-normalized flat images
        flat_norm = mosaic_images(flat_norm_lst, master_aperset)

        # pack and save to fits file
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(flat_data),
                    fits.ImageHDU(mask_data),
                    fits.ImageHDU(flat_map),
                    fits.ImageHDU(flat_norm),
                    ])
        hdu_lst.writeto(flat_file, overwrite=True)

        master_aperset.save_txt(trac_file)
        master_aperset.save_reg(treg_file)

    ############################## Extract ThAr ################################

    # get the data shape
    h, w = flat_map.shape

    # define dtype of 1-d spectra
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux',       (np.float32, w)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})
    
    calib_lst = {}
    count_thar = 0
    for logitem in logtable:

        if logitem['imgtype'] != 'cal':
            continue

        if logitem['object'].strip().lower() != 'thar':
            continue

        count_thar += 1
        frameid = logitem['frameid']
        fileid  = logitem['fileid']

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst, overmean = correct_overscan(data, head, mask)
        for key, value in card_lst:
            head.append((key, value))

        # correct bias for ThAr, if has bias
        if bias is None:
            data = data - bias
            logger.info('Bias corrected')
        else:
            logger.info('No bias. skipped bias correction')

        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )
        head = master_aperset.to_fitsheader(head)
    
        # pack to a structured array
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            spec.append((aper, 0, flux_sum.size,
                    np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
        spec = np.array(spec, dtype=spectype)
    
        wlcalib_fig = os.path.join(report,
                'wlcalib_{}.{}'.format(fileid, fig_format))

        section = config['reduce.wlcalib']

        title = fileid+'.fits'

        if count_thar == 1:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                database_path = section.get('database_path')

                result = select_calib_from_database(
                            database_path, head[statime_key])
                ref_spec, ref_calib = result
    
                if ref_spec is None or ref_calib is None:
                    # if failed, pop up a calibration window and identify
                    # the wavelengths manually
                    calib = wlcalib(spec,
                        figfilename = wlcalib_fig,
                        title       = title,
                        linelist    = section.get('linelist'),
                        window_size = section.getint('window_size'),
                        xorder      = section.getint('xorder'),
                        yorder      = section.getint('yorder'),
                        maxiter     = section.getint('maxiter'),
                        clipping    = section.getfloat('clipping'),
                        q_threshold = section.getfloat('q_threshold'),
                        )
                else:
                    # if success, run recalib
                    # determien the direction
                    ref_direction = ref_calib['direction']
                    aperture_k = ((-1, 1)[direction[1]==ref_direction[1]],
                                    None)[direction[1]=='?']
                    pixel_k = ((-1, 1)[direction[2]==ref_direction[2]],
                                None)[direction[2]=='?']
                    # determine the name of the output figure during lamp shift
                    # finding.
                    fig_ccf = {'normal': None,
                                'debug': os.path.join(report,
                                        'lamp_ccf_{:+2d}_{:+03d}.png')}[mode]
                    fig_scatter = {'normal': None,
                                    'debug': os.path.join(report,
                                        'lamp_ccf_scatter.png')}[mode]

                    result = find_caliblamp_offset(ref_spec, spec,
                                aperture_k  = aperture_k,
                                pixel_k     = pixel_k,
                                fig_ccf     = fig_ccf,
                                fig_scatter = fig_scatter,
                                )
                    aperture_koffset = (result[0], result[1])
                    pixel_koffset    = (result[2], result[3])

                    print('Aperture offset =', aperture_koffset)
                    print('Pixel offset =', pixel_koffset)

                    use = section.getboolean('use_prev_fitpar')
                    xorder      = (section.getint('xorder'), None)[use]
                    yorder      = (section.getint('yorder'), None)[use]
                    maxiter     = (section.getint('maxiter'), None)[use]
                    clipping    = (section.getfloat('clipping'), None)[use]
                    window_size = (section.getint('window_size'), None)[use]
                    q_threshold = (section.getfloat('q_threshold'), None)[use]

                    calib = recalib(spec,
                        figfilename      = wlcalib_fig,
                        title            = title,
                        ref_spec         = ref_spec,
                        linelist         = section.get('linelist'),
                        aperture_koffset = aperture_koffset,
                        pixel_koffset    = pixel_koffset,
                        ref_calib        = ref_calib,
                        xorder           = xorder,
                        yorder           = yorder,
                        maxiter          = maxiter,
                        clipping         = clipping,
                        window_size      = window_size,
                        q_threshold      = q_threshold,
                        direction        = direction,
                        )
            else:
                # do not search the database
                calib = wlcalib(spec,
                    figfilename   = wlcalib_fig,
                    title         = title,
                    identfilename = section.get('ident_file', None),
                    linelist      = section.get('linelist'),
                    window_size   = section.getint('window_size'),
                    xorder        = section.getint('xorder'),
                    yorder        = section.getint('yorder'),
                    maxiter       = section.getint('maxiter'),
                    clipping      = section.getfloat('clipping'),
                    q_threshold   = section.getfloat('q_threshold'),
                    )

            # then use this thar as reference
            ref_calib = calib
            ref_spec  = spec
        else:
            # for other ThArs, no aperture offset
            calib = recalib(spec,
                figfilename      = wlcalib_fig,
                title            = title,
                ref_spec         = ref_spec,
                linelist         = section.get('linelist'),
                ref_calib        = ref_calib,
                aperture_koffset = (1, 0),
                pixel_koffset    = (1, 0),
                xorder           = ref_calib['xorder'],
                yorder           = ref_calib['yorder'],
                maxiter          = ref_calib['maxiter'],
                clipping         = ref_calib['clipping'],
                window_size      = ref_calib['window_size'],
                q_threshold      = ref_calib['q_threshold'],
                direction        = direction,
                )
                
        #hdu_lst = self_reference_singlefiber(spec, head, calib)
        filename = os.path.join(onedspec,
                                '{}_{}.fits'.format(fileid, oned_suffix))
        hdu_lst.writeto(filename, overwrite=True)
    
        # add more infos in calib
        calib['fileid']   = fileid
        calib['date-obs'] = head[statime_key]
        calib['exptime']  = head[exptime_key]
        # pack to calib_lst
        calib_lst[logitem['frameid']] = calib

        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(spec, calib)

        # save calib results and the oned spec for this fiber
        for key,value in card_lst:
            key = 'HIERARCH GAMSE WLCALIB '+key
            head.append((key, value))

        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    fits.BinTableHDU(identlist),
                    ])
        filename = os.path.join(midproc, 'wlcalib.{}.fits'.format(fileid))
        hdu_lst.writeto(filename, overwrite=True)

        filename = os.path.join(onedspec,
                                '{}_{}.fits'.format(fileid, oned_suffix))
        hdu_lst.writeto(filename, overwrite=True)

        # pack to calib_lst
        if frameid not in calib_lst:
            calib[frameid] = calib

    
    # print fitting summary
    fmt_string = (' [{:3d}] {}'
                    ' - ({:4g} sec)'
                    ' - {:4d}/{:4d} r.m.s. = {:7.5f}')
    for frameid, calib in sorted(calib_lst.items()):
        print(fmt_string.format(frameid, calib['fileid'], calib['exptime'],
            calib['nuse'], calib['ntot'], calib['std']))
    
    # print promotion and read input frameid list
    while(True):
        string = input('Select References: ')
        ref_frameid_lst  = []
        ref_calib_lst    = []
        ref_datetime_lst = []
        succ = True
        for s in string.split(','):
            s = s.strip()
            if len(s)>0 and s.isdigit() and int(s) in calib_lst:
                frameid = int(s)
                calib   = calib_lst[frameid]
                ref_frameid_lst.append(frameid)
                ref_calib_lst.append(calib)
                ref_datetime_lst.append(calib['date-obs'])
            else:
                print('Warning: "{}" is an invalid calib frame'.format(s))
                succ = False
                break
        if succ:
            break
        else:
            continue

    '''
    ############################## Extract Flat ################################
    flat_norm = flat_norm/flat_map
    section = config['reduce.extract']
    spectra1d = extract_aperset(flat_norm, mask_data,
                apertureset = master_aperset,
                lower_limit = section.getfloat('lower_limit'),
                upper_limit = section.getfloat('upper_limit'),
                )
    # pack spectrum
    spec = []
    for aper, item in sorted(spectra1d.items()):
        flux_sum = item['flux_sum']
        spec.append((aper, 0, flux_sum.size,
            np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
    spec = np.array(spec, dtype=spectype)
    
    # wavelength calibration
    weight_lst = get_time_weight(ref_datetime_lst, head[statime_key])
    head = fits.Header()
    spec, head = wl_reference_singlefiber(spec, head, ref_calib_lst, weight_lst)

    # pack and save wavelength referenced spectra
    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(header=head),
                fits.BinTableHDU(spec),
                ])
    filename = os.path.join(onedspec, 'flat'+oned_suffix+'.fits')
    hdu_lst.writeto(filename, overwrite=True)
    '''

    #################### Extract Science Spectrum ##############################
    for logitem in logtable:

        # logitem alias
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objname = logitem['object'].strip().lower()

        #if (imgtype == 'cal' and objname == 'i2') or imgtype == 'sci':
        if imgtype != 'sci' and objname != 'i2':
            continue

        filename = os.path.join(rawdata, fileid+'.fits')

        logger.info('FileID: {} ({}) - start reduction: {}'.format(
            fileid, imgtype, filename))

        # read raw data
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data, card_lst, overmean = correct_overscan(data, head, mask)
        for key, value in card_lst:
            head.append((key, value))
        message = 'FileID: {} - overscan corrected'.format(fileid)
        logger.info(message)
        print(message)

        # correct bias
        if bias is not None:
            data = data - bias
            fmt_str = 'FileID: {} - bias corrected. mean value = {}'
            message = fmt_str.format(fileid, bias.mean())
        else:
            message = 'FileID: {} - no bias'%(fileid)
        logger.info(message)
        print(message)

        # correct flat
        data = data/flat_map
        message = 'FileID: {} - flat corrected'.format(fileid)
        logger.info(message)
        print(message)

        # correct background
        section = config['reduce.background']
        fig_sec = os.path.join(report,
                    'bkg_{}_sec.{}'.format(fileid, fig_format))

        stray = find_background(data, mask,
                apertureset_lst = master_aperset,
                ncols           = section.getint('ncols'),
                distance        = section.getfloat('distance'),
                yorder          = section.getint('yorder'),
                fig_section     = fig_sec,
                )
        data = data - stray

        ####
        #outfilename = os.path.join(midproc, '%s_bkg.fits'%fileid)
        #fits.writeto(outfilename, data)

        # plot stray light
        fig_stray = os.path.join(report,
                    'bkg_{}_stray.{}'.format(fileid, fig_format))
        plot_background_aspect1(data+stray, stray, fig_stray)

        logger.info('FileID: {} - background corrected'.format(fileid))

        # extract 1d spectrum
        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )
        logger.info('FileID: {} - 1D spectra of {} orders are extracted'.format(
            fileid, len(spectra1d)))

        # pack spectrum
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            spec.append((aper, 0, flux_sum.size,
                    np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
        spec = np.array(spec, dtype=spectype)

        # wavelength calibration
        weight_lst = get_time_weight(ref_datetime_lst, head[statime_key])

        logger.info('FileID: {} - wavelength calibration weights: {}'.format(
            fileid, ','.join(['%8.4f'%w for w in weight_lst])))

        spec, head = wl_reference_singlefiber(spec, head,
                        ref_calib_lst, weight_lst)

        # pack and save wavelength referenced spectra
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    ])
        filename = os.path.join(onedspec,
                                '{}_{}.fits'.format(fileid, oned_suffix))
        hdu_lst.writeto(filename, overwrite=True)
        logger.info('FileID: {} - Spectra written to {}'.format(
            fileid, filename))

def reduce_multifiber(logtable, config):
    """Reduce the multi-fiber data of Xinglong 2.16m HRS.

    Args:
        logtable ():
        config ():

    """

    # extract keywords from config file
    section     = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')
    # if mulit-fiber, get fiber offset list from config file
    fiber_offsets = [float(v) for v in section.get('fiberoffset').split(',')]

    section     = config['reduce']
    midproc     = section.get('midproc')
    onedspec    = section.get('onedspec')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):   os.mkdir(report)
    if not os.path.exists(onedspec): os.mkdir(onedspec)
    if not os.path.exists(midproc):  os.mkdir(midproc)

    # initialize printing infomation
    pinfo1 = FormattedInfo(all_columns, ['frameid', 'fileid', 'imgtype',
                'object', 'exptime', 'obsdate', 'nsat', 'q95'])
    pinfo2 = pinfo1.add_columns([('overscan', 'float', '{:^8s}', '{1:8.2f}')])

    # count the number of fibers
    n_fiber = 1
    for logitem in logtable:
        n = len(logitem['object'].split(';'))
        n_fiber = max(n_fiber, n)
    message = ', '.join(['multi_fiber = True',
                         'number of fiber = {}'.format(n_fiber)])
    print(message)

    ############################# parse bias ###################################

    bias_file = config['reduce.bias']['bias_file']
    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        bias, head = fits.getdata(bias_file, header=True)
        message = 'Load bias from image: {}'.format(bias_file)
        logger.info(message)
        print(message)
        bias_card_lst = []
    else:
        bias, bias_card_lst = parse_bias_frames(logtable, config, pinfo2)

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for multi-fibers
    flat_groups = {chr(ifiber+65): {} for ifiber in range(n_fiber)}
    # flat_groups = {'A':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}
    #                'B':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}}

    for logitem in logtable:
        fiberobj_lst = [v.strip() for v in logitem['object'].split(';')]

        if n_fiber > len(fiberobj_lst):
            continue

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            objname = fiberobj_lst[ifiber].lower().strip()
            mobj = re.match('^flat[\s\S]*', objname)
            if mobj is not None:
                # the object name of the channel matches "flat ???"
            
                # check the lengthes of names for other channels
                # if this list has no elements (only one fiber) or has no
                # names, this frame is a single-channel flat
                other_lst = [name for i, name in enumerate(fiberobj_lst)
                                    if i != ifiber and len(name)>0]
                if len(other_lst)>0:
                    # this frame is not a single chanel flat. Skip
                    continue

                # find a proper name (flatname) for this flat
                if objname=='flat':
                    # no special names given, use exptime
                    flatname = '{:g}'.format(logitem['exptime'])
                else:
                    # flatname is given. replace space with "_"
                    # remove "flat" before the objectname. e.g.,
                    # "Flat Red" becomes "Red" 
                    char = objname[4:].strip()
                    flatname = char.replace(' ','_')
            
                # add flatname to flat_groups
                if flatname not in flat_groups[fiber]:
                    flat_groups[fiber][flatname] = []
                flat_groups[fiber][flatname].append(logitem)

    '''
    # print the flat_groups
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        print(fiber)
        for flatname, item_lst in flat_groups[fiber].items():
            print(flatname)
            for item in item_lst:
                print(fiber, flatname, item['fileid'], item['exptime'])
    '''
    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_norm_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_mask_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    aperset_lst   = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_info_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}

    # first combine the flats
    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        for flatname, item_lst in sorted(fiber_flat_lst.items()):
            nflat = len(item_lst)       # number of flat fieldings

            flat_filename = os.path.join(midproc,
                    'flat_{}_{}.fits'.format(fiber, flatname))
            aperset_filename = os.path.join(midproc,
                    'trace_flat_{}_{}.trc'.format(fiber, flatname))
            aperset_regname = os.path.join(midproc,
                    'trace_flat_{}_{}.reg'.format(fiber, flatname))
            trace_figname = os.path.join(report,
                    'trace_flat_{}_{}.{}'.format(fiber, flatname, fig_format))

            # get flat_data and mask_array for each flat group
            if mode=='debug' and os.path.exists(flat_filename) \
                and os.path.exists(aperset_filename):
                # read flat data and mask array
                hdu_lst = fits.open(flat_filename)
                flat_data  = hdu_lst[0].data
                exptime    = hdu_lst[0].header[exptime_key]
                mask_array = hdu_lst[1].data
                hdu_lst.close()
                aperset = load_aperture_set(aperset_filename)
            else:
                # if the above conditions are not satisfied, comine each flat
                data_lst = []
                head_lst = []
                exptime_lst = []

                print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
                print(' '*2 + pinfo2.get_separator())
                print(' '*2 + pinfo2.get_title())
                print(' '*2 + pinfo2.get_separator())

                for i_item, logitem in enumerate(item_lst):
                    # read each individual flat frame
                    filename = os.path.join(rawdata, logitem['fileid']+'.fits')
                    data, head = fits.getdata(filename, header=True)
                    exptime_lst.append(head[exptime_key])
                    mask = get_mask(data, head)

                    # generate the mask for all images
                    sat_mask = (mask&4>0)
                    bad_mask = (mask&2>0)
                    if i_item == 0:
                        allmask = np.zeros_like(mask, dtype=np.int16)
                    allmask += sat_mask

                    # correct overscan for flat
                    data, card_lst, overmean = correct_overscan(data, head, mask)
                    for key, valaue in card_lst:
                        head.append((key, value))

                    # correct bias for flat, if has bias
                    if bias is not None:
                        data = data - bias
                        message = 'Bias corrected'
                    else:
                        message = 'No bias. skipped bias correction'
                    logger.info(message)
                    print(message)

                    # print info
                    string = pinfo2.get_format().format(logitem, overmean)
                    print(' '*2 + print_wrapper(string, logitem))

                    data_lst.append(data)

                print(' '*2 + pinfo2.get_separator())

                if nflat == 1:
                    flat_data = data_lst[0]
                else:
                    data_lst = np.array(data_lst)
                    flat_data = combine_images(data_lst,
                                    mode       = 'mean',
                                    upper_clip = 10,
                                    maxiter    = 5,
                                    mask       = (None, 'max')[nflat>3],
                                    )

                # get mean exposure time and write it to header
                head = fits.Header()
                exptime = np.array(exptime_lst).mean()
                head[exptime_key] = exptime

                # find saturation mask
                sat_mask = allmask > nflat/2.
                mask_array = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

                # pack results and save to fits
                hdu_lst = fits.HDUList([
                            fits.PrimaryHDU(flat_data, head),
                            fits.ImageHDU(mask_array),
                            ])
                hdu_lst.writeto(flat_filename, overwrite=True)

                # now flt_data and mask_array are prepared

                # create the trace figure
                tracefig = TraceFigure()

                # if debackground before detecting the orders, then we lose the 
                # ability to detect the weak blue orders.
                #xnodes = np.arange(0, flat_data.shape[1], 200)
                #flat_debkg = simple_debackground(flat_data, mask_array, xnodes,
                # smooth=5)
                #aperset = find_apertures(flat_debkg, mask_array,
                section = config['reduce.trace']
                aperset = find_apertures(flat_data, mask_array,
                            scan_step  = section.getint('scan_step'),
                            minimum    = section.getfloat('minimum'),
                            separation = section.get('separation'),
                            align_deg  = section.getint('align_deg'),
                            filling    = section.getfloat('filling'),
                            degree     = section.getint('degree'),
                            display    = section.getboolean('display'),
                            fig        = tracefig,
                            )

                # save the trace figure
                tracefig.adjust_positions()
                tracefig.suptitle('Trace for {}'.format(flat_filename), fontsize=15)
                tracefig.savefig(trace_figname)

                aperset.save_txt(aperset_filename)
                aperset.save_reg(aperset_regname, fiber=fiber,
                                color={'A':'green','B':'yellow'}[fiber])

            # append the flat data and mask
            flat_data_lst[fiber][flatname] = flat_data
            flat_norm_lst[fiber][flatname] = flat_data/exptime
            flat_mask_lst[fiber][flatname] = mask_array
            aperset_lst[fiber][flatname]   = aperset
            flat_info_lst[fiber][flatname] = {'exptime': exptime}

    ########################### Get flat fielding ##############################
    flatmap_lst = {}

    for fiber, fiber_group in sorted(flat_groups.items()):
        for flatname in sorted(fiber_group.keys()):

            # get filename of flat
            flat_filename = os.path.join(midproc,
                    'flat_{}_{}.fits'.format(fiber, flatname))

            hdu_lst = fits.open(flat_filename, mode='update')
            if len(hdu_lst)>=3:
                # sensitivity map already exists in fits file
                flatmap = hdu_lst[2].data
                hdu_lst.close()
            else:
                # do flat fielding
                print('*** Start parsing flat fielding: %s ***'%flat_filename)
                fig_aperpar = {
                    'debug': os.path.join(report,
                            'flat_aperpar_{}_{}_%03d.{}'.format(
                                fiber, flatname, fig_format)),
                    'normal': None,
                    }[mode]

                fig_slit = os.path.join(report,
                                'slit_flat_{}_{}.{}'.format(
                                    fiber, flatname, fig_format))
    
                section = config['reduce.flat']
    
                flatmap = get_fiber_flat(
                            data            = flat_data_lst[fiber][flatname],
                            mask            = flat_mask_lst[fiber][flatname],
                            apertureset     = aperset_lst[fiber][flatname],
                            slit_step       = section.getint('slit_step'),
                            nflat           = len(flat_groups[fiber][flatname]),
                            q_threshold     = section.getfloat('q_threshold'),
                            smooth_A_func   = smooth_aperpar_A,
                            smooth_k_func   = smooth_aperpar_k,
                            smooth_c_func   = smooth_aperpar_c,
                            smooth_bkg_func = smooth_aperpar_bkg,
                            fig_aperpar     = fig_aperpar,
                            fig_overlap     = None,
                            fig_slit        = fig_slit,
                            slit_file       = None,
                            )

                # append the sensivity map to fits file
                hdu_lst.append(fits.ImageHDU(flatmap))
                # write back to the original file
                hdu_lst.flush()
    
            # append the flatmap
            if fiber not in flatmap_lst:
                flatmap_lst[fiber] = {}
            flatmap_lst[fiber][flatname] = flatmap
    
            # continue to the next colored flat
        # continue to the next fiber

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    master_aperset = {}

    flat_fiber_lst = []

    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        fiber_flat_lst = flat_groups[fiber]

        # determine the mosaiced flat filename
        flat_fiber_file = os.path.join(midproc,
                            'flat_{}.fits'.format(fiber))
        trac_fiber_file = os.path.join(midproc,
                            'trace_{}.trc'.format(fiber))
        treg_fiber_file = os.path.join(midproc,
                            'trace_{}.reg'.format(fiber))

        if len(fiber_flat_lst) == 1:
            # there's only ONE "color" of flat
            flatname = list(fiber_flat_lst)[0]

            # copy the flat fits
            oriname = 'flat_{}_{}.fits'.format(fiber, flatname)
            shutil.copyfile(os.path.join(midproc, oriname), flat_fiber_file)

            '''
            # copy the trc file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.trc'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.trc'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), trac_fiber_file)

            # copy the reg file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.reg'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.reg'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), treg_fiber_file)
            '''

            flat_map = flatmap_lst[fiber][flatname]
    
            # no need to mosaic aperset
            master_aperset[fiber] = list(aperset_lst[fiber].values())[0]
        else:
            # mosaic apertures
            section = config['reduce.flat']
            # determine the mosaic order
            name_lst = sorted(flat_info_lst[fiber],
                        key=lambda x: flat_info_lst[fiber].get(x)['exptime'])

            # if there is no flat data in this fiber. continue
            if len(aperset_lst[fiber])==0:
                continue

            master_aperset[fiber] = mosaic_flat_auto(
                    aperture_set_lst = aperset_lst[fiber],
                    max_count        = section.getfloat('mosaic_maxcount'),
                    name_lst         = name_lst,
                    )
            # mosaic original flat images
            flat_data = mosaic_images(flat_data_lst[fiber],
                                        master_aperset[fiber])
            # mosaic flat mask images
            mask_data = mosaic_images(flat_mask_lst[fiber],
                                        master_aperset[fiber])
            # mosaic sensitivity map
            flat_map = mosaic_images(flatmap_lst[fiber],
                                        master_aperset[fiber])
            # mosaic exptime-normalized flat images
            flat_norm = mosaic_images(flat_norm_lst[fiber],
                                        master_aperset[fiber])

            # change contents of several lists
            flat_data_lst[fiber] = flat_data
            flat_mask_lst[fiber] = mask_data
            flatmap_lst[fiber]   = flat_map
            flat_norm_lst[fiber] = flat_norm

            flat_fiber_lst.append(fiber)
    
            # pack and save to fits file
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data),
                        fits.ImageHDU(mask_data),
                        fits.ImageHDU(flat_map),
                        fits.ImageHDU(flat_norm),
                        ])
            hdu_lst.writeto(flat_fiber_file, overwrite=True)

    # fill blank fibers
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        if fiber not in master_aperset:
            master_aperset[fiber] = master_aperset['A'].copy()
            offset = fiber_offsets[ifiber-1]
            master_aperset[fiber].add_offset(offset)

    # align different fibers

    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)

        # align different fibers
        if ifiber == 0:
            ref_aperset = master_aperset[fiber]
        else:
            # find the postion offset (yshift) relative to the first fiber ("A")
            # the postion offsets are identified by users in the config file.
            # the first one (index=0) is shift of fiber B. second one is C...
            yshift = fiber_offsets[ifiber-1]
            offset = master_aperset[fiber].find_aper_offset(
                        ref_aperset, yshift=yshift)

            # print and logging
            message = 'fiber {}, aperture offset = {}'.format(fiber, offset)
            print(message)
            logger.info(message)

            # correct the aperture offset
            master_aperset[fiber].shift_aperture(-offset)

    # find all the aperture list for all fibers
    allmax_aper = -99
    allmin_aper = 999
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        allmax_aper = max(allmax_aper, max(master_aperset[fiber]))
        allmin_aper = min(allmin_aper, min(master_aperset[fiber]))

    #fig = plt.figure(dpi=150)
    #ax = fig.gca()
    #test_data = {'A': np.ones((2048, 2048))+1,
    #             'B': np.ones((2048, 2048))+2}

    # pack all aperloc into a single list
    all_aperloc_lst = []
    for fiber in flat_fiber_lst:
        aperset = master_aperset[fiber]
        for aper, aperloc in aperset.items():
            x, y = aperloc.get_position()
            center = aperloc.get_center()
            all_aperloc_lst.append([fiber, aper, aperloc, center])
            #ax.plot(x, y, color='gy'[ifiber], lw=1)

    # mosaic flat map
    sorted_aperloc_lst = sorted(all_aperloc_lst, key=lambda x:x[3])
    h, w = flat_map.shape
    master_flatdata = np.ones_like(flat_data)
    master_flatmask = np.ones_like(mask_data)
    master_flatmap  = np.ones_like(flat_map)
    master_flatnorm = np.ones_like(flat_norm)
    yy, xx = np.mgrid[:h, :w]
    prev_line = np.zeros(w)
    for i in np.arange(len(sorted_aperloc_lst)-1):
        fiber, aper, aperloc, center = sorted_aperloc_lst[i]
        x, y = aperloc.get_position()
        next_fiber, _, next_aperloc, _ = sorted_aperloc_lst[i+1]
        next_x, next_y = next_aperloc.get_position()
        next_line = np.int32(np.round((y + next_y)/2.))
        #print(fiber, aper, center, prev_line, next_line)
        mask = (yy >= prev_line)*(yy < next_line)
        master_flatdata[mask] = flat_data_lst[fiber][mask]
        master_flatmask[mask] = flat_mask_lst[fiber][mask]
        master_flatmap[mask]  = flatmap_lst[fiber][mask]
        master_flatnorm[mask] = flat_norm_lst[fiber][mask]
        prev_line = next_line
    # parse the last order
    mask = yy >= prev_line
    master_flatdata[mask] = flat_data_lst[next_fiber][mask]
    master_flatmask[mask] = flat_mask_lst[next_fiber][mask]
    master_flatmap[mask] = flatmap_lst[next_fiber][mask]
    master_flatnorm[mask] = flat_norm_lst[next_fiber][mask]

    #ax.imshow(master_flatmap, alpha=0.6)
    #plt.show()
    #print(h, w)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(master_flatdata),
                fits.ImageHDU(master_flatmask),
                fits.ImageHDU(master_flatmap),
                fits.ImageHDU(master_flatnorm),
                ])
    hdu_lst.writeto(flat_file, overwrite=True)

    ############################## Extract ThAr ################################

    # get the data shape
    h, w = flat_map.shape

    # define dtype of 1-d spectra for all fibers
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux',       (np.float32, w)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    calib_lst = {}
    # calib_lst is a hierarchical dict of calibration results
    # calib_lst = {
    #       'frameid1': {'A': calib_dict1, 'B': calib_dict2, ...},
    #       'frameid2': {'A': calib_dict1, 'B': calib_dict2, ...},
    #       ... ...
    #       }
    count_thar = 0
    for logitem in logtable:

        frameid = logitem['frameid']
        imgtype = logitem['imgtype']
        fileid  = logitem['fileid']
        exptime = logitem['exptime']

        if imgtype != 'cal':
            continue

        fiberobj_lst = [v.strip().lower()
                        for v in logitem['object'].split(';')]

        # check if there's any other objects
        has_others = False
        for fiberobj in fiberobj_lst:
            if len(fiberobj)>0 and fiberobj != 'thar':
                has_others = True
        if has_others:
            continue

        # now all objects in fiberobj_lst must be thar

        count_thar += 1
        print('Wavelength Calibration for {}'.format(fileid))

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst, overmean = correct_overscan(data, head, mask)
        for key, value in card_lst:
            head.append((key, value))

        # correct bias for ThAr, if has bias
        if bias is not None:
            data = data - bias
            logger.info('Bias corrected')
        else:
            logger.info('No bias. skipped bias correction')

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if fiberobj_lst[ifiber] != 'thar':
                continue

            section = config['reduce.extract']
            spectra1d = extract_aperset(data, mask,
                        apertureset = master_aperset[fiber],
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )

            # pack to a structured array
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                spec.append((aper, 0, flux_sum.size,
                        np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
            spec = np.array(spec, dtype=spectype)

            wlcalib_fig = os.path.join(report,
                    'wlcalib_{}_{}.{}'.format(fileid, fiber, fig_format))

            section = config['reduce.wlcalib']

            title = '{}.fits - Fiber {}'.format(fileid, fiber)

            if count_thar == 1:
                # this is the first ThAr frame in this observing run
                if section.getboolean('search_database'):
                    # find previouse calibration results
                    database_path = section.get('database_path')

                    ref_spec, ref_calib = select_calib_from_database(
                            database_path, head[statime_key])

                    if ref_spec is None or ref_calib is None:
                        # if failed, pop up a calibration window and
                        # identify the wavelengths manually
                        calib = wlcalib(spec,
                            figfilename = wlcalib_fig,
                            title       = title,
                            linelist    = section.get('linelist'),
                            window_size = section.getint('window_size'),
                            xorder      = section.getint('xorder'),
                            yorder      = section.getint('yorder'),
                            maxiter     = section.getint('maxiter'),
                            clipping    = section.getfloat('clipping'),
                            q_threshold = section.getfloat('q_threshold'),
                            )
                    else:
                        # if success, run recalib
                        # determine the direction
                        ref_direction = ref_calib['direction']
                        aperture_k = ((-1, 1)[direction[1]==ref_direction[1]],
                                        None)[direction[1]=='?']
                        pixel_k = ((-1, 1)[direction[2]==ref_direction[2]],
                                    None)[direction[2]=='?']
                        # determine the name of the output figure during lamp
                        # shift finding.
                        fig_ccf = {'normal': None,
                                    'debug': os.path.join(report,
                                        'lamp_ccf_{:+2d}_{:+03d}.png')}[mode]
                        fig_scatter = {'normal': None,
                                        'debug': os.path.join(report,
                                            'lamp_ccf_scatter.png')}[mode]

                        result = find_caliblamp_offset(ref_spec, spec,
                                    aperture_k  = aperture_k,
                                    pixel_k     = pixel_k,
                                    fig_ccf     = fig_ccf,
                                    fig_scatter = fig_scatter,
                                    )
                        aperture_koffset = (result[0], result[1])
                        pixel_koffset    = (result[2], result[3])

                        print('Aperture offset =', aperture_koffset)
                        print('Pixel offset =', pixel_koffset)

                        use = section.getboolean('use_prev_fitpar')
                        xorder      = (section.getint('xorder'), None)[use]
                        yorder      = (section.getint('yorder'), None)[use]
                        maxiter     = (section.getint('maxiter'), None)[use]
                        clipping    = (section.getfloat('clipping'), None)[use]
                        window_size = (section.getint('window_size'), None)[use]
                        q_threshold = (section.getfloat('q_threshold'), None)[use]

                        calib = recalib(spec,
                            figfilename      = wlcalib_fig,
                            title            = title,
                            ref_spec         = ref_spec,
                            linelist         = section.get('linelist'),
                            aperture_koffset = aperture_koffset,
                            pixel_koffset    = pixel_koffset,
                            ref_calib        = ref_calib,
                            xorder           = xorder,
                            yorder           = yorder,
                            maxiter          = maxiter,
                            clipping         = clipping,
                            window_size      = window_size,
                            q_threshold      = q_threshold,
                            direction        = direction,
                            )
                else:
                    # do not search the database
                    calib = wlcalib(spec,
                        figfilename   = wlcalib_fig,
                        title         = title,
                        identfilename = section.get('ident_file', None),
                        linelist      = section.get('linelist'),
                        window_size   = section.getint('window_size'),
                        xorder        = section.getint('xorder'),
                        yorder        = section.getint('yorder'),
                        maxiter       = section.getint('maxiter'),
                        clipping      = section.getfloat('clipping'),
                        q_threshold   = section.getfloat('q_threshold'),
                        )

                # then use this ThAr as the reference
                ref_calib = calib
                ref_spec  = spec
            else:
                # for other ThArs, no aperture offset
                calib = recalib(spec,
                    figfilename      = wlcalib_fig,
                    title            = title,
                    ref_spec         = ref_spec,
                    linelist         = section.get('linelist'),
                    ref_calib        = ref_calib,
                    aperture_koffset = (1, 0),
                    pixel_koffset    = (1, 0),
                    xorder           = ref_calib['xorder'],
                    yorder           = ref_calib['yorder'],
                    maxiter          = ref_calib['maxiter'],
                    clipping         = ref_calib['clipping'],
                    window_size      = ref_calib['window_size'],
                    q_threshold      = ref_calib['q_threshold'],
                    direction        = direction,
                    )

            # add more infos in calib
            calib['fileid']   = fileid
            calib['date-obs'] = head[statime_key]
            calib['exptime']  = head[exptime_key]

            # reference the ThAr spectra
            spec, card_lst, identlist = reference_self_wavelength(spec, calib)

            # save calib results into fits header
            for key, value in card_lst:
                key = 'HIERARCH GAMSE WLCALIB '+key
                head.append((key, value))

            # save onedspec into FITS
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        fits.BinTableHDU(identlist),
                        ])
            filename = os.path.join(midproc,
                                    'wlcalib.{}.{}.fits'.format(fileid, fiber))
            hdu_lst.writeto(filename, overwrite=True)

            # pack to calib_lst
            if frameid not in calib_lst:
                calib_lst[frameid] = {}
            calib_lst[frameid][fiber] = calib
            
        # fiber loop ends here

    # print fitting summary
    fmt_string = (' [{:3d}] {}'
                    ' - fiber {:1s} ({:4g} sec)'
                    ' - {:4d}/{:4d} r.m.s. = {:7.5f}')
    for frameid, calib_fiber_lst in sorted(calib_lst.items()):
        for fiber, calib in sorted(calib_fiber_lst.items()):
            print(fmt_string.format(frameid, calib['fileid'], fiber,
                calib['exptime'], calib['nuse'], calib['ntot'], calib['std']))

    # print promotion and read input frameid list
    ref_frameid_lst  = {}
    ref_calib_lst    = {}
    ref_datetime_lst = {}
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        while(True):
            string = input('Select References for fiber {}: '.format(fiber))
            ref_frameid_lst[fiber]  = []
            ref_calib_lst[fiber]    = []
            ref_datetime_lst[fiber] = []
            succ = True
            for s in string.split(','):
                s = s.strip()
                if len(s)>0 and s.isdigit() and int(s) in calib_lst:
                    frameid = int(s)
                    calib   = calib_lst[frameid]
                    ref_frameid_lst[fiber].append(frameid)
                    if fiber in calib:
                        usefiber = fiber
                    else:
                        usefiber = list(calib.keys())[0]
                        print(('Warning: no ThAr for fiber {}. '
                                'Use fiber {} instead').format(fiber, usefiber))
                    use_calib = calib[usefiber]
                    ref_calib_lst[fiber].append(use_calib)
                    ref_datetime_lst[fiber].append(use_calib['date-obs'])
                else:
                    print('Warning: "{}" is an invalid calib frame'.format(s))
                    succ = False
                    break
            if succ:
                break
            else:
                continue

    #################### Extract Science Spectrum ##############################

    for logitem in logtable:

        # logitem alias
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']

        if imgtype != 'sci':
            continue

        filename = os.path.join(rawdata, fileid+'.fits')

        logger.info('FileID: {} ({}) - start reduction: {}'.format(
            fileid, imgtype, filename))

        # read raw data
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data, card_lst, overmean = correct_overscan(data, head, mask)
        for key, value in card_lst:
            head.append((key, value))
        message = 'FileID: {} - overscan corrected'.format(fileid)
        logger.info(message)
        print(message)

        # correct bias
        if bias is not None:
            data = data - bias
            fmt_str = 'FileID: {} - bias corrected. mean value = {}'
            message = fmt_str.format(fileid, bias.mean())
        else:
            message = 'FileID: {} - no bias'%(fileid)
        logger.info(message)
        print(message)

        # correct flat
        data = data/flat_map
        message = 'FileID: {} - flat corrected'.format(fileid)
        logger.info(message)
        print(message)

        # correct background
        fiberobj_lst = [v.strip().lower() for v in logitem['object'].split(';')]
        fig_sec = os.path.join(report,
                  'bkg_{}_sec.{}'.format(fileid, fig_format))

        # find apertureset list for this item
        apersets = {}
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if len(fiberobj_lst[ifiber])>0:
                apersets[fiber] = master_aperset[fiber]

        section = config['reduce.background']
        stray = find_background(data, mask,
                aperturesets = apersets,
                ncols        = section.getint('ncols'),
                distance     = section.getfloat('distance'),
                yorder       = section.getint('yorder'),
                fig_section  = fig_sec,
                )
        data = data - stray

        # plot stray light
        fig_stray = os.path.join(report,
                    'bkg_{}_stray.{}'.format(fileid, fig_format))
        plot_background_aspect1(data+stray, stray, fig_stray)

        message = 'FileID: {} - background corrected. max value = {}'.format(
                fileid, stray.max())
        logger.info(message)
        print(message)

        # extract 1d spectrum
        section = config['reduce.extract']
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if fiberobj_lst[ifiber]=='':
                # nothing in this fiber
                continue
            lower_limits = {'A':section.getfloat('lower_limit'), 'B':4}
            upper_limits = {'A':section.getfloat('upper_limit'), 'B':4}

            spectra1d = extract_aperset(data, mask,
                            apertureset = master_aperset[fiber],
                            lower_limit = lower_limits[fiber],
                            upper_limit = upper_limits[fiber],
                        )

            fmt_string = ('FileID: {}'
                            ' - fiber {}'
                            ' - 1D spectra of {} orders extracted')
            message = fmt_string.format(fileid, fiber, len(spectra1d))
            logger.info(message)
            print(message)

            # pack spectrum
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                item = (aper, 0, flux_sum.size,
                        np.zeros_like(flux_sum, dtype=np.float64),
                        flux_sum
                        )
                spec.append(item)
            spec = np.array(spec, dtype=spectype)

            # wavelength calibration
            weight_lst = get_time_weight(ref_datetime_lst[fiber],
                                        head[statime_key])

            message = ('FileID: {} - fiber {}'
                        ' - wavelength calibration weights: {}').format(
                        fileid, fiber,
                        ','.join(['%8.4f'%w for w in weight_lst])
                        )
            logger.info(message)
            print(message)

            spec, card_lst = reference_wavelength(
                                spec,
                                ref_calib_lst[fiber],
                                weight_lst,
                                )

            for key, value in card_lst:
                key = 'HIERARCH GAMSE WLCALIB '+key
                head.append((key, value))

            # pack and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        ])
            filename = os.path.join(onedspec, '{}_{}_{}.fits'.format(
                                            fileid, fiber, oned_suffix))
            hdu_lst.writeto(filename, overwrite=True)

            message = 'FileID: {} - Spectra written to {}'.format(
                        fileid, filename)
            logger.info(message)
            print(message)


class Xinglong216HRS(Reduction):

    def __init__(self):
        super(Xinglong216HRS, self).__init__(instrument='Xinglong216HRS')

    def config_ccd(self):
        '''Set CCD images configurations.
        '''
        self.ccd_config

    def overscan(self):
        '''
        Overscan correction for Xinglong 2.16m Telescope HRS.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **skip**,    *bool*, Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,  *str*,  Suffix of the corrected files.
           **plot**,    *bool*, Plot the overscan levels if *yes*.
           **var_fig**, *str*,  Filename of the overscan variation figure.

        '''


        def fix_cr(a):
            m = a.mean(dtype=np.float64)
            s = a.std(dtype=np.float64)
            mask = a > m + 3.*s
            if mask.sum()>0:
                x = np.arange(a.size)
                f = InterpolatedUnivariateSpline(x[~mask],a[~mask],k=3)
                return f(x)
            else:
                return a
        
        # find output suffix for fits
        self.output_suffix = self.config.get('overscan', 'suffix')

        if self.config.getboolean('overscan', 'skip'):
            logger.info('Skip [overscan] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        # keywords for mask
        saturation_adu = 65535

        # path alias
        rawdata = self.paths['rawdata']
        midproc = self.paths['midproc']
        report  = self.paths['report']

        # loop over all files (bias, dark, ThAr, flat...)
        # to correct for the overscan

        # prepare the item list
        item_lst = [item for item in self.log]

        for i, item in enumerate(item_lst):
            logger.info('Correct overscan for item %3d: "%s"'%(
                         item.frameid, item.fileid))

            # read FITS data
            filename = '%s%s.fits'%(item.fileid, self.input_suffix)
            filepath = os.path.join(rawdata, filename)
            data, head = fits.getdata(filepath, header=True)

            h, w = data.shape
            x1, x2 = w-head['COVER'], w

            # find the overscan level along the y-axis
            ovr_lst1 = data[0:h//2,x1+2:x2].mean(dtype=np.float64, axis=1)
            ovr_lst2 = data[h//2:h,x1+2:x2].mean(dtype=np.float64, axis=1)

            ovr_lst1_fix = fix_cr(ovr_lst1)
            ovr_lst2_fix = fix_cr(ovr_lst2)

            # apply the sav-gol fitler to the mean of overscan
            ovrsmooth1 = savgol_filter(ovr_lst1_fix, window_length=301, polyorder=3)
            ovrsmooth2 = savgol_filter(ovr_lst2_fix, window_length=301, polyorder=3)

            # plot the overscan regions
            if i%5 == 0:
                fig = plt.figure(figsize=(10,6), dpi=150)

            ax1 = fig.add_axes([0.08, 0.83-(i%5)*0.185, 0.42, 0.15])
            ax2 = fig.add_axes([0.55, 0.83-(i%5)*0.185, 0.42, 0.15])

            ax1.plot([0,0],[ovr_lst1_fix.min(), ovr_lst1_fix.max()], 'w-', alpha=0)
            _y1, _y2 = ax1.get_ylim()
            ax1.plot(np.arange(0, h//2), ovr_lst1, 'r-', alpha=0.3)
            ax1.set_ylim(_y1, _y2)

            ax2.plot([0,0],[ovr_lst2_fix.min(), ovr_lst2_fix.max()], 'w-', alpha=0)
            _y1, _y2 = ax2.get_ylim()
            ax2.plot(np.arange(h//2, h), ovr_lst2, 'b-', alpha=0.3)
            ax2.set_ylim(_y1, _y2)

            ax1.plot(np.arange(0, h//2), ovrsmooth1, 'm', ls='-')
            ax2.plot(np.arange(h//2, h), ovrsmooth2, 'c', ls='-')
            ax1.set_ylabel('ADU')
            ax2.set_ylabel('')
            ax1.set_xlim(0, h//2-1)
            ax2.set_xlim(h//2, h-1)
            for ax in [ax1, ax2]:
                _x1, _x2 = ax.get_xlim()
                _y1, _y2 = ax.get_ylim()
                _x = 0.95*_x1 + 0.05*_x2
                _y = 0.20*_y1 + 0.80*_y2
                ax.text(_x, _y, item.fileid, fontsize=9)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.xaxis.set_major_locator(tck.MultipleLocator(500))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            if i%5==4 or i==len(item_lst)-1:
                ax1.set_xlabel('Y (pixel)')
                ax2.set_xlabel('Y (pixel)')
                figname = 'overscan_%02d.png'%(i//5+1)
                figfile = os.path.join(report, figname)
                fig.savefig(figfile)
                logger.info('Save image: %s'%figfile)
                plt.close(fig)

            # determine shape of output image (also the shape of science region)
            y1 = head['CRVAL2']
            y2 = y1 + head['NAXIS2'] - head['ROVER']
            ymid = (y1 + y2)//2
            x1 = head['CRVAL1']
            x2 = x1 + head['NAXIS1'] - head['COVER']
            newshape = (y2-y1, x2-x1)

            # find the saturation mask
            mask_sat = (data[y1:y2,x1:x2]>=saturation_adu)
            # get bad pixel mask
            bins = (head['RBIN'], head['CBIN'])
            mask_bad = self._get_badpixel_mask(newshape, bins=bins)

            mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2
            # save the mask
            mask_table = array_to_table(mask)
            maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
            maskpath = os.path.join(midproc, maskname)
            fits.writeto(maskpath, mask_table, overwrite=True)

            # subtract overscan
            new_data = np.zeros(newshape, dtype=np.float64)
            ovrdata1 = np.transpose(np.repeat([ovrsmooth1],x2-x1,axis=0))
            ovrdata2 = np.transpose(np.repeat([ovrsmooth2],x2-x1,axis=0))
            new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
            new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2

            # fix bad pixels
            new_data = fix_pixels(new_data, mask_bad, 'x', 'linear')

            # update fits header
            head['HIERARCH GAMSE OVERSCAN']        = True
            head['HIERARCH GAMSE OVERSCAN METHOD'] = 'smooth'

            # save data
            outname = '%s%s.fits'%(item.fileid, self.output_suffix)
            outpath = os.path.join(midproc, outname)
            fits.writeto(outpath, new_data, head, overwrite=True)
            print('Correct Overscan {} -> {}'.format(filename, outname))


        logger.info('Overscan corrected. Change suffix: %s -> %s'%
                    (self.input_suffix, self.output_suffix))
        self.input_suffix = self.output_suffix

    def _get_badpixel_mask(self, shape, bins):
        '''Get bad-pixel mask.


        Args:
            shape (tuple): Shape of the science data region.
            bins (tuple): Number of pixel bins of (ROW, COLUMN).
        Returns:
            :class:`numpy.array`: Binary mask indicating the bad pixels. The
                shape of the mask is the same as the input shape.

        The bad pixels are found when readout mode = Left Top & Bottom.

        '''
        mask = np.zeros(shape, dtype=np.bool)
        if bins == (1, 1) and shape == (4136, 4096):
            h, w = shape

            mask[349:352, 627:630] = True
            mask[349:h//2, 628]    = True

            mask[1604:h//2, 2452] = True

            mask[280:284,3701]   = True
            mask[274:h//2, 3702] = True
            mask[272:h//2, 3703] = True
            mask[274:282, 3704]  = True

            mask[1720:1722, 3532:3535] = True
            mask[1720, 3535]           = True
            mask[1722, 3532]           = True
            mask[1720:h//2,3533]       = True

            mask[347:349, 4082:4084] = True
            mask[347:h//2,4083]      = True

            mask[h//2:2631, 1909] = True
        else:
            print('No bad pixel information for this CCD size.')
            raise ValueError
        return mask



    def bias(self):
        '''Bias corrrection for Xinglong 2.16m Telescope HRS.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **skip**,        *bool*,  Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,      *str*,   Suffix of the corrected files.
           **cosmic_clip**, *float*, Upper clipping threshold to remove cosmic-rays.
           **file**,        *str*,   Name of bias file.

        '''
        self.output_suffix = self.config.get('bias', 'suffix')

        if self.config.getboolean('bias', 'skip'):
            logger.info('Skip [bias] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        midproc = self.paths['midproc']

        bias_id_lst = self.find_bias()

        if len(bias_id_lst) == 0:
            # no bias frame found. quit this method.
            # update suffix
            logger.info('No bias found.')
            return True

        infile_lst = [os.path.join(midproc,
                        '%s%s.fits'%(item.fileid, self.input_suffix))
                        for item in self.log if item.frameid in bias_id_lst]

        # import and stack all bias files in a data cube
        tmp = [fits.getdata(filename, header=True) for filename in infile_lst]
        all_data, all_head = list(zip(*tmp))
        all_data = np.array(all_data)

        if self.config.has_option('bias', 'cosmic_clip'):
            # use sigma-clipping method to mask cosmic rays
            cosmic_clip = self.config.getfloat('bias', 'cosmic_clip')

            all_mask = np.ones_like(all_data, dtype=np.bool)

            mask = (all_data == all_data.max(axis=0))
        
            niter = 0
            maxiter = 2
            while niter <= maxiter:
                niter += 1
                mdata = np.ma.masked_array(all_data, mask=mask)
                # calculate mean and standard deviation
                mean = mdata.mean(axis=0, dtype=np.float64).data
                std  = mdata.std(axis=0, dtype=np.float64, ddof=1).data
                # initialize new mask
                new_mask = np.ones_like(mask)>0
                # masking all the upper outliers.
                for i in np.arange(all_data.shape[0]):
                    new_mask[i,:,:] = all_data[i,:,:] > mean + cosmic_clip*std
                mask = new_mask

            mdata = np.ma.masked_array(all_data, mask=mask)
            bias = mdata.mean(axis=0, dtype=np.float64).data
        else:
            # no sigma clipping
            bias = alldata.mean(axis=0, dtype=np.float64)

        # create new FITS Header for bias
        head = fits.Header()
        head['HIERARCH GAMSE BIAS NFILE'] = len(bias_id_lst)

        # get final bias filename from the config file
        bias_file = self.config.get('bias', 'file')

        if self.config.has_option('bias', 'smooth_method'):
            # perform smoothing for bias
            smooth_method = self.config.get('bias', 'smooth_method')
            smooth_method = smooth_method.strip().lower()

            logger.info('Smoothing bias: %s'%smooth_method)

            if smooth_method in ['gauss','gaussian']:
                # perform 2D gaussian smoothing

                smooth_sigma = self.config.getint('bias', 'smooth_sigma')
                smooth_mode  = self.config.get('bias', 'smooth_mode')

                logger.info('Smoothing bias: sigma = %f'%smooth_sigma)
                logger.info('Smoothing bias: mode = %s'%smooth_mode)

                from scipy.ndimage.filters import gaussian_filter
                h, w = bias.shape
                bias_smooth = np.zeros((h, w), dtype=np.float64)
                bias_smooth[0:h/2,:] = gaussian_filter(bias[0:h/2,:],
                                                       smooth_sigma,
                                                       mode=smooth_mode)
                bias_smooth[h/2:h,:] = gaussian_filter(bias[h/2:h,:],
                                                       smooth_sigma,
                                                       mode=smooth_mode)

                logger.info('Smoothing bias: Update bias FITS header')

                head['HIERARCH GAMSE BIAS SMOOTH']        = True
                head['HIERARCH GAMSE BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                head['HIERARCH GAMSE BIAS SMOOTH SIGMA']  = smooth_sigma
                head['HIERARCH GAMSE BIAS SMOOTH MODE']   = smooth_mode

            else:
                pass

            # bias_data is a proxy for bias to be corrected for each frame
            bias_data = bias_smooth

            # plot comparison between un-smoothed and smoothed data
            self.plot_bias_smooth(bias, bias_smooth)

        else:
            # no smoothing
            logger.info('No smoothing parameter for bias. Skip bias smoothing')
            head['HIERARCH GAMSE BIAS SMOOTH'] = False
            bias_data = bias

        # save the bias to FITS
        fits.writeto(bias_file, bias_data, head, overwrite=True)
        
        self.plot_bias_variation(all_data, all_head, time_key='DATE-STA')

        # finally all files are corrected for the bias
        for item in self.log:
            if item.frameid not in bias_id_lst:
                infile  = '%s%s.fits'%(item.fileid, self.input_suffix)
                outfile = '%s%s.fits'%(item.fileid, self.output_suffix)
                inpath  = os.path.join(midproc, infile)
                outpath = os.path.join(midproc, outfile)
                data, head = fits.getdata(inpath, header=True)
                data_new = data - bias_data
                # write information into FITS header
                head['HIERARCH GAMSE BIAS'] = True
                # save the bias corrected data
                fits.writeto(outpath, data_new, head, overwrite=True)
                info = ['Correct bias for item no. %d.'%item.frameid,
                        'Save bias corrected file: "%s"'%outpath]
                logger.info((os.linesep+'  ').join(info))
                print('Correct bias: {} => {}'.format(infile, outfile))

        # update suffix
        logger.info('Bias corrected. Change suffix: %s -> %s'%
                    (self.input_suffix, self.output_suffix))
        self.input_suffix = self.output_suffix
        return True

all_columns = [
        ('frameid', 'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',  'str',   '{:^12s}', '{0[fileid]:12s}'),
        ('imgtype', 'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',  'str',   '{:^12s}', '{0[object]:12s}'),
        ('i2cell',  'bool',  '{:^6s}',  '{0[i2cell]!s: <6}'),
        ('exptime', 'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate', 'time',  '{:^23s}', '{0[obsdate]:}'),
        ('nsat',    'int',   '{:^10s}', '{0[nsat]:10d}'),
        ('q95',     'int',   '{:^10s}', '{0[q95]:10d}'),
        ]

def print_wrapper(string, item):
    """A wrapper for log printing for Xinglong216HRS pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype = item['imgtype']
    obj     = item['object']

    if len(obj)>=4 and obj[0:4].lower()=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif len(obj)>=4 and obj[0:4].lower()=='thar':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

def make_config():
    """Generate a config file for reducing the data taken with Xinglong 2.16m
    HRS.


    """

    # find date of data obtained
    current_pathname = os.path.basename(os.getcwd())
    guess_date = extract_date(current_pathname)

    while(True):
        if guess_date is None:
            prompt = 'YYYYMMDD'
        else:
            prompt = guess_date

        string = input('Date of observation [{}]: '.format(prompt))
        input_date = extract_date(string)
        if input_date is None:
            if guess_date is None:
                continue
            else:
                input_date = guess_date
                break
        else:
            break
   
    input_datetime = datetime.datetime.strptime(input_date, '%Y-%m-%d')

    # determine the fiber mode
    while(True):
        string = input(
            'The data was obatined with Single fiber or Double fibers? [s/d]:')
        if string == 's':
            fibermode = 'single'
            break
        elif string == 'd':
            fibermode = 'double'
            break
        else:
            print('Invalid input: {}'.format(string))
            continue

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    # determine the time-dependent keywords
    if input_datetime > datetime.datetime(2009, 1, 1):
        # since 2019 there's another type of FITS header
        statime_key = 'DATE-OBS'
        exptime_key = 'EXPOSURE'
    else:
        statime_key = 'DATE-STA'
        exptime_key = 'EXPTIME'

    config.set('data', 'telescope',   'Xinglong216')
    config.set('data', 'instrument',  'HRS')
    config.set('data', 'rawdata',     'rawdata')
    config.set('data', 'statime_key', statime_key)
    config.set('data', 'exptime_key', exptime_key)
    config.set('data', 'direction',   'xr-')
    config.set('data', 'fibermode',   fibermode)
    if fibermode == 'double':
        config.set('data', 'fiberoffset', str(-12))

    config.add_section('reduce')
    config.set('reduce', 'midproc',     'midproc')
    config.set('reduce', 'report',      'report')
    config.set('reduce', 'onedspec',    'onedspec')
    config.set('reduce', 'mode',        'normal')
    config.set('reduce', 'oned_suffix', 'ods')
    config.set('reduce', 'fig_format',  'png')
    
    config.add_section('reduce.bias')
    config.set('reduce.bias', 'bias_file',     '${reduce:midproc}/bias.fits')
    config.set('reduce.bias', 'cosmic_clip',   str(10))
    config.set('reduce.bias', 'maxiter',       str(5))
    config.set('reduce.bias', 'smooth',        'yes')
    config.set('reduce.bias', 'smooth_method', 'gaussian')
    config.set('reduce.bias', 'smooth_sigma',  str(3))
    config.set('reduce.bias', 'smooth_mode',   'nearest')

    config.add_section('reduce.trace')
    config.set('reduce.trace', 'minimum',    str(8))
    config.set('reduce.trace', 'scan_step',  str(100))
    config.set('reduce.trace', 'separation', '500:21, 3500:52')
    config.set('reduce.trace', 'filling',    str(0.3))
    config.set('reduce.trace', 'align_deg',  str(2))
    config.set('reduce.trace', 'display',    'no')
    config.set('reduce.trace', 'degree',     str(3))

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'slit_step',       str(256))
    config.set('reduce.flat', 'q_threshold',     str(50))
    config.set('reduce.flat', 'mosaic_maxcount', str(50000))

    config.add_section('reduce.wlcalib')
    config.set('reduce.wlcalib', 'search_database', 'yes')
    config.set('reduce.wlcalib', 'database_path',
                                    '/opt/gamse/Xinglong216.HRS/wlcalib')
    config.set('reduce.wlcalib', 'linelist',        'thar.dat')
    config.set('reduce.wlcalib', 'use_prev_fitpar', 'yes')
    config.set('reduce.wlcalib', 'window_size',     str(13))
    config.set('reduce.wlcalib', 'xorder',          str(3))
    config.set('reduce.wlcalib', 'yorder',          str(3))
    config.set('reduce.wlcalib', 'maxiter',         str(5))
    config.set('reduce.wlcalib', 'clipping',        str(3))
    config.set('reduce.wlcalib', 'q_threshold',     str(10))

    config.add_section('reduce.background')
    config.set('reduce.background', 'ncols',    str(9))
    config.set('reduce.background', 'distance', str(7))
    config.set('reduce.background', 'yorder',   str(7))

    config.add_section('reduce.extract')
    config.set('reduce.extract', 'upper_limit', str(7))
    config.set('reduce.extract', 'lower_limit', str(7))

    # write to config file
    filename = 'Xinglong216HRS.{}.cfg'.format(input_date)
    outfile = open(filename, 'w')
    for section in config.sections():
        maxkeylen = max([len(key) for key in config[section].keys()])
        outfile.write('[{}]'.format(section)+os.linesep)
        fmt = '{{:{}s}} = {{}}'.format(maxkeylen)
        for key, value in config[section].items():
            outfile.write(fmt.format(key, value)+os.linesep)
        outfile.write(os.linesep)
    outfile.close()

    print('Config file written to {}'.format(filename))
    

def make_obslog(path):
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running.
    The name of the ascii file is `YYYY-MM-DD.obslog`, where `YYYY-MM-DD` is the
    date of the *first* FITS image in the data folder.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.

    Args:
        path (str): Path to the raw FITS files.

    """
    cal_objects = ['bias', 'flat', 'dark', 'i2', 'thar']
    regular_names = ('Bias', 'Flat', 'ThAr', 'I2')

    # if the obsinfo file exists, read and pack the information
    addinfo_lst = {}
    obsinfo_file = 'obsinfo.txt'
    has_obsinfo = os.path.exists(obsinfo_file)
    if has_obsinfo:
        #io_registry.register_reader('obslog', Table, read_obslog)
        #addinfo_table = Table.read(obsinfo_file, format='obslog')
        addinfo_table = read_obslog(obsinfo_file)
        addinfo_lst = {row['frameid']:row for row in addinfo_table}
        # prepare the difference list between real observation time and FITS
        # time
        real_obsdate_lst = []
        delta_t_lst = []

    # scan the raw files
    fname_lst = sorted(os.listdir(path))

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid', 'S12'),  ('imgtype', 'S3'),
        ('object',  'S12'), ('i2cell', 'bool'), ('exptime', 'f4'),
        ('obsdate', Time),  ('nsat',   'i4'),   ('q95',     'i4'),
        ])

    # prepare infomation to print
    pinfo = FormattedInfo(all_columns,
            ['frameid', 'fileid', 'imgtype', 'object', 'i2cell', 'exptime',
            'obsdate', 'nsat', 'q95'])

    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    #print(pinfo.get_dtype())
    print(pinfo.get_separator())

    prev_frameid = -1
    # start scanning the raw files
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid  = fname[0:-5]
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        # determine the science and overscan regions
        naxis1 = head['NAXIS1']
        naxis2 = head['NAXIS2']
        x1 = head.get('CRVAL1', 0)
        y1 = head.get('CRVAL2', 0)
        # get science region along x axis
        cover = head.get('COVER')
        if cover is None:
            if naxis1 >= 4096:
                cover = naxis1 - 4096
        # get science region along y axis
        rover = head.get('ROVER')
        if rover is None:
            if naxis2 >= 4136:
                rover = naxis2 - 4136

        # get start and end indices of science region
        y2 = y1 + naxis2 - rover
        x2 = x1 + naxis1 - cover
        data = data[y1:y2,x1:x2]

        # find frame-id
        frameid = int(fileid[8:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(
                    frameid, prev_frameid))

        # parse obsdate
        if 'DATE-STA' in head:
            obsdate = Time(head['DATE-STA'])
        else:
            obsdate = Time(head['DATE-OBS'])
        if (frameid in addinfo_lst and 'obsdate' in addinfo_table.colnames
            and addinfo_lst[frameid]['obsdate'] is not np.ma.masked):
            real_obsdate = addinfo_lst[frameid]['obsdate'].datetime
            file_obsdate = obsdate.datetime
            delta_t = real_obsdate - file_obsdate
            real_obsdate_lst.append(real_obsdate)
            delta_t_lst.append(delta_t.total_seconds())

        if 'EXPTIME' in head:
            exptime = head['EXPTIME']
        else:
            exptime = head['EXPOSURE']

        # parse object name
        if 'OBJECT' in head:
            objectname = head['OBJECT'].strip()
        else:
            objectname = ''
        if (frameid in addinfo_lst and 'object' in addinfo_table.colnames
            and addinfo_lst[frameid]['object'] is not np.ma.masked):
            objectname = addinfo_lst[frameid]['object']
        # change to regular name
        for regname in regular_names:
            if objectname.lower() == regname.lower():
                objectname = regname
                break

        # parse I2 cell
        i2cell = objectname.lower()=='i2'
        if (frameid in addinfo_lst and 'i2cell' in addinfo_table.colnames
            and addinfo_lst[frameid]['i2cell'] is not np.ma.masked):
            i2cell = addinfo_lst[frameid]['i2cell']

        imgtype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        item = [frameid, fileid, imgtype, objectname, i2cell, exptime, obsdate,
                saturation, quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string = pinfo.get_format(has_esc=False).format(item)
        print(print_wrapper(string, item))

        prev_frameid = frameid

    print(pinfo.get_separator())
    
    # sort by obsdate
    #logtable.sort('obsdate')

    if has_obsinfo and len(real_obsdate_lst)>0:
        # determine the time offset as median value
        time_offset = np.median(np.array(delta_t_lst))
        time_offset_dt = datetime.timedelta(seconds=time_offset)
        # plot time offset
        fig = plt.figure(figsize=(9, 6), dpi=100)
        ax = fig.add_axes([0.12,0.16,0.83,0.77])
        xdates = mdates.date2num(real_obsdate_lst)
        ax.plot_date(xdates, delta_t_lst, 'o-', ms=6)
        ax.axhline(y=time_offset, color='k', ls='--', alpha=0.6)
        ax.set_xlabel('Log Time', fontsize=12)
        ax.set_ylabel('Log Time - FTIS Time (sec)', fontsize=12)
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.text(0.95*x1+0.05*x2, 0.1*y1+0.9*y2,
                'Time offset = %d seconds'%time_offset, fontsize=14)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.grid(True, ls='-', color='w')
        ax.set_facecolor('#eaeaf6')
        ax.set_axisbelow(True)
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        for t in ax.xaxis.get_ticklines():
            t.set_color('none')
        for t in ax.yaxis.get_ticklines():
            t.set_color('none')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        #plt.setp(ax.get_xticklabels(), rotation=30)i
        fig.autofmt_xdate()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        fig.suptitle('Time Offsets Between Log and FITS', fontsize=15)
        fig.savefig('obsdate_offset.png')
        plt.close(fig)

        # correct time offset
        for row in logtable:
            row['obsdate'] = row['obsdate'] + time_offset_dt

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'].iso[0:10]
    outname = '{}.obslog'.format(obsdate)
    if os.path.exists(outname):
        i = 0
        while(True):
            i += 1
            outname = '{}.{}.obslog'.format(obsdate, i)
            if not os.path.exists(outname):
                outfilename = outname
                break
    else:
        outfilename = outname

    # save the logtable
    outfile = open(outfilename, 'w')
    outfile.write(pinfo.get_title()+os.linesep)
    outfile.write(pinfo.get_dtype()+os.linesep)
    outfile.write(pinfo.get_separator()+os.linesep)
    for row in logtable:
        outfile.write(pinfo.get_format().format(row)+os.linesep)
    outfile.close()
