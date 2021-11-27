import os
import re
import math
import datetime
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
import scipy.signal as sg
import scipy.interpolate as intp
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ...echelle.imageproc import combine_images
from ...echelle.trace import TraceFigureCommon
from ...echelle.flat import ProfileNormalizerCommon
from ...echelle.background import BackgroundFigureCommon
from ...echelle.wlcalib import get_calib_from_header
from ...utils.download import get_file
from ...utils.onedarray import iterative_savgol_filter, get_edge_bin


def get_region_lst(header, readout_mode):
    """Get a list of (science, overscan) rectangles of the CCD regions based on
    input header and readout mode.

    Args:
        header (:class:`astropy.io.fits.Header`): FITS header.
        readout_mode (str): Readout Mode of CCD.

    Returns:
        tuple: A tuple of region indices ``((sci1, ovr1), (sci2, ovr2), ...)``,
        where ``sciN`` and ``ovrN`` are indices ``(y1, y2, x1, x2)`` of the N-th
        science region and overscan region.

    See also:

        * :func:`get_sci_region`
        * :func:`get_ovr_region`
        * :func:`get_ccd_geometry`

    """
    naxis1 = header['NAXIS1']     # size along X axis
    naxis2 = header['NAXIS2']     # size along Y axis
    x1, y1, xbin, ybin, cover, rover = get_ccd_geometry(header)

    if readout_mode in ['Left Top & Bottom', 'Left Bottom & Right Top']:
        # 2 regions vertically
        # science & overscan region
        sci1 = (y1, y1+(naxis2-rover)//2, x1, x1+(naxis1-cover))
        ovr1 = (y1, y1+(naxis2-rover)//2, x1+(naxis1-cover), naxis1)

        sci2 = (y1+(naxis2-rover)//2, naxis2-rover, x1, x1+(naxis1-cover))
        ovr2 = (y1+(naxis2-rover)//2, naxis2-rover, x1+(naxis1-cover), naxis1)

        return ((sci1, ovr1), (sci2, ovr2))
    else:
        print('Error: Wrong Readout Mode:', readout_mode)
        raise ValueError

def get_sci_region(header):
    """Get over science region rectangle.

    Args:
        header (:class:`astropy.io.fits.header`): FITS header.

    Returns:
        tuple: A tuple of indices ``(y1, y2, x1, x2)``, where the overall
        science region is ``data[y1:y2, x1:x2]``.

    See also:

        * :func:`get_ovr_region`
        * :func:`get_region_lst`
        * :func:`get_ccd_geometry`

    """
    naxis1 = header['NAXIS1']     # size along X axis
    naxis2 = header['NAXIS2']     # size along Y axis
    x1, y1, xbin, ybin, cover, rover = get_ccd_geometry(header)
    return (y1, y1+(naxis2-rover), x1, x1+(naxis1-cover))


def get_ovr_region(header):
    """Get over science region rectangle.

    Args:
        header (:class:`astropy.io.fits.header`): FITS header.

    Returns:
        tuple: A tuple of indices ``(y1, y2, x1, x2)``, where the overall
        overscan region is ``data[y1:y2, x1:x2]``.

    See also:

        * :func:`get_sci_region`
        * :func:`get_region_lst`
        * :func:`get_ccd_geometry`

    See also:

    """
    naxis1 = header['NAXIS1']     # size along X axis
    naxis2 = header['NAXIS2']     # size along Y axis
    x1, y1, xbin, ybin, cover, rover = get_ccd_geometry(header)
    return (y1, y1+(naxis2-rover), x1+(naxis1-cover), x1+naxis1)

def get_ccd_geometry(header):
    """Get basic geometry of CCD.

    Args:
        header (:class:`astropy.io.fits.header`): FITS header.

    Returns:
        tuple: A tuple of **(x1, y1, xbin, ybin, cover, rover)**, where
        **(x1, y1)** is the starting point of science region,
        **(xbin, ybin)** are the CCD binning along X and Y axes,
        and **(cover, rover)** are the numbers of overscan columns and rows.

    See also:

        * :func:`get_region_lst`
        * :func:`get_sci_region`
        * :func:`get_ovr_region`

    Check the information in the FITS header, and determined some important
    geometric parameters of the image.
    The return values are **(x1, y1, xbin, ybin, cover, rover)**, where

    * **(x1, y1)** is the starting point of science region. Normally
      **(x1, y1)** = (0, 0).
    * **(xbin, ybin)** are the CCD binning along X and Y axes.
    * **(cover, rover)** are the numbers of overscan columns and rows.
      **rover** is always 0 as the readout direction of this CCD is along X axis
      and the overscan is after the readout of each row.
      If the ``ROVER`` in the FITS header is not 0, this function will raise an
      exception.
    """
    naxis1 = header['NAXIS1']     # size along X axis
    naxis2 = header['NAXIS2']     # size along Y axis
    x1 = header.get('CRVAL1', 0)  # X origin
    y1 = header.get('CRVAL2', 0)  # Y origin

    # total pixels along Y and X axis
    ny, nx = 4136, 4096

    # get XBIN
    if naxis1 >= nx:
        xbin = 1
    elif naxis1 >= nx//2:
        xbin = 2
    elif naxis1 >= nx//4:
        xbin = 4
    else:
        raise ValueError

    # get YBIN
    if naxis2 == ny:
        ybin = 1
    elif naxis2 == ny//2:
        ybin = 2
    elif naxis2 == ny//4:
        ybin = 4
    else:
        raise ValueError

    # check if the determined xbin and ybin are consistent with the header
    if 'CBIN' in header and xbin != header['CBIN']:
        print('Warning: CBIN ({}) not consistent with XBIN ({})'.format(
                header['CBIN'], xbin))
        raise ValueError

    if 'RBIN' in header and ybin != header['RBIN']:
        print('Warning: RBIN ({}) not consistent with YBIN ({})'.format(
                header['RBIN'], ybin))
        raise ValueError

    # get COVER
    cover = header.get('COVER')
    if cover is None:
        if naxis1 >= nx:
            cover = naxis1 - nx
        elif naxis1 >= nx//2:
            cover = naxis1 - nx//2
        elif naxis1 >= nx//4:
            cover = naxis1 - nx//4
        else:
            raise ValueError


    # get ROVER
    rover = header.get('ROVER', 0)
    # rover should = 0. if not, there must be addtional overscan region along Y
    if rover != 0:
        raise ValueError
    
    return x1, y1, xbin, ybin, cover, rover

def get_bias(config, logtable):
    """Get bias image.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) – Output bias image.
            * **bias_card_lst** (list) – List of FITS header cards related to
              the bias correction.

    """
    mode = config['reduce'].get('mode')
    bias_file = config['reduce.bias'].get('bias_file')
    
    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        bias = hdu_lst[-1].data
        head = hdu_lst[0].header
        hdu_lst.close()

        reobj = re.compile('GAMSE BIAS[\s\S]*')
        # filter header cards that match the above pattern
        bias_card_lst = [(card.keyword, card.value) for card in head.cards
                            if reobj.match(card.keyword)]

        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        bias, bias_card_lst = combine_bias(config, logtable)

    return bias, bias_card_lst

def combine_bias(config, logtable):
    """Combine the bias images.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) – Output bias image.
            * **bias_card_lst** (list) – List of FITS header cards related to
              the bias correction.

    """

    rawpath      = config['data']['rawpath']
    readout_mode = config['data']['readout_mode']

    # determine number of cores to be used
    ncores = config['reduce'].get('ncores')
    if ncores == 'max':
        ncores = os.cpu_count()
    else:
        ncores = min(os.cpu_count(), int(ncores))

    section = config['reduce.bias']
    bias_file = section['bias_file']

    bias_data_lst = []
    bias_card_lst = []

    bias_items = list(filter(lambda item: item['object'].lower()=='bias',
                             logtable))
    # get the number of bias images
    n_bias = len(bias_items)

    if n_bias == 0:
        # there is no bias frames
        return None, []


    fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^19s}'
    head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                'obsdate')

    for iframe, logitem in enumerate(bias_items):

        # now filter the bias frames
        fname = '{}.fits'.format(logitem['fileid'])
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)
        data, card_lst = correct_overscan(data, head, readout_mode)

        # pack the data and fileid list
        bias_data_lst.append(data)

        # append the file information
        prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(iframe+1)
        card = (prefix+' FILEID', logitem['fileid'])
        bias_card_lst.append(card)

        # append the overscan information of each bias frame to
        # bias_card_lst
        for keyword, value in card_lst:
            mobj = re.match('^HIERARCH GAMSE (OVERSCAN[\s\S]*)', keyword)
            if mobj:
                newkey = prefix + ' ' + mobj.group(1)
                bias_card_lst.append((newkey, value))

        # print info
        if iframe == 0:
            print('* Combine Bias Images: "{}"'.format(bias_file))
            print(head_str)
        message = fmt_str.format(
                    '[{:d}]'.format(logitem['frameid']),
                    logitem['fileid'], logitem['object'],
                    logitem['exptime'], logitem['obsdate'],
                    )
        print(message)

    prefix = 'HIERARCH GAMSE BIAS '
    bias_card_lst.append((prefix + 'NFILE', n_bias))

    # combine bias images
    bias_data_lst = np.array(bias_data_lst)

    combine_mode = 'mean'
    cosmic_clip  = section.getfloat('cosmic_clip')
    maxiter      = section.getint('maxiter')
    maskmode    = (None, 'max')[n_bias>=3]

    bias_combine = combine_images(bias_data_lst,
            mode        = combine_mode,
            upper_clip  = cosmic_clip,
            maxiter     = maxiter,
            maskmode    = maskmode,
            ncores      = ncores,
            )

    bias_card_lst.append((prefix+'COMBINE_MODE', combine_mode))
    bias_card_lst.append((prefix+'COSMIC_CLIP',  cosmic_clip))
    bias_card_lst.append((prefix+'MAXITER',      maxiter))
    bias_card_lst.append((prefix+'MASK_MODE',    str(maskmode)))

    # create the hdu list to be saved
    hdu_lst = fits.HDUList()
    # create new FITS Header for bias
    head = fits.Header()
    for card in bias_card_lst:
        head.append(card)
    head['HIERARCH GAMSE FILECONTENT 0'] = 'BIAS COMBINED'
    hdu_lst.append(fits.PrimaryHDU(data=bias_combine, header=head))

    ############## bias smooth ##################
    if section.getboolean('smooth'):
        # bias needs to be smoothed
        smooth_method = section.get('smooth_method')

        ny, nx = bias_combine.shape
        newcard_lst = []
        if smooth_method in ['gauss', 'gaussian']:
            # perform 2D gaussian smoothing
            smooth_sigma = section.getint('smooth_sigma')
            smooth_mode  = section.get('smooth_mode')
            bias_smooth = np.zeros_like(bias_combine, dtype=np.float64)
            bias_smooth[0:ny//2, :] = gaussian_filter(
                                        bias_combine[0:ny//2, :],
                                        sigma = smooth_sigma,
                                        mode  = smooth_mode)
            bias_smooth[ny//2:ny, :] = gaussian_filter(
                                        bias_combine[ny//2:ny, :],
                                        sigma = smooth_sigma,
                                        mode  = smooth_mode)

            # write information to FITS header
            newcard_lst.append((prefix+'SMOOTH CORRECTED',  True))
            newcard_lst.append((prefix+'SMOOTH METHOD', 'GAUSSIAN'))
            newcard_lst.append((prefix+'SMOOTH SIGMA',  smooth_sigma))
            newcard_lst.append((prefix+'SMOOTH MODE',   smooth_mode))
        else:
            print('Unknown smooth method: ', smooth_method)
            pass

        # pack the cards to bias_card_lst and also hdu_lst
        for card in newcard_lst:
            hdu_lst[0].header.append(card)
            bias_card_lst.append(card)
        hdu_lst.append(fits.ImageHDU(data=bias_smooth))
        card = ('HIERARCH GAMSE FILECONTENT 1', 'BIAS SMOOTHED')
        hdu_lst[0].header.append(card)

        # bias is the result array to return
        bias = bias_smooth
    else:
        # bias not smoothed
        card = (prefix+'SMOOTH CORRECTED', False)
        bias_card_lst.append(card)
        hdu_lst[0].header.append(card)

        # bias is the result array to return
        bias = bias_combine

    ############### save to FITS ##############
    hdu_lst.writeto(bias_file, overwrite=True)

    message = 'Bias image written to "{}"'.format(bias_file)
    logger.info(message)
    print(message)

    return bias, bias_card_lst

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
        ny, nx = shape

        mask[349:352, 627:630] = True
        mask[349:ny//2, 628]   = True

        mask[1604:ny//2, 2452] = True

        mask[280:284,3701]    = True
        mask[274:ny//2, 3702] = True
        mask[272:ny//2, 3703] = True
        mask[274:282, 3704]   = True

        mask[1720:1722, 3532:3535] = True
        mask[1720, 3535]           = True
        mask[1722, 3532]           = True
        mask[1720:ny//2,3533]      = True

        mask[347:349, 4082:4084] = True
        mask[347:ny//2,4083]     = True

        mask[ny//2:2631, 1909] = True
    else:
        print('No bad pixel information for this CCD size.')
        raise ValueError
    return mask

def get_mask(data, header):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        header (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.

    The shape of output mask is determined by the keywords in the input FITS
    header. The numbers of columns and rows are given by::
     
        N (columns) = head['NAXIS1'] - head['COVER']

        N (rows)    = head['NAXIS2'] - head['ROVER']

    where *head* is the input FITS header. 

    """

    saturation_adu = 65535

    x1, y1, xbin, ybin, cover, rover = get_ccd_geometry(header)
    sci_y1, sci_y2, sci_x1, sci_x2 = get_sci_region(header)
    newshape = (sci_y2-sci_y1, sci_x2-sci_x1)

    # find the saturation mask
    mask_sat = (data[sci_y1:sci_y2, sci_x1:sci_x2] >= saturation_adu)

    # get bad pixel mask
    mask_bad = get_badpixel_mask(newshape, bins=(ybin, xbin))

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

def fix_cr(data):
    """Cosmic ray fixing function.

    Args:
        data (:class:`numpy.ndarray`): Input image data.

    Returns:
        :class:`numpy.dtype`: Fixed image data.
    """
    m = data.mean(dtype=np.float64)
    s = data.std(dtype=np.float64)
    _mask = data > m + 3.*s
    if _mask.sum()>0:
        x = np.arange(data.size)
        f = InterpolatedUnivariateSpline(x[~_mask], data[~_mask], k=3)
        return f(x)
    else:
        return data

def correct_overscan(data, header, readout_mode=None):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        header (:class:`astropy.io.fits.Header`): Input FITS header.
        readout_mode (str): Readout mode of the CCD.
    
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
    """
    region_lst = get_region_lst(header, readout_mode)

    y1, y2, x1, x2 = get_sci_region(header)

    newdata = np.zeros((y2-y1, x2-x1), dtype=np.float64)

    # prepare card list to be returned
    card_lst = []
    prefix = 'HIERARCH GAMSE OVERSCAN '

    for iregion, (sci_region, ovr_region) in enumerate(region_lst):

        sci_y1, sci_y2, sci_x1, sci_x2 = sci_region
        ovr_y1, ovr_y2, ovr_x1, ovr_x2 = ovr_region

        scidata = data[sci_y1:sci_y2, sci_x1:sci_x2]
        ovrdata = data[ovr_y1:ovr_y2, ovr_x1+2:ovr_x2]

        # find the overscan level along the y-axis
        ovr_lst = ovrdata.mean(axis=1)
        # apply the sav-gol fitler to the mean of overscan
        winlen = 301
        order = 3
        upper_clip = 3.0
        ovr_smooth, _, _, _ = iterative_savgol_filter(ovr_lst,
                        winlen=winlen, order=order, upper_clip=upper_clip)

        # expand the 1d overscan values to 2D image that fits the sci region
        nysci = scidata.shape[1]
        ovrimg = np.repeat(ovr_smooth, nysci).reshape(-1, nysci)
        cordata = scidata - ovrimg

        new_y1 = sci_y1 - y1
        new_y2 = sci_y2 - y1
        new_x1 = sci_x1 - x1
        new_x2 = sci_x2 - x1
        
        newdata[new_y1:new_y2, new_x1:new_x2] = cordata

        prefix2 = prefix + 'REGION {} '.format(iregion)
        card_lst.append((prefix2+'SCI AXIS-1', '{}:{}'.format(sci_x1+1, sci_x2)))
        card_lst.append((prefix2+'SCI AXIS-2', '{}:{}'.format(sci_y1+1, sci_y2)))
        card_lst.append((prefix2+'OVR AXIS-1', '{}:{}'.format(ovr_x1+3, ovr_x2)))
        card_lst.append((prefix2+'OVR AXIS-2', '{}:{}'.format(ovr_y1+1, ovr_y2)))
        card_lst.append((prefix2+'COR AXIS-1', '{}:{}'.format(new_x1+1, new_x2)))
        card_lst.append((prefix2+'COR AXIS-2', '{}:{}'.format(new_y1+1, new_y2)))
        card_lst.append((prefix2+'METHOD',     'iterative_savgol'))
        card_lst.append((prefix2+'WINLEN',      winlen))
        card_lst.append((prefix2+'ORDER',       order))
        card_lst.append((prefix2+'UPPERCLIP',   upper_clip))
        card_lst.append((prefix2+'LOWERCLIP',   'None'))
        card_lst.append((prefix2+'OVERMIN',     ovr_smooth.min()))
        card_lst.append((prefix2+'OVERMAX',     ovr_smooth.max()))
        card_lst.append((prefix2+'OVERMEAN',    ovr_smooth.mean()))

    card_lst.append((prefix + 'CORRECTED', True))

    return newdata, card_lst

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

    # after 1st Dec 2018, echelle format and keywords in FITS header changed
    time_node1 = datetime.datetime(2018, 12, 1)

    if input_date > time_node1:
        mask = [dateutil.parser.parse(t) > time_node1
                for t in calibtable['obsdate']]
    else:                         
        mask = [dateutil.parser.parse(t) < time_node1
                for t in calibtable['obsdate']]
    
    # select the latest ThAr (deprecated)
    #row = calibtable[mask][-1]

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable[mask]['obsdate']]
    irow = np.abs(timediff).argmin()
    row = calibtable[mask][irow]
    fileid = row['fileid']  # selected fileid
    md5    = row['md5']

    message = 'Select {} from database index as ThAr reference'.format(fileid)
    logger.info(message)

    filepath = os.path.join('xinglong216hrs', 'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    calib = get_calib_from_header(head)

    return spec, calib

def plot_time_offset(real_obsdate_lst, delta_t_lst, time_offset, figname):
    """Plot the offset between the real observing time and log time.

    Args:
        real_obsdate_lst (list): List of real observing time.
        delta_t_lst (list): List of time differences in second between real time
            and log time
        time_offset (float): Determined offset in second between real time and
            log time.
        figname (str): Filename of output figure.
    """
    
    fig = plt.figure(figsize=(9, 6), dpi=200)
    ax = fig.add_axes([0.12, 0.16, 0.83, 0.76])
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
    fig.suptitle('Time Offsets Between Log and FITS', fontsize=14)
    fig.savefig(figname)
    plt.close(fig)

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.43,0.86])
        self.ax2 = self.add_axes([0.52,0.50,0.43,0.40])
        self.ax3 = self.add_axes([0.52,0.10,0.43,0.40])
        self.ax4 = self.ax3.twinx()

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

class BackgroundFigure(BackgroundFigureCommon):
    """Figure to plot the background correction.
    """
    def __init__(self, data=None, background=None, dpi=300, figsize=(12, 5.5),
           title=None, figname=None, zscale=('log', 'linear'), contour=True):
        BackgroundFigureCommon.__init__(self, figsize=figsize, dpi=dpi)
        width = 0.36
        height = width*figsize[0]/figsize[1]
        self.ax1  = self.add_axes([0.06, 0.1, width, height])
        self.ax2  = self.add_axes([0.55, 0.1, width, height])
        self.ax1c = self.add_axes([0.06+width+0.01, 0.1, 0.015, height])
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
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X (pixel)')
            ax.set_ylabel('Y (pixel)')
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))

class BrightnessProfileFigure(Figure):
    """Figure to plot the background combinations.
    """
    def __init__(self,
            fiber_obs_bkg_lst = None,
            fiber_sel_bkg_lst = None,
            fiber_scale_lst   = None,
            title=None, filename=None, dpi=300, figsize=(8,6)):
        # create figure
        Figure.__init__(self, figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self)
        # plot profiles
        if None not in [fiber_obs_bkg_lst, fiber_sel_bkg_lst, fiber_scale_lst]:
            self.plot_profile(fiber_obs_bkg_lst,
                              fiber_sel_bkg_lst,
                              fiber_scale_lst)
        # set figure title
        if title is not None:
            self.suptitle(title)

        # save figure
        if filename is not None:
            self.savefig(filename)


    def plot_profile(self, fiber_obs_bkg_lst, fiber_sel_bkg_lst,
            fiber_scale_lst):
        """Plot the brightness profiles of observed and scaled brightness
        profiles

        Args:
            fiber_obs_bkg_lst (dict):
            fiber_sel_bkg_lst (dict):
            fiber_scale_lst (dict):
        """
        ax1 = self.add_axes([0.1,0.58,0.85,0.32])
        ax2 = self.add_axes([0.1,0.10,0.85,0.32])

        # check if the keys of input dicts are identical
        keys1 = sorted(fiber_obs_bkg_lst.keys())
        keys2 = sorted(fiber_sel_bkg_lst.keys())
        keys3 = sorted(fiber_scale_lst.keys())

        if keys1 != keys2 or keys1 != keys3:
            print('Warning: input keys are different:',keys1, keys2, keys3)
            raise ValueError

        # plot observed background profiles
        alpha = 0.7
        lw = 1.0
        ls = '-'
        for ifiber, fiber in enumerate(keys1):
            obs_bkg = fiber_obs_bkg_lst[fiber]
            color = 'C{:d}'.format(ifiber%10)
            ax1.plot(obs_bkg.aper_pos_lst, obs_bkg.aper_brt_lst,
                    label='Fiber {}'.format(fiber),
                    color=color, lw=lw, ls=ls, alpha=alpha,
                    )
            ax2.plot(obs_bkg.aper_ord_lst, obs_bkg.aper_brt_lst,
                    label='Fiber {}'.format(fiber),
                    color=color, lw=lw, ls=ls, alpha=alpha,
                    )

        # plot scaled background profiles
        ls = '--'
        for ifiber, fiber in enumerate(keys1):
            sel_bkg = fiber_sel_bkg_lst[fiber]
            scale   = fiber_scale_lst[fiber]
            color = 'C{:d}'.format(ifiber%10)
            ax1.plot(sel_bkg.aper_pos_lst, sel_bkg.aper_brt_lst*scale,
                    label=u'saved \xd7 {:4.2f}'.format(scale),
                    color=color, lw=lw, ls=ls, alpha=alpha,
                    )
            ax2.plot(sel_bkg.aper_ord_lst, sel_bkg.aper_brt_lst*scale,
                    label=u'saved \xd7 {:4.2f}'.format(scale),
                    color=color, lw=lw, ls=ls, alpha=alpha,
                    )

        for ax in self.get_axes():
            # set legends
            leg = ax.legend(loc='upper left')
            #leg.get_frame().set_alpha(0.1)
            ax.grid(True, ls='--')
            ax.set_axisbelow(True)

        # set xlim of ax1
        ny, nx = sel_bkg.data.shape
        ax1.set_xlim(0, ny-1)

        # set xlim of ax2
        ord1 = max(obs_bkg.aper_ord_lst)
        ord2 = min(obs_bkg.aper_ord_lst)
        ax2.set_xlim(ord1, ord2)

        # interpolate function converting wavelength (lambda) to order number
        idx = obs_bkg.aper_wav_lst.argsort()
        f = InterpolatedUnivariateSpline(
                    obs_bkg.aper_wav_lst[idx],
                    obs_bkg.aper_ord_lst[idx], k=3)
        # find the exponential part of wavelength span
        wavmin = min(obs_bkg.aper_wav_lst)
        wavmax = max(obs_bkg.aper_wav_lst)
        wavdiff = wavmax - wavmin
        exp = int(math.log10(wavdiff))
        # adjust the exponential part if too large
        if wavdiff/(10**exp)<=2:
            exp -= 1
        w = 10**int(math.log10(wavmin))
        # find the major ticks of the wavelength axis
        wticks = []
        wlabels = []
        while(w <= wavmax):
            if w >= wavmin:
                order = f(w)
                wticks.append(float(order))
                wlabels.append('{:g}'.format(w))
            w += 10**exp

        # plot a series of wavelength ticks in top
        ax22 = ax2.twiny()
        ax22.set_xticks(wticks)
        ax22.set_xticklabels(wlabels)
        ax22.set_xlim(ax2.get_xlim())
        ax22.set_xlabel(u'Wavelength (\xc5)')

        # others
        ax1.set_xlabel('Pixel')
        ax2.set_xlabel('Order')

    def close(self):
        plt.close(self)


class ProfileNormalizer(ProfileNormalizerCommon):
    def __init__(self, xdata, ydata, mask):
        self.xdata = xdata
        self.ydata = ydata
        self.mask  = mask

        sat_mask = (mask&4 > 0)
        bad_mask = (mask&2 > 0)

        # iterative fitting using fitfunc
        A0 = ydata.max() - ydata.min()
        c0 = (xdata[0] + xdata[-1])/2
        b0 = ydata.min()
        p0 = [A0, c0, 5.0, 4.0, b0]
        lower_bounds = [-np.inf, xdata[0],  0.5,    0.5,    -np.inf]
        upper_bounds = [np.inf,  xdata[-1], np.inf, 100., ydata.max()]
        _m = (~sat_mask)*(~bad_mask)

        for i in range(10):
            opt_result = opt.least_squares(self.errfunc, p0,
                        args=(xdata[_m], ydata[_m]),
                        bounds=(lower_bounds, upper_bounds))
            p1 = opt_result['x']
            residuals = self.errfunc(p1, xdata, ydata)
            std = residuals[_m].std(ddof=1)
            _new_m = (np.abs(residuals) < 3*std)*_m
            if _m.sum() == _new_m.sum():
                break
            _m = _new_m
            p0 = p1
    
        A, c, alpha, beta, bkg = p1
        self.x = xdata - c
        self.y = (ydata - bkg)/A
        self.m = _m
        
        self.param = p1
        self.std = std

    def is_succ(self):
        A, center, alpha, beta, bkg = self.param
        std = self.std

        if A>0 and A/std>10 and alpha<10 and beta<10 and \
            (bkg>0 or (bkg<0 and abs(bkg)<A/10)):
            return True
        else:
            return False

    def fitfunc(self, param, x):
        """Use Generalized Gaussian.
        """
        A, center, alpha, beta, bkg = param
        return A*np.exp(-np.power(np.abs(x-center)/alpha, beta)) + bkg


def norm_profile(xdata, ydata, mask):
    # define the fitting and error functions
    def gaussian_gen_bkg(A, center, alpha, beta, bkg, x):
        return A*np.exp(-np.power(np.abs(x-center)/alpha, beta)) + bkg
    def fitfunc(p, x):
        return gaussian_gen_bkg(p[0], p[1], p[2], p[3], p[4], x)
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # iterative fitting using gaussian + bkg function
    A0 = ydata.max()-ydata.min()
    c0 = (xdata[0]+xdata[-1])/2
    b0 = ydata.min()
    p0 = [A0, c0, 5.0, 4.0, b0]
    lower_bounds = [-np.inf, xdata[0],  0.5,    0.5,    -np.inf]
    upper_bounds = [np.inf,  xdata[-1], np.inf, np.inf, ydata.max()]
    _m = (~sat_mask)*(~bad_mask)

    for i in range(10):
        opt_result = opt.least_squares(errfunc, p0,
                    args=(xdata[_m], ydata[_m], fitfunc),
                    bounds=(lower_bounds, upper_bounds))
        p1 = opt_result['x']
        residuals = errfunc(p1, xdata, ydata, fitfunc)
        std = residuals[_m].std(ddof=1)
        _new_m = (np.abs(residuals) < 3*std)*_m
        if _m.sum() == _new_m.sum():
            break
        _m = _new_m
        p0 = p1

    A, c, alpha, beta, bkg = p1
    newx = xdata - c
    newy = ydata - bkg

    param = (A, c, alpha, beta, bkg, std)

    if A < 1e-3:
        return None
    return newx, newy/A, param


def norm_profile_gaussian(xdata, ydata, mask):
    # define the fitting and error functions
    def gaussian_bkg(A, center, fwhm, bkg, x):
        s = fwhm/2./math.sqrt(2*math.log(2))
        return A*np.exp(-(x-center)**2/2./s**2) + bkg
    def fitfunc(p, x):
        return gaussian_bkg(p[0], p[1], p[2], p[3], x)
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # iterative fitting using gaussian + bkg function
    p0 = [ydata.max()-ydata.min(), (xdata[0]+xdata[-1])/2., 3.0, ydata.min()]
    _m = (~sat_mask)*(~bad_mask)

    for i in range(10):
        p1, succ = opt.leastsq(errfunc, p0,
                    args=(xdata[_m], ydata[_m], fitfunc))
        res = errfunc(p1, xdata, ydata, fitfunc)
        std = res[_m].std(ddof=1)
        _new_m = (np.abs(res) < 3*std)*_m
        if _m.sum() == _new_m.sum():
            break
        _m = _new_m

    A, c, fwhm, bkg = p1
    newx = xdata - c
    newy = ydata - bkg

    param = (A, c, fwhm, bkg)

    if A < 1e-3:
        return None
    return newx, newy/A, param

class SpatialProfileFigure(Figure):
    """Figure to plot the cross-dispersion profiles.

    """
    def __init__(self,
            nrow = 3,
            ncol = 3,
            figsize = (12,8),
            dpi = 200,
            ):

        # create figure
        Figure.__init__(self, figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self)

        # add axes
        _w = 0.27
        _h = 0.26
        for irow in range(nrow):
            for icol in range(ncol):
                _x = 0.08 + icol*0.31
                _y = 0.06 + (nrow-1-irow)*0.30

                ax = self.add_axes([_x, _y, _w, _h])

    def close(self):
        plt.close(self)

def get_interorder_background(data, mask=None, apertureset=None, **kwargs):
    figname = kwargs.pop('figname', 'bkg_{:04d}.png')
    distance = kwargs.pop('distance', 7)

    if mask is None:
        mask = np.zeros_like(data, dtype=np.int32)

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    bkg_image = np.zeros_like(data, dtype=np.float32)
    plot_x = [10, 509, 2505]

    saturated_cols = [610, 3422, 3595]
    masked_x = []
    for col in saturated_cols:
        for x in np.arange(col-4, col+4):
            masked_x.append(x)

    for x in allx:
        if x in masked_x:
            continue

        if x in plot_x:
            plot = True
            fig1 = plt.figure(figsize=(12,8))
            ax01 = fig1.add_subplot(211)
            ax02 = fig1.add_subplot(212)
        else:
            plot = False
        mask_rows = np.zeros_like(ally, dtype=bool)
        for aper, aperloc in sorted(apertureset.items()):
            ycen = aperloc.position(x)
            if plot:
                ax01.axvline(x=ycen, color='C0', ls='--', lw=0.5, alpha=0.4)
                ax02.axvline(x=ycen, color='C0', ls='--', lw=0.5, alpha=0.4)
    
            imask = np.abs(ally - ycen) < distance
            mask_rows += imask
        if plot:
            ax01.plot(ally, data[:, x], color='C0', alpha=0.3, lw=0.7)
        x_lst, y_lst = [], []
        for (y1, y2) in get_edge_bin(~mask_rows):
            if plot:
                ax01.plot(ally[y1:y2], data[y1:y2,x],
                            color='C0', alpha=1, lw=0.7)
                ax02.plot(ally[y1:y2], data[y1:y2,x],
                            color='C0', alpha=1, lw=0.7)
            if y2 - y1 > 1:
                yflux = data[y1:y2, x]
                ymask = mask[y1:y2, x]
                xlist = np.arange(y1, y2)
    
                # block the highest point and calculate mean
                _m = xlist == y1 + np.argmax(yflux)
                mean = yflux[~_m].mean()
                std  = yflux[~_m].std()
                if yflux.max() < mean + 3.*std:
                    meany = yflux.mean()
                    meanx = (y1+y2-1)/2
                else:
                    meanx = xlist[~_m].mean()
                    meany = mean
            else:
                meany = data[y1,x]
                meanx = y1
            x_lst.append(meanx)
            y_lst.append(meany)
        x_lst = np.array(x_lst)
        y_lst = np.array(y_lst)
        y_lst = np.maximum(y_lst, 0)
        y_lst = sg.medfilt(y_lst, 3)
        f = intp.InterpolatedUnivariateSpline(x_lst, y_lst, k=3, ext=3)
        bkg = f(ally)
        bkg_image[:, x] = bkg
        if plot:
            ax01.plot(x_lst, y_lst, 'o', color='C3', ms=3)
            ax02.plot(x_lst, y_lst, 'o', color='C3', ms=3)
            ax01.plot(ally, bkg, ls='-', color='C3', lw=0.7, alpha=1)
            ax02.plot(ally, bkg, ls='-', color='C3', lw=0.7, alpha=1)
            _y1, _y2 = ax02.get_ylim()
            ax02.plot(ally, data[:, x], color='C0', alpha=0.3, lw=0.7)
            ax02.set_ylim(_y1, _y2)
            ax01.set_xlim(0, ny-1)
            ax02.set_xlim(0, ny-1)
            fig1.savefig(figname.format(x))
            plt.close(fig1)

    for y in np.arange(ally):
        m = 0
        func = intp.InterpolateUnivariateSpline(allx, bkg_img[y, :], k=3)
        bkg_img = 0
    return bkg_image
