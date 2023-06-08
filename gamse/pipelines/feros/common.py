import os
import re
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ...echelle.trace import TraceFigureCommon
from ...echelle.background import BackgroundFigureCommon

def get_ccd_geometry(header):
    """Get basic geometry of CCD.

    Args:
        header (:class:`astropy.io.fits.header`): FITS header.

    Returns:
        tuple: A tuple of **(x1, x2, xbin, ybin)**, where
        **(0, x1)** is the columns of the prescan region,
        **(x1, x2)** is the columns of the science region,
        **(x2, )** is the columns of the overscan region,
        and **(xbin, ybin)** are the CCD binning along X and Y axes.
    """
    cdelt1 = int(round(header['CDELT1']))
    cdelt2 = int(round(header['CDELT2']))

    # check nx
    if header['ESO DET CHIP1 NX'] == header['ESO DET OUT1 NX']:
        nx = header['ESO DET OUT1 NX']
    else:
        raise ValueError

    # check ny
    if header['ESO DET CHIP1 NY'] == header['ESO DET OUT1 NY']:
        ny = header['ESO DET OUT1 NY']
    else:
        raise ValueError

    prex = header['HIERARCH ESO DET OUT1 PRSCX']
    prey = header['HIERARCH ESO DET OUT1 PRSCY']
    ovrx = header['HIERARCH ESO DET OUT1 OVSCX']
    ovry = header['HIERARCH ESO DET OUT1 OVSCY']

    if prey != 0 or ovry != 0:
        raise ValueError

    if prex + nx + ovrx != header['NAXIS1']:
        raise ValueError

    if prey + ny + ovry != header['NAXIS2']:
        raise ValueError

    if cdelt1*nx != 2048:
        raise ValueError
    if cdelt2*ny != 4096:
        raise ValueError

    x1 = prex
    x2 = prex + nx
    binx = cdelt1
    biny = cdelt2

    return x1, x2, binx, biny

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(18,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.32,0.86])
        self.ax2 = self.add_axes([0.40,0.50,0.55,0.40])
        self.ax3 = self.add_axes([0.40,0.10,0.55,0.40])
        self.ax4 = self.ax3.twinx()


class BackgroundFigure(BackgroundFigureCommon):
    """Figure to plot the background correction.
    """
    def __init__(self, dpi=300, figsize=(12, 5.5)):
        BackgroundFigureCommon.__init__(self, figsize=figsize, dpi=dpi)
        width = 0.36
        height = width*figsize[0]/figsize[1]
        self.ax1  = self.add_axes([0.06, 0.1, width, height])
        self.ax2  = self.add_axes([0.55, 0.1, width, height])
        self.ax1c = self.add_axes([0.06+width+0.01, 0.1, 0.015, height])
        self.ax2c = self.add_axes([0.55+width+0.01, 0.1, 0.015, height])

    def plot(self, data, background, scale=(5, 99)):
        """Plot the image data with background and the subtracted background
        light.

        Args:
            data (:class:`numpy.ndarray`): Image data to be background
                subtracted.
            background (:class:`numpy.ndarray`): Background light as a 2D array.
        """
        # find the minimum and maximum value of plotting
        vmin = np.percentile(data, scale[0])
        vmax = np.percentile(data, scale[1])

        cax1 = self.ax1.imshow(data, cmap='gray', vmin=vmin, vmax=vmax,
                origin='lower')
        cax2 = self.ax2.imshow(background, cmap='viridis',
                origin='lower')
        cs = self.ax2.contour(background, colors='r', linewidths=0.5)
        self.ax2.clabel(cs, inline=1, fontsize=7, use_clabeltext=True)
        self.colorbar(cax1, cax=self.ax1c)
        self.colorbar(cax2, cax=self.ax2c)
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X (pixel)')
            ax.set_ylabel('Y (pixel)')
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))

def print_wrapper(string, item):
    """A wrapper for log printing for FEROS pipeline.

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

def correct_overscan(data, header):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        header (:class:`astropy.io.fits.Header`): Input FITS header.
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
    """

    ny, nx = data.shape

    x1, x2, binx, biny = get_ccd_geometry(header)

    prescandata = data[:, 0:x1-2]
    ovrscandata = data[:, x2+2:]
    scidata = data[:, x1:x2]

    premean = prescandata.mean(axis=1)
    ovrmean = ovrscandata.mean(axis=1)
    scanmean = (premean + ovrmean)/2

    # expand the 1d overscan values to 2D image that fits the sci region
    nysci = scidata.shape[1]
    scandata = np.repeat(scanmean, nysci).reshape(-1, nysci)

    cordata = scidata - scandata

    card_lst = []

    card_lst.append(('HIERARCH GAMSE OVERSCAN CORRECTED', True))

    return cordata, card_lst

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
    rawpath = config['data']['rawpath']

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

    def filter_bias(item):
        return item['object'].lower()=='bias' and item['binning']=='(1, 1)'

    bias_items = list(filter(filter_bias, logtable))

    # get the number of bias images
    n_bias = len(bias_items)

    if n_bias == 0:
        # there is no bias frames
        return None, []

    fmt_str = '  - {:>7s} {:^23} {:^8s} {:^7} {:^7} {:^5}'
    head_str = fmt_str.format('frameid', 'FileID', 'Object', 'binning',
                            'exptime', 'q95')

    for iframe, logitem in enumerate(bias_items):
        # now filter the bias frames
        fname = 'FEROS.{}.fits'.format(logitem['fileid'])
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)
        data, card_lst = correct_overscan(data, head)

        # pack the data and fileid list
        bias_data_lst.append(data)

        # append the file information
        prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(iframe+1)
        card = (prefix+' FILEID', logitem['fileid'])
        bias_card_lst.append(card)

        # print info
        if iframe == 0:
            print('* Combine Bias Images: "{}"'.format(bias_file))
            print(head_str)
        message = fmt_str.format(
                    '[{:d}]'.format(logitem['frameid']),
                    logitem['fileid'], logitem['object'], logitem['binning'],
                    logitem['exptime'], logitem['q95'],
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

    bias_combine = fix_badpixels(bias_combine, mask&2==2)

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
    head['HIERARCH GAMSE FILECONTENT 1'] = 'BIAS MASK'
    hdu_lst.append(fits.PrimaryHDU(data=bias_combine, header=head))
    hdu_lst.append(fits.ImageHDU(data=mask, header=head))

    ############## bias smooth ##################
    if section.getboolean('smooth'):
        # bias needs to be smoothed
        smooth_method = section.get('smooth_method')

        newcard_lst = []
        if smooth_method in ['gauss', 'gaussian']:
            # perform 2D gaussian smoothing
            smooth_sigma = section.getint('smooth_sigma')
            smooth_mode  = section.get('smooth_mode')
            bias_smooth = gaussian_filter(bias_combine,
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
        card = ('HIERARCH GAMSE FILECONTENT 2', 'BIAS SMOOTHED')
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

    hdu_lst.writeto(bias_file, overwrite=True)

    message = 'Bias image written to "{}"'.format(bias_file)
    logger.info(message)
    print(message)

    return bias, bias_card_lst

def get_mask(data, header):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        header (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.
    """
    saturation_adu = 65535

    x1, x2, binx, biny = get_ccd_geometry(header)

    # find the saturation mask
    mask_sat = (data[:, x1:x2] >= saturation_adu)

    # get bad pixel mask
    mask_bad = get_badpixel_mask(binx, biny)

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

def get_badpixel_mask(binx, biny):
    """Get the mask of bad pixels and columns.

    Args:
        binx (int): CCD binning along X axes.
        biny (int): CCD binning along Y axes.

    Returns:
        :class:`numpy.ndarray`: 2D binary mask, where bad pixels are marked with
        *True*, others *False*.

    The bad pixels are found *empirically*.
        
    """
    shape = (4096//biny, 2048//binx)
    mask = np.zeros(shape, dtype=bool)

    if (binx, biny) == (1, 1):
        mask[:,     320] = True
        mask[:,     326] = True
        mask[604:, 1349] = True
        '''
        mask[1676:,     219] = True
        mask[1678:,     222] = True
        mask[1675:,     223] = True
        mask[130,       315] = True
        mask[136:160,   316] = True
        mask[127,       320] = True
        mask[128:140,   321] = True
        mask[193:240,   321] = True
        mask[127:170,   322] = True
        mask[189:240,   322] = True
        mask[131:175,   323] = True
        mask[189:220,   323] = True
        mask[136:,      326] = True
        mask[137:180,   327] = True
        mask[139:150,   329] = True
        mask[125,       330] = True
        mask[1614,      332] = True
        mask[125:150,   333] = True
        mask[1616:,     334] = True
        mask[1616:,     335] = True
        mask[1616:,     336] = True
        mask[1617:,     337] = True
        mask[1618:,     338] = True
        mask[1619,      340] = True
        mask[1621:1660, 342] = True
        mask[1621:1650, 343] = True
        mask[535:580,   638] = True
        mask[868:,      645] = True
        mask[868:,      646] = True
        mask[1513:1613, 842] = True
        mask[1513:1613, 843] = True
        mask[1500:1520, 857] = True
        mask[1501:1530, 858] = True
        mask[1476:1490, 883] = True
        mask[1459:1460, 899] = True
        mask[1459:,     900] = True
        mask[1462:1463, 901] = True
        mask[1465:1480, 903] = True
        mask[1453:,     916] = True
        mask[1415:1435, 959] = True
        mask[1417:,     960] = True
        mask[1420:1450, 961] = True
        mask[1423:1453, 962] = True
        mask[1425:,     963] = True
        mask[1428:,     964] = True
        mask[1431:1440, 965] = True
        mask[1431:1435, 966] = True
        mask[1436:,     967] = True
        mask[1449:1454, 972] = True
        mask[1400:,    1061] = True
        mask[1403:,    1062] = True
        mask[1404:,    1063] = True
        mask[1405:,    1064] = True
        mask[1409:,    1065] = True
        mask[1409:,    1066] = True
        mask[1413,     1067] = True
        mask[1416,     1069] = True
        mask[1425,     1073] = True
        mask[607:,     1296] = True
        mask[607:,     1297] = True
        mask[607:,     1298] = True
        mask[:,        1299] = True
        mask[1529:,    1701] = True
        mask[1232:1270,1729] = True
        mask[88:130,   1829] = True
        '''

    else:
        print('No bad pixel information for this CCD size.')
        raise ValueError
    return mask

def fix_badpixels(data, mask):
    """Fix bad pixels of FEROS CCD.

    Args:
        data ():
        mask ():

    Returns:
        :class:`numpy.dtype`: 
    """

    fixdata = data.copy()

    ny, nx = data.shape
    allx = np.arange(nx)
    for y in np.arange(ny):
        m = mask[y,:]
        if m.sum()==0:
            continue
        f = InterpolatedUnivariateSpline(allx[~m], data[y,~m], k=3)
        fixdata[y,m] = f(allx[m])

    return fixdata
