import os
import re
import math
import shutil
import datetime
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates  as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ...echelle.imageproc import combine_images
from ...echelle.trace import TraceFigureCommon
from ...echelle.flat import ProfileNormalizerCommon
from ...echelle.background import BackgroundFigureCommon
from ...echelle.wlcalib import get_calib_from_header
from ...utils.download import get_file

def correct_overscan(data, mask=None, direction=None):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        mask (:class:`numpy.ndarray`): Input image mask.
        direction (str): CCD direction code.
    
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
            * **overmean** (*float*) – Mean value of overscan pixels.
            * **overstd** (*float*) – Standard deviation of overscan pixels.
    """
    ny, nx = data.shape
    overdata1 = data[:, 0:20]
    overdata2 = data[:, nx-18:nx]
    overdata_tot = np.hstack((overdata1, overdata2))

    # find the overscan level along the y axis
    # 1 for the left region, 2 for the right region
    # calculate the mean of overscan region along x direction
    ovr_lst1 = overdata1.mean(dtype=np.float64, axis=1)
    ovr_lst2 = overdata2.mean(dtype=np.float64, axis=1)
    
    # only used the bluer ~1/2 regions for calculating mean of overscan
    if direction[1]=='b':
        vy1, vy2 = 0, ny//2
    elif direction[1]=='r':
        vy1, vy2 = ny//2, ny
    else:
        print('Unknown direction:', direction)
        raise ValueError

    # find the mean and standard deviation for left & right overscan
    '''
    ovrmean1 = ovr_lst1[vy1:vy2].mean(dtype=np.float64)
    ovrmean2 = ovr_lst2[vy1:vy2].mean(dtype=np.float64)
    ovrstd1  = ovr_lst1[vy1:vy2].std(dtype=np.float64, ddof=1)
    ovrstd2  = ovr_lst2[vy1:vy2].std(dtype=np.float64, ddof=1)
    '''

    ovrmean1 = data[vy1:vy2, 0:20].mean(dtype=np.float64)
    ovrstd1  = data[vy1:vy2, 0:20].std(dtype=np.float64)
    # subtract overscan
    new_data = data[:, 20:2068] - ovrmean1
    
    card_lst = []
    prefix = 'HIERARCH GAMSE OVERSCAN '
    card_lst.append((prefix + 'CORRECTED', True))
    card_lst.append((prefix + 'METHOD',    'mean'))
    card_lst.append((prefix + 'AXIS1',     '1:20'))
    card_lst.append((prefix + 'AXIS2',     '{}:{}'.format(vy1+1,vy2)))
    card_lst.append((prefix + 'MEAN',      ovrmean1))
    card_lst.append((prefix + 'STDEV',     ovrstd1))

    return new_data, card_lst, ovrmean1, ovrstd1

def get_bias(config, logtable):
    """Get bias image.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) --- Output bias image.
            * **bias_card_lst** (list) --- List of FITS header cards related to
              the bias correction.
            * **n_bias** (int) --- Number of bias images.
            * **ovrstd** (float) --- Standard deviation of overscan of bias.
            * **ron** (float) --- Readout error caused by bias.

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

        n_bias = head['GAMSE BIAS NFILE']
        ovrstd = head['GAMSE BIAS OVERSTD']
        ron    = head['GAMSE BIAS RON_ERROR']

        # print info
        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
        return bias, bias_card_lst, n_bias, ovrstd, ron
    else:
        return combine_bias(config, logtable)


def combine_bias(config, logtable):
    """Combine bias images.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) --- Output bias image.
            * **bias_card_lst** (list) --- List of FITS header cards related to
              the bias correction.
            * **n_bias** (int) --- Number of bias images.
            * **ovrstd** (float) --- Standard deviation of overscan of bias.
            * **ron** (float) --- Readout error caused by bias.
    
    Combine bias frames found in observing log.
    The resulting array **bias** is combined using sigma-clipping method with
    an uppper clipping value given by "cosmic_clip" in "reduce.bias" section in
    **config**.
    Meanwhile, a card list containing the method, mean value and standard
    deviation to be added to the FITS header is also returned.

    """
    rawdata   = config['data']['rawdata']
    direction = config['data']['direction']

    # determine number of cores to be used
    ncores = config['reduce'].get('ncores')
    if ncores == 'max':
        ncores = os.cpu_count()
    else:
        ncores = min(os.cpu_count(), int(ncores))

    section = config['reduce.bias']
    bias_file = section['bias_file']

    # read each individual CCD
    bias_data_lst = []
    bias_card_lst = []
    bias_overstd_lst = []

    bias_items = list(filter(lambda item: item['object'].lower()=='bias',
                             logtable))
    # get the number of bias images
    n_bias = len(bias_items)

    if n_bias == 0:
        # there is no bias frames
        return None, [], 0, 0.0, 0.0

    print('* Combine Bias Images: "{}"'.format(bias_file))
    fmt_str = '    - {:>5s} {:18s} {:10s} {:7} {:23s} {:6} {:5} {:7}'
    head_str= fmt_str.format('FID', 'fileid', 'object', 'exptime',
                    'obsdate', 'nsat', 'q95', 'overmean')
    print(head_str)

    for ifile, logitem in enumerate(bias_items):

        # now filter the bias frames
        filename = os.path.join(rawdata, logitem['fileid']+'.fits')
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)
        data, card_lst, overmean, overstd = correct_overscan(
                                                data, mask, direction)
        bias_overstd_lst.append(overstd)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']

        # pack the data and fileid list
        bias_data_lst.append(data)

        # append the file information
        prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ifile+1)
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
        string = fmt_str.format('[{:d}]'.format(logitem['frameid']),
                    logitem['fileid'], logitem['object'],
                    '{:<5g}'.format(logitem['exptime']),
                    str(logitem['obsdate']),
                    logitem['nsat'], logitem['q95'],
                    '{:<7.2f}'.format(overmean))
        print(string)

    prefix = 'HIERARCH GAMSE BIAS '
    bias_card_lst.append((prefix + 'NFILE', n_bias))

    # combine bias images
    bias_data_lst = np.array(bias_data_lst)

    combine_mode = 'mean'
    cosmic_clip  = section.getfloat('cosmic_clip')
    maxiter      = section.getint('maxiter')
    maskmode     = (None, 'max')[n_bias>=3]

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
    # pack new card list into header and bias_card_lst
    for card in bias_card_lst:
        head.append(card)
    head['HIERARCH GAMSE FILECONTENT 0'] = 'BIAS COMBINED'
    hdu_lst.append(fits.PrimaryHDU(data=bias_combine, header=head))

    # plot the bias
    bias_fig = BiasFigure(data=bias_combine, title='Bias Mean')
    # save and close the figure
    figpath = config['reduce'].get('figpath', None)
    if figpath is None:
        figpath = config['reduce'].get('report') # old style
    figfilename = os.path.join(figpath, 'bias_mean.png')
    bias_fig.savefig(figfilename)
    plt.close(bias_fig)

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
                            sigma=smooth_sigma, mode=smooth_mode)

            # factor of readout noise caused by bias subtraction
            ron_factor = 2*math.pi*smooth_sigma**2

            # write information to FITS header
            newcard_lst.append((prefix+'SMOOTH CORRECTED', True))
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

        # plot the smoothed bias
        bias_fig = BiasFigure(data=bias_smooth, title='Bias Smoothed')
        # save and close the figure
        figpath = config['reduce'].get('figpath', None)
        if figpath is None:
            figpath = config['reduce'].get('report')    # old style
        figfilename = os.path.join(figpath, 'bias_smooth.png')
        bias_fig.savefig(figfilename)
        plt.close(bias_fig)

        # bias is the result array to return
        bias = bias_smooth
    else:
        # bias not smoothed
        card = (prefix+'SMOOTH CORRECTED', False)
        bias_card_lst.append(card)
        hdu_lst[0].header.append(card)

        # bias is the result array to return
        bias = bias_combine

        # factor of readout noise caused by bias subtraction
        ron_factor = 1

    # calculate average overstd values for all bias frames
    bias_overstd = np.mean(bias_overstd_lst)
    # put this parameter to the header
    card = (prefix + 'OVERSTD', bias_overstd)
    bias_card_lst.append(card)
    hdu_lst[0].header.append(card)

    # calculate the readout noise caused by bias subtraction
    ron = bias_overstd/math.sqrt(n_bias*ron_factor)
    # put this parameter to the header
    card = (prefix + 'RON_ERROR', ron)
    bias_card_lst.append(card)
    hdu_lst[0].header.append(card)

    # write to FITS file
    hdu_lst.writeto(bias_file, overwrite=True)

    message = 'Bias image written to "{}"'.format(bias_file)
    logger.info(message)
    print(message)

    return bias, bias_card_lst, n_bias, bias_overstd, ron


def get_mask(data):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.

    Returns:
        :class:`numpy.ndarray`: Image mask.
    """
    # saturated CCD count
    saturation_adu = 63000

    mask_sat = (data[:, 20:-20] >= saturation_adu)

    mask_bad = np.zeros_like(data[:, 20:-20], dtype=np.int16)
    # currently no bad pixels in FOCES CCD

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

def print_wrapper(string, item):
    """A wrapper for log printing for FOCES pipeline.

    Args:
        string (str): The string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype    = item['imgtype']
    objectname = item['object'].strip().lower()

    if imgtype=='cal' and objectname=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='cal':
        if objectname == 'thar':
            # arc lamp, use light yellow (93)
            return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
        else:
            return string
        #elif (item['fiber_A'], item['fiber_B']) in [('ThAr', ''),
        #                                            ('', 'ThAr'),
        #                                            ('ThAr', 'ThAr')]:
        #    # arc lamp, use light yellow (93)
        #    return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
        #else:
        #    return string
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

    filepath = os.path.join('instruments/foces',
                'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    calib = get_calib_from_header(head)

    return spec, calib

def get_primary_header(input_lst):
    """Return a list of header records with length of 80 characters.
    The order and datatypes of the records follow the FOCES FITS standards.

    Args:
        input_lst (tuple): A tuple containing the keywords and their values

    Returns:
        *list*: A list containing the records

    """
    lst = [
        # 12345678    12345678901234567890123456789012345678901234567
        ('SIMPLE'  , 'file does conform to FITS standard'             ),
        ('BITPIX'  , 'number of bits per data pixel'                  ),
        ('NAXIS'   , 'number of data axes'                            ),
        ('NAXIS1'  , 'length of data axis 1'                          ),
        ('NAXIS2'  , 'length of data axis 2'                          ),
        ('BSCALE'  , 'factor to linearly scale the data pixel values' ),
        ('BZERO'   , 'offset to linearly scale the data pixel values' ),
        ('BUNIT'   , 'physical unit of the data pixel values'         ),
        ('BLANK'   , 'value representing undefined physical values'   ),
        ('DISPAXIS', 'main dispersion axis of the spectral data'      ),
        ('DATATYPE', 'type of data (calibration/science)'             ),
        ('OBJECT'  , 'object observed'                                ),
        ('DATE-OBS', 'start date of observation run'                  ),
        ('MJD-OBS' , 'Modified Julian Date of observation run'        ),
        ('TIMESYS' , 'time system'                                    ),
        ('FRAMEID' , 'frame ID in observation run'                    ),
        ('RA'      , 'right ascension of object'                      ),
        ('DEC'     , 'declination of object'                          ),
        ('RADESYS' , 'name of reference frame'                        ),
        ('EQUINOX' , 'epoch of the mean equator and equinox in years' ),
        ('EXPTIME' , 'exposure time in seconds'                       ),
        ('PHO-OFF' , 'offset of photon middle time'                   ),
        ('UTC-STA' , 'UTC at start of exposure'                       ),
        ('UTC-MID' , 'UTC at middle of exposure'                      ),
        ('UTC-PHO' , 'UTC at photon middle of exposure'               ),
        ('UTC-END' , 'UTC at end of exposure'                         ),
        ('LT-STA'  , 'local time at start of exposure'                ),
        ('LT-MID'  , 'local time at middle of exposure'               ),
        ('LT-PHO'  , 'local time at photon middle of exposure'        ),
        ('LT-END'  , 'local time at end of exposure'                  ),
        ('LST-STA' , 'local sidereal time at start'                   ),
        ('LST-MID' , 'local sidereal time at middle'                  ),
        ('LST-PHO' , 'local sidereal time at photon middle'           ),
        ('LST-END' , 'local sidereal time at end'                     ),
        ('MJD-STA' , 'Modified Julian Date of UTC-STA'                ),
        ('MJD-MID' , 'Modified Julian Date of UTC-MID'                ),
        ('MJD-PHO' , 'Modified Julian Date of UTC-PHO'                ),
        ('MJD-END' , 'Modified Julian Date of UTC-END'                ),
        ('AIRM-STA', 'airmass at start of exposure'                   ),
        ('AIRM-MID', 'airmass at middle of exposure'                  ),
        ('AIRM-PHO', 'airmass at photon middle of exposure'           ),
        ('AIRM-END', 'airmass at end of exposure'                     ),
        ('AIRMASS' , 'effective airmass during exposure'              ),
        ('ALT-STA' , 'telescope altitude at start'                    ),
        ('ALT-MID' , 'telescope altitude at middle'                   ),
        ('ALT-PHO' , 'telescope altitude at photon middle'            ),
        ('ALT-END' , 'telescope altitude at end'                      ),
        ('AZ-STA'  , 'telescope azimuth at start'                     ),
        ('AZ-MID'  , 'telescope azimuth at middle'                    ),
        ('AZ-PHO'  , 'telescope azimuth at photon middle'             ),
        ('AZ-END'  , 'telescope azimuth at end'                       ),
        ('MOON-AGE', 'days past new moon at middle of exposure'       ),
        ('MOON-ALT', 'moon altitude at middle of exposure'            ),
        ('MOON-AZ' , 'moon azimuth at middle of exposure'             ),
        ('MOON-DIS', 'angular distance to moon (in degree)'           ),
        ('TWI-END' , 'end time of astronomical twilight in UTC'       ),
        ('TWI-STA' , 'start time of astronomical twilight in UTC'     ),
        ('PROP-ID' , 'proposal ID'                                    ),
        ('PROP-TIT', 'title of proposal'                              ),
        ('PROP-PI' , 'principal investigator of proposal'             ),
        ('OBSERVER', 'people who acquire the data'                    ),
        ('OBSERVAT', 'observatory where the data is acquired'         ),
        ('TELESCOP', 'telescope used to acquire the data'             ),
        ('OBS-LONG', 'longitude of the telescope'                     ), 
        ('OBS-LAT' , 'latitude of the telescope'                      ),
        ('OBS-ALT' , 'altitude of the telescope in meter'             ),
        ('INSTRUME', 'instrument used to acquire the data'            ),
        ('SETUP-ID', 'ID of the instrument setup'                     ),
        ('SLT-WID' , 'slit width (in mm)'                             ),
        ('SLT-LEN' , 'slit length (in mm)'                            ),
        ('NCHANNEL', 'number of simultaneous channels'                ),
        ('CHANNEL1', 'object of channel 1'                            ),
        ('CHANNEL2', 'object of channel 2'                            ),
        ('FILTER1' , 'filter in channel 1'                            ),
        ('FILTER2' , 'filter in channel 2'                            ),
        ('EXPMETER', 'usage of exposure meter'                        ),
        ('SHAK_STA', 'status of fiber shaker (on/off)'                ),
        ('SHAK_FRE', 'frequency of fiber shaker (in Hz)'              ),
        ('SHAK_AMP', 'amplitude of fiber shaker'                      ),
        ('DETECTOR', 'detector used to acquire the data'              ),
        ('GAIN'    , 'readout gain of detector (in electron/ADU)'     ),
        ('RO-SPEED', 'read out speed of detector'                     ),
        ('RO-NOISE', 'read out noise of detector'                     ),
        ('BINAXIS1', 'binning factor of data axis 1'                  ),
        ('BINAXIS2', 'binning factor of data axis 2'                  ),
        ('TEMP-DET', 'temperature of detector (in degree)'            ),
        ('TEMP-BOX', 'temperature inside instrument box (in degree)'  ),
        ('TEMP-ROO', 'temperature inside instrument room (in degree)' ),
        ('PRES-BOX', 'pressure inside instrument box (in hPa)'        ),
        ('DATE'    , 'file creation date'                             ),
        ('ORI-NAME', 'original filename'                              ),
        ('ORIGIN'  , 'organization responsible for the FITS file'     ),
        ('HEADVER' , 'version of header'                              ),
        ]
    now = datetime.datetime.now()
    header_lst = []
    for key, comment in lst:
        if key in input_lst.keys():
            value = input_lst[key]
        else:
            value = None
        if type(value) == type('a'):
            value = "'%-8s'"%value
            value = value.ljust(20)
        elif type(value) == type(u'a'):
            value = value.encode('ascii','replace')
            value = "'%-8s'"%value
            value = value.ljust(20)
        elif type(value) == type(1):
            value = '%20d'%value
        elif type(value) == type(1.0):
            if key[0:4]=='MJD-':
                # for any keywords related to MJD, keep 6 decimal places.
                # for reference, 1 sec = 1.16e-5 days
                value = '%20.6f'%value
            else:
                value = str(value).rjust(20)
                value = value.replace('e','E')
        elif type(value) == type(now):
            # if value is a python datetime object
            value = "'%04d-%02d-%02dT%02d:%02d:%02d.%03d'"%(
                    value.year, value.month, value.day,
                    value.hour, value.minute, value.second,
                    int(round(value.microsecond*1e-3))
                    )
        elif value == True:
            value = 'T'.rjust(20)
        elif value == False:
            value = 'F'.rjust(20)
        elif value == None:
            value = "''".ljust(20)
        else:
            print('Unknown value: {}'.format(value))
        string = '%-8s= %s / %s'%(key,value,comment)
        if len(string)>=80:
            string = string[0:80]
        else:
            string = string.ljust(80)

        header_lst.append(string)

    return header_lst

def plot_overscan_variation(t_lst, overscan_lst, figfile):
    """Plot the variation of overscan.
    """
    
    # Quality check plot of the mean overscan value over time 
    fig = plt.figure(figsize=(8,6), dpi=150)
    ax2  = fig.add_axes([0.1,0.60,0.85,0.35])
    ax1  = fig.add_axes([0.1,0.15,0.85,0.35])
    #conversion of the DATE-string to a number
    date_lst = [dateutil.parser.parse(t) for t in t_lst]
    datenums = mdates.date2num(date_lst)

    ax1.plot_date(datenums, overscan_lst, 'r-', label='mean')
    ax2.plot(overscan_lst, 'r-', label='mean')
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right')
        leg.get_frame().set_alpha(0.1)
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Frame')
    ax1.set_ylabel('Overscan mean ADU')
    ax2.set_ylabel('Overscan mean ADU')
    # adjust x and y limit
    y11,y12 = ax1.get_ylim()
    y21,y22 = ax2.get_ylim()
    z1 = min(y11,y21)
    z2 = max(y21,y22)
    ax1.set_ylim(z1,z2)
    ax2.set_ylim(z1,z2)
    ax2.set_xlim(0, len(overscan_lst)-1)
    # adjust rotation angle of ticks in time axis
    plt.setp(ax1.get_xticklabels(),rotation=30)

    # save figure
    fig.savefig(figfile)
    plt.close(fig)

def plot_bias_smooth(bias, bias_smooth, comp_figfile, hist_figfile):
    """Plot the bias, smoothed bias, and residual after smoothing.

    A figure will be generated in the report directory of the reduction.
    The name of the figure is given in the config file.

    Args:
        bias (:class:`numpy.ndarray`): Bias array.
        bias_smooth (:class:`numpy.ndarray`): Smoothed bias array.
        comp_figfile (str): Filename of the comparison figure.
        hist_figfile (str): Filename of the histogram figure.
    """
    h, w = bias.shape
    # calculate the residual between bias and smoothed bias data
    bias_res = bias - bias_smooth

    fig1 = plt.figure(figsize=(12,4), dpi=150)
    ax1 = fig1.add_axes([0.055, 0.12, 0.25, 0.75])
    ax2 = fig1.add_axes([0.355, 0.12, 0.25, 0.75])
    ax3 = fig1.add_axes([0.655, 0.12, 0.25, 0.75])
    mean = bias.mean(dtype=np.float64)
    std  = bias.std(dtype=np.float64, ddof=1)
    vmin = mean - 2.*std
    vmax = mean + 2.*std
    cax1 = ax1.imshow(bias,        vmin=vmin, vmax=vmax, cmap='gray')
    cax2 = ax2.imshow(bias_smooth, vmin=vmin, vmax=vmax, cmap='gray')
    cax3 = ax3.imshow(bias_res,    vmin=vmin, vmax=vmax, cmap='gray')
    cbar_ax = fig1.add_axes([0.925, 0.12, 0.02, 0.75])
    cbar = fig1.colorbar(cax1, cax=cbar_ax)
    ax1.set_title('bias')
    ax2.set_title('bias_smooth')
    ax3.set_title('bias - bias_smooth')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(0, bias.shape[1]-1)
        ax.set_ylim(bias.shape[1]-1, 0)
        ax.set_xlabel('X', fontsize=11)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
    # only show y label in the left panel
    ax1.set_ylabel('Y',fontsize=11)
    
    # plot the histogram of smoothed bias
    # prepare the bin list
    bins = np.linspace(-4, 4, 40+1)
    
    # prepare the gaussian fitting and error function
    fitfunc = lambda p,x:p[0]*np.exp(-0.5*(x-p[1])**2/p[2]**2)
    errfunc = lambda p,x,y: y-fitfunc(p,x)
    
    # create figure
    fig2 = plt.figure(figsize=(8,6), dpi=150)
    for i, j in [(i, j) for i in range(3) for j in range(3)]:
        ax = fig2.add_axes([0.1+j*0.3, 0.7-i*0.3, 0.27, 0.27])
    
    labels = 'abcdefghi'
    alpha  = 0.7
    # plot both bias and smoothed bias
    for idata,data in enumerate([bias,bias_res]):
        message = ['Parameters for gaussian fitting of the histograms',
                   'y, x, A, center, sigma']
        for iy, ix in [(iy, ix) for iy in range(3) for ix in range(3)]:
            yc = iy*(h//4) + h//4
            xc = ix*(w//4) + w//4
            x1, x2 = xc-200, xc+200
            y1, y2 = yc-200, yc+200
            ax1.plot([x1,x2], [y1,y1], 'm-', alpha=alpha)
            ax1.plot([x1,x2], [y2,y2], 'm-', alpha=alpha)
            ax1.plot([x1,x1], [y1,y2], 'm-', alpha=alpha)
            ax1.plot([x2,x2], [y1,y2], 'm-', alpha=alpha)
            ax3.plot([x1,x2], [y1,y1], 'c-', alpha=alpha)
            ax3.plot([x1,x2], [y2,y2], 'c-', alpha=alpha)
            ax3.plot([x1,x1], [y1,y2], 'c-', alpha=alpha)
            ax3.plot([x2,x2], [y1,y2], 'c-', alpha=alpha)
            ax1.text(xc-50,yc-20,'(%s)'%labels[iy*3+ix],color='m')
            ax3.text(xc-50,yc-20,'(%s)'%labels[iy*3+ix],color='c')
            data_cut = data[y1:y2,x1:x2]
            y,_ = np.histogram(data_cut, bins=bins)
            x = (np.roll(bins,1) + bins)/2
            x = x[1:]
            # use least square minimization function in scipy
            p1,succ = opt.leastsq(errfunc,[y.max(),0.,1.],args=(x,y))
            ax = fig2.get_axes()[iy*3+ix]
            color1 = ('r', 'b')[idata]
            color2 = ('m', 'c')[idata]
            # plot the histogram
            ax.bar(x, y, align='center', color=color1, width=0.2, alpha=0.5)
            # plot the gaussian fitting of histogram
            xnew = np.linspace(x[0], x[-1], 201)
            ax.plot(xnew, fitfunc(p1, xnew), color2+'-', lw=2)
            ax.set_xlim(-4, 4)
            x1,x2 = ax.get_xlim()
            y1,y2 = ax.get_ylim()
            message.append('%4d %4d %+10.8e %+10.8e %+6.3f'%(
                            yc, xc, p1[0], p1[1], p1[2]))
    
        # write the fitting parameters into running log
        logger.info((os.linesep+'  ').join(message))
   
    # find maximum y in different axes
    max_y = 0
    for iax, ax in enumerate(fig2.get_axes()):
        y1, y2 = ax.get_ylim()
        if y2 > max_y:
            max_y = y2
    
    # set y range for all axes
    for iax, ax in enumerate(fig2.get_axes()):
        x1, x2 = ax.get_xlim()
        ax.text(0.9*x1+0.1*x2, 0.2*y1+0.8*y2, '(%s)'%labels[iax],
                fontsize=12)
        ax.set_ylim(0, max_y)
    
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
    
        if iax in [0, 3, 6]:
            ax.set_ylabel('$N$', fontsize=11)
        else:
            ax.set_yticklabels([])
        if iax in [6, 7, 8]:
            ax.set_xlabel('Counts', fontsize=11)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(9)

    # save figures
    fig1.savefig(comp_figfile)
    fig2.savefig(hist_figfile)
    plt.close(fig1)
    plt.close(fig2)

class BiasFigure(Figure):
    """Figure to plot the bias image.
    """
    def __init__(self, dpi=150, figsize=(12,8), data=None, scale=(5, 95),
            title=None):
        Figure.__init__(self, dpi=dpi, figsize=figsize)
        l1 = 0.07
        b1 = 0.10
        h1 = 0.80
        w1 = h1/figsize[0]*figsize[1]
        l2 = 0.67
        w2 = 0.30
        hgap1 = 0.08
        h3 = 0.02
        h2 = (h1-2*hgap1-h3)/2
        self.ax_image = self.add_axes([l1, b1, w1, h1])
        self.ax_hist1 = self.add_axes([l2, b1+h3+hgap1*2+h2, w2, h2])
        self.ax_hist2 = self.add_axes([l2, b1+h3+hgap1, w2, h2])
        self.ax_cbar0 = self.add_axes([l2, b1, w2, h3])
        
        vmin = np.percentile(data, scale[0])
        vmax = np.percentile(data, scale[1])
        cax = self.ax_image.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
        self.colorbar(cax, cax=self.ax_cbar0, orientation='horizontal')
        self.ax_image.set_xlabel('X (pixel)')
        self.ax_image.set_ylabel('X (pixel)')

        # plot hist1, the whole histogram
        self.ax_hist1.hist(data.flatten(), bins=50)
        self.ax_hist1.axvline(x=vmin, color='C1', ls='--', lw=0.7,
                label='{} %'.format(scale[0]))
        self.ax_hist1.axvline(x=vmax, color='C2', ls='--', lw=0.7,
                label='{} %'.format(scale[1]))
        self.ax_hist1.set_xlabel('Count')
        legend = self.ax_hist1.legend(loc='upper right')
        legend.get_frame().set_alpha(0.1)

        self.ax_hist2.hist(data.flatten(), bins=np.linspace(vmin, vmax, 50))
        self.ax_hist2.set_xlim(vmin, vmax)
        self.ax_hist2.set_xlabel('Count')

        if title is not None:
            self.suptitle(title)


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
        p0 = [A0, c0, 3.0, b0]
        lower_bounds = [-np.inf, xdata[0],  0.3,    -np.inf]
        upper_bounds = [np.inf,  xdata[-1], np.inf, ydata.max()]
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
    
        A, c, sigma, bkg = p1
        self.x = xdata - c
        self.y = (ydata - bkg)/A
        self.m = _m

        self.param = p1
        self.std = std

    def is_succ(self):
        A, center, sigma, bkg = self.param
        std = self.std

        if A>0 and 1<sigma<5 and A/std>10:
            return True
        else:
            return False

    def fitfunc(self, param, x):
        """Use Gaussian function.
        """
        A, center, sigma, bkg = param
        return A*np.exp(-(x-center)**2/2./sigma**2) + bkg

def norm_profile(xdata, ydata, mask):
    # define the fitting and error functions
    def gaussian_bkg(A, center, sigma, bkg, x):
        return A*np.exp(-(x-center)**2/2./sigma**2) + bkg
    def fitfunc(p, x):
        return gaussian_bkg(p[0], p[1], p[2], p[3], x)
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # iterative fitting using gaussian + bkg function
    A0 = ydata.max()-ydata.min()
    c0 = (xdata[0]+xdata[-1])/2
    b0 = ydata.min()
    p0 = [A0, c0, 3.0, b0]
    lower_bounds = [-np.inf, xdata[0],  0.3,    -np.inf]
    upper_bounds = [np.inf,  xdata[-1], np.inf, ydata.max()]
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

    A, c, sigma, bkg = p1
    newx = xdata - c
    newy = ydata - bkg

    param = (A, c, sigma, bkg, std)

    if A < 1e-3:
        return None
    return newx, newy/A, param

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.43,0.86])
        self.ax2 = self.add_axes([0.52,0.50,0.43,0.40])
        self.ax3 = self.add_axes([0.52,0.10,0.43,0.40])
        self.ax4 = self.ax3.twinx()

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
            self.plot_background(data, background)
        if title is not None:
            self.suptitle(title)
        if figname is not None:
            self.savefig(figname)

    def plot_background(self, data, background, scale=(5, 99)):
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

class SpatialProfileFigure(Figure):
    """Figure to plot the cross-dispersion profiles.

    """
    def __init__(self,
            nrow = 2,
            ncol = 3,
            figsize = (12,6),
            dpi = 200,
            ):

        # create figure
        Figure.__init__(self, figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self)

        # add axes
        _w = 0.27
        _h = 0.40
        for irow in range(nrow):
            for icol in range(ncol):
                _x = 0.08 + icol*0.31
                _y = 0.06 + (nrow-1-irow)*0.45

                ax = self.add_axes([_x, _y, _w, _h])

    def close(self):
        plt.close(self)
