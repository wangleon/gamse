import os
import re
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time  import Time

from ..echelle.imageproc import combine_images
from ..echelle.trace import find_apertures, load_aperture_set, TraceFigureCommon
from ..echelle.background import simple_debackground
from ..echelle.extract import extract_aperset
from ..echelle.flat import get_slit_flat
from ..utils.obslog import parse_num_seq, read_obslog
from ..utils.misc import extract_date

from .common import FormattedInfo

all_columns = [
        ('frameid',  'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',   'str',   '{:^17s}', '{0[fileid]:17s}'),
        ('imgtype',  'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',   'str',   '{:^20s}', '{0[object]:20s}'),
        ('i2cell',   'bool',  '{:^6s}',  '{0[i2cell]!s: <6}'),
        ('exptime',  'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate',  'time',  '{:^23s}', '{0[obsdate]:}'),
        ('deckname', 'str',   '{:^8s}',  '{0[deckname]:^8s}'),
        ('filter1',  'str',   '{:^7s}',  '{0[filter1]:^7s}'),
        ('filter2',  'str',   '{:^7s}',  '{0[filter2]:^7s}'),
        ('nsat_1',   'int',   '{:^8s}',  '\033[34m{0[nsat_1]:8d}\033[0m'),
        ('nsat_2',   'int',   '{:^8s}',  '\033[32m{0[nsat_2]:8d}\033[0m'),
        ('nsat_3',   'int',   '{:^8s}',  '\033[31m{0[nsat_3]:8d}\033[0m'),
        ('q95_1',    'int',   '{:^8s}',  '\033[34m{0[q95_1]:8d}\033[0m'),
        ('q95_2',    'int',   '{:^8s}',  '\033[32m{0[q95_2]:8d}\033[0m'),
        ('q95_3',    'int',   '{:^8s}',  '\033[31m{0[q95_3]:8d}\033[0m'),
        ]

def print_wrapper(string, item):
    """A wrapper for log printing for HIRES pipeline.

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

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    config.set('data', 'telescope',   'Keck-I')
    config.set('data', 'instrument',  'HIRES')
    config.set('data', 'rawdata',     'rawdata')
    #config.set('data', 'statime_key', statime_key)
    #config.set('data', 'exptime_key', exptime_key)

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
    config.set('reduce.trace', 'minimum',    str(1e-3))
    config.set('reduce.trace', 'scan_step',  str(100))
    config.set('reduce.trace', 'separation', '100:84, 1500:45, 3000:14')
    config.set('reduce.trace', 'filling',    str(0.2))
    config.set('reduce.trace', 'align_deg',  str(2))
    config.set('reduce.trace', 'display',    'no')
    config.set('reduce.trace', 'degree',     str(4))
    config.set('reduce.trace', 'file',       '${reduce:midproc}/trace.fits')

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'file', '${reduce:midproc}/flat.fits')

    # write to config file
    filename = 'HIRES.{}.cfg'.format(input_date)
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

def make_obslog(path):
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (str): Path to the raw FITS files.

    """
    name_pattern = '^HI\.\d{8}\.\d{5}\.fits$'

    # scan the raw files
    fname_lst = sorted(os.listdir(path))

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),   ('fileid', 'S17'),   ('imgtype', 'S3'),
        ('object',  'S20'),  ('i2cell', 'bool'),  ('exptime', 'f4'),
        ('obsdate',  Time),
        ('deckname', 'S2'),  ('filter1', 'S5'),   ('filter2', 'S5'),
        ('nsat_1',   'i4'),  ('nsat_2',  'i4'),   ('nsat_3',  'i4'),
        ('q95_1',    'i4'),  ('q95_2',   'i4'),   ('q95_3',   'i4'),
        ])

    # prepare infomation to print
    pinfo = FormattedInfo(all_columns,
            ['frameid', 'fileid', 'imgtype', 'object', 'i2cell', 'exptime',
             'obsdate', 'deckname', 'nsat_2', 'q95_2'])

    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    print(pinfo.get_separator())

    # start scanning the raw files
    prev_frameid = -1
    for fname in fname_lst:
        if not re.match(name_pattern, fname):
            continue
        fileid = fname[0:17]
        filename = os.path.join(path, fname)
        hdu_lst = fits.open(filename)
        # parse images
        data_lst, mask_lst = parse_3ccd_images(hdu_lst)

        head0 = hdu_lst[0].header

        frameid = prev_frameid + 1

        # get obsdate in 'YYYY-MM-DDTHH:MM:SS' format
        date = head0.get('DATE-OBS')
        utc  = head0.get('UTC', head0.get('UT'))
        obsdate = Time('%sT%s'%(date, utc))

        exptime    = head0.get('ELAPTIME')
        i2in       = head0.get('IODIN', False)
        i2out      = head0.get('IODOUT', True)
        i2cell     = i2in
        imagetyp   = head0.get('IMAGETYP')
        targname   = head0.get('TARGNAME', '')
        lampname   = head0.get('LAMPNAME', '')

        if imagetyp == 'object':
            # science frame
            imgtype    = 'sci'
            objectname = targname
        elif imagetyp == 'flatlamp':
            # flat
            imgtype    = 'cal'
            objectname = '{} ({})'.format(imagetyp, lampname)
        elif imagetyp == 'arclamp':
            # arc lamp
            imgtype    = 'cal'
            objectname = '{} ({})'.format(imagetyp, lampname)
        elif imagetyp == 'bias':
            imgtype    = 'cal'
            objectname = 'bias'
        else:
            print('Unknown IMAGETYP:', imagetyp)

        # get deck and filter information
        deckname = head0.get('DECKNAME', '')
        filter1  = head0.get('FIL1NAME', '')
        filter2  = head0.get('FIL2NAME', '')

        # determine the numbers of saturated pixels for 3 CCDs
        mask_sat1 = (mask_lst[0] & 4)>0
        mask_sat2 = (mask_lst[1] & 4)>0
        mask_sat3 = (mask_lst[2] & 4)>0
        nsat_1 = mask_sat1.sum()
        nsat_2 = mask_sat2.sum()
        nsat_3 = mask_sat3.sum()

        # find the 95% quantile
        q95_lst = [np.sort(data.flatten())[int(data.size*0.95)]
                    for data in data_lst]
        q95_1, q95_2, q95_3 = q95_lst

        # close the fits file
        hdu_lst.close()

        item = [frameid, fileid, imgtype, objectname, i2cell, exptime, obsdate,
                deckname, filter1, filter2,
                nsat_1, nsat_2, nsat_3, q95_1, q95_2, q95_3]

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

    # determine filename of logtable.
    # use the obsdate of the LAST frame.
    obsdate = logtable[-1]['obsdate'].iso[0:10]
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

    # loginfo is not pinfo because not all columns need to be printed in the
    # screen, but all columns should be written in logfile.
    loginfo = FormattedInfo(all_columns)
    outfile = open(outfilename, 'w')
    outfile.write(loginfo.get_title()+os.linesep)
    outfile.write(loginfo.get_dtype()+os.linesep)
    outfile.write(loginfo.get_separator()+os.linesep)
    for row in logtable:
        outfile.write(loginfo.get_format(has_esc=False).format(row)+os.linesep)
    outfile.close()

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

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.50,0.86])
        self.ax2 = self.add_axes([0.59,0.55,0.36,0.34])
        self.ax3 = self.add_axes([0.59,0.13,0.36,0.34])
        self.ax4 = self.ax3.twinx()

def mosaic_3_images(data_lst, mask_lst):
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

def reduce():
    """2D to 1D pipeline for Keck/HIRES.
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

    # load config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation = configparser.ExtendedInterpolation(),
                )
    # find local config file
    for fname in os.listdir(os.curdir):
        if fname[-4:]=='.cfg':
            config.read(fname)
            print('Load Congfile File: {}'.format(fname))
            break

    # extract keywords from config file
    section = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    section = config['reduce']
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

    nccd = 3

    ########################## load file selection #############################
    sel_lst = {}
    filesel_filename = 'file_selection.txt'
    if os.path.exists(filesel_filename):
        sel_file = open(filesel_filename)
        for row in sel_file:
            row = row.strip()
            if len(row)==0 or row[0] in '#':
                continue
            g = row.split(':')
            key, value = g[0].strip(), g[1].strip()
            if len(value)>0:
                sel_lst[key] = value
        sel_file.close()

    ################################ parse bias ################################
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        # pack bias image
        bias = [hdu_lst[iccd+1].data for iccd in range(nccd)]
        hdu_lst.close()
        message = 'Load bias data from file: {}'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        # read each individual CCD
        bias_data_lst = [[] for iccd in range(nccd)]

        # initialize printing infomation
        pinfo1 = FormattedInfo(all_columns, ['frameid', 'fileid', 'object',
                    'exptime', 'nsat_1', 'q95_1', 'nsat_2', 'q95_2',
                    'nsat_3', 'q95_3'])

        for logitem in logtable:
            if logitem['object'].strip().lower()=='bias':
                fname = '{}.fits'.format(logitem['fileid'])
                filename = os.path.join(rawdata, fname)
                hdu_lst = fits.open(filename)
                data_lst, mask_lst = parse_3ccd_images(hdu_lst)
                hdu_lst.close()

                # print info
                if len(bias_data_lst[0]) == 0:
                    print('* Combine Bias Images: {}'.format(bias_file))
                    print(' '*2 + pinfo1.get_separator())
                    print(' '*2 + pinfo1.get_title())
                    print(' '*2 + pinfo1.get_separator())
                string = pinfo1.get_format().format(logitem)
                print(' '*2 + print_wrapper(string, logitem))

                for iccd in range(nccd):
                    bias_data_lst[iccd].append(data_lst[iccd])

        n_bias = len(bias_data_lst[0])      # get number of bias images
        has_bias = n_bias > 0

        if has_bias:
            # there is bias frames
            print(' '*2 + pinfo1.get_separator())

            bias = []
            # the final HDU list
            bias_hdu_lst = fits.HDUList([fits.PrimaryHDU()])

            # scan for each ccd
            for iccd in range(nccd):
                ### 3 CCDs loop begins here ###
                bias_data_lst[iccd] = np.array(bias_data_lst[iccd])

                section = config['reduce.bias']
                sub_bias = combine_images(bias_data_lst[iccd],
                            mode       = 'mean',
                            upper_clip = section.getfloat('cosmic_clip'),
                            maxiter    = section.getint('maxiter'),
                            mask       = (None, 'max')[n_bias>=3],
                            )

                message = '\033[{2}mCombined bias for CCD {0}: Mean = {1:6.2f}\033[0m'.format(
                    iccd+1, sub_bias.mean(), (34, 32, 31)[iccd])

                print(message)

                head = fits.Header()
                head['HIERARCH GAMSE BIAS NFILE'] = n_bias

                ############## bias smooth ##################
                section = config['reduce.bias']
                if section.getboolean('smooth'):
                    # bias needs to be smoothed
                    smooth_method = section.get('smooth_method')

                    h, w = sub_bias.shape
                    if smooth_method in ['gauss', 'gaussian']:
                        # perform 2D gaussian smoothing
                        smooth_sigma = section.getint('smooth_sigma')
                        smooth_mode  = section.get('smooth_mode')
                        
                        bias_smooth = gaussian_filter(sub_bias,
                                        sigma=smooth_sigma, mode=smooth_mode)

                        # write information to FITS header
                        head['HIERARCH GAMSE BIAS SMOOTH']        = True
                        head['HIERARCH GAMSE BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                        head['HIERARCH GAMSE BIAS SMOOTH SIGMA']  = smooth_sigma
                        head['HIERARCH GAMSE BIAS SMOOTH MODE']   = smooth_mode
                    else:
                        print('Unknown smooth method: ', smooth_method)
                        pass

                    sub_bias = bias_smooth
                else:
                    # bias not smoothed
                    head['HIERARCH GAMSE BIAS SMOOTH'] = False

                bias.append(sub_bias)
                bias_hdu_lst.append(fits.ImageHDU(data=sub_bias, header=head))
                ### 3 CCDs loop ends here ##

            # write bias into file
            bias_hdu_lst.writeto(bias_file, overwrite=True)

        else:
            # no bias found
            pass

    ########################## find flat groups #########################
    flat_file = config['reduce.flat'].get('flat_file')

    flatdata_lst = []
    # a list of 3 combined flat images. [Image1, Image2, Image3]
    # bias has been corrected already. but not rotated yet.
    flatmask_lst = []
    # a list of 3 flat masks

    if mode=='debug' and os.path.exists(flat_file):
        # read flat data from existing file
        hdu_lst = fits.open(flat_file)
        for iccd in range(nccd):
            flatdata_lst.append(hdu_lst[iccd*2+1].data)
            flatmask_lst.append(hdu_lst[iccd*2+2].data)
        flatdata = hdu_lst[nccd*2+1].data.T
        flatmask = hdu_lst[nccd*2+2].data.T
        hdu_lst.close()
        message = 'Loaded flat data from file: {}'.format(flat_file)
        print(message)

        # alias of flat data and mask
        flatdata1 = flatdata_lst[0].T
        flatmask1 = flatmask_lst[0].T
        flatdata2 = flatdata_lst[1].T
        flatmask2 = flatmask_lst[1].T
        flatdata3 = flatdata_lst[2].T
        flatmask3 = flatmask_lst[2].T

    else:
        print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)
        # print the flat list
        pinfo_flat = FormattedInfo(all_columns, ['frameid', 'fileid', 'object',
            'exptime', 'nsat_1', 'q95_1', 'nsat_2', 'q95_2', 'nsat_3', 'q95_3'])
        print(' '*2 + pinfo_flat.get_separator())
        print(' '*2 + pinfo_flat.get_title())
        print(' '*2 + pinfo_flat.get_separator())
        for logitem in logtable:
            if len(logitem['object'])>=8 and logitem['object'][0:8]=='flatlamp':
                string = pinfo_flat.get_format().format(logitem)
                print(' '*2 + print_wrapper(string, logitem))
        print(' '*2 + pinfo_flat.get_separator())


        flat_group_lst = {}
        for iccd in range(nccd):

            key = 'flat CCD%d'%(iccd+1)
            sel_string = sel_lst[key] if key in sel_lst else ''
            prompt = '\033[{1}mSelect flats for CCD {0} [{2}]: \033[0m'.format(
                      iccd+1, (34, 32, 31)[iccd], sel_string)

            # read selected files from terminal
            while(True):
                input_string = input(prompt)
                if len(input_string.strip())==0:
                    # nothing input
                    if key in sel_lst:
                        # nothing input but already in selection list
                        flat_group_lst[iccd] = parse_num_seq(sel_lst[key])
                        break
                    else:
                        # repeat prompt
                        continue
                else:
                    # something input
                    frameid_lst = parse_num_seq(input_string)
                    # pack
                    flat_group_lst[iccd] = frameid_lst
                    # put input string into selection list
                    sel_lst[key] = input_string.strip()
                    break

        # now combine flat images

        flat_hdu_lst = [fits.PrimaryHDU()]
        # flat_hdu_lst is the final HDU list to be saved as fits

        for iccd in range(nccd):
            frameid_lst = flat_group_lst[iccd]

            # now combine flats for this CCD
            flat_data_lst = []
            # flat_data_lst is a list of flat images to be combined.
            # flat_data_lst = [Image1, Image2, Image3, Image4, ... ...]

            #scan the logtable
            # log loop inside the CCD loop because flats for different CCDs are
            # in different files
            for logitem in logtable:
                if logitem['frameid'] in frameid_lst:
                    filename = os.path.join(rawdata, logitem['fileid']+'.fits')
                    hdu_lst = fits.open(filename)
                    data_lst, mask_lst = parse_3ccd_images(hdu_lst)
                    hdu_lst.close()

                    # correct bias and pack into flat_data_lst
                    if has_bias:
                        flat_data_lst.append(data_lst[iccd]-bias[iccd])
                    else:
                        flat_data_lst.append(data_lst[iccd])

                    # initialize flat mask
                    if len(flat_data_lst) == 1:
                        flatmask = mask_lst[iccd]
                    flatmask = flatmask | mask_lst[iccd]

            n_flat = len(flat_data_lst)

            if n_flat == 0:
                continue
            elif n_flat == 1:
                flatdata = flat_data_lst[0]
            else:
                flat_data_lst = np.array(flat_data_lst)
                flatdata = combine_images(flat_data_lst,
                            mode       = 'mean',
                            upper_clip = 10,
                            maxiter    = 5,
                            mask       = (None, 'max')[n_flat>=3],
                            )
                #print('\033[{1}mCombined flat data for CCD {0}: \033[0m'.format(
                #    iccd+1, (34, 32, 31)[iccd]))
            flatdata_lst.append(flatdata)
            flatmask_lst.append(flatmask)

            # pack the combined flat data into flat_hdu_lst
            head = fits.Header()
            head['HIERARCH GAMSE FLAT CCD{} NFILE'.format(iccd+1)] = n_flat
            flat_hdu_lst.append(fits.ImageHDU(flatdata, head))
            flat_hdu_lst.append(fits.ImageHDU(flatmask))
        # CCD loop ends here

        # alias of flat data and mask
        flatdata1 = flatdata_lst[0].T
        flatmask1 = flatmask_lst[0].T
        flatdata2 = flatdata_lst[1].T
        flatmask2 = flatmask_lst[1].T
        flatdata3 = flatdata_lst[2].T
        flatmask3 = flatmask_lst[2].T

        # mosaic flat data
        flatdata, flatmask = mosaic_3_images(
                                data_lst = (flatdata1, flatdata2, flatdata3),
                                mask_lst = (flatmask1, flatmask2, flatmask3),
                                )

        flat_hdu_lst.append(fits.ImageHDU(flatdata.T))
        flat_hdu_lst.append(fits.ImageHDU(flatmask.T))
        # write flat data to file
        flat_hdu_lst = fits.HDUList(flat_hdu_lst)
        flat_hdu_lst.writeto(flat_file, overwrite=True)
        print('Flat data writed to {}'.format(flat_file))

    ######################### find & trace orders ##########################

    # simple debackground for all 3 CCDs
    xnodes = np.arange(0, flatdata1.shape[1], 200)
    flatdbkg1 = simple_debackground(flatdata1, flatmask1, xnodes, smooth=20,
                    deg=3, maxiter=10)

    xnodes = np.arange(0, flatdata2.shape[1], 200)
    flatdbkg2 = simple_debackground(flatdata2, flatmask2, xnodes, smooth=20,
                    deg=3, maxiter=10)

    xnodes = np.arange(0, flatdata3.shape[1], 200)
    flatdbkg3 = simple_debackground(flatdata3, flatmask3, xnodes, smooth=20,
                    deg=3, maxiter=10)

    allimage, allmask = mosaic_3_images(
                        data_lst = (flatdbkg1, flatdbkg2, flatdbkg3),
                        mask_lst = (flatmask1, flatmask2, flatmask3),
                        )

    tracefig = TraceFigure()

    section = config['reduce.trace']
    aperset = find_apertures(allimage, allmask,
                scan_step  = section.getint('scan_step'),
                minimum    = section.getfloat('minimum'),
                separation = section.get('separation'),
                align_deg  = section.getint('align_deg'),
                filling    = section.getfloat('filling'),
                degree     = section.getint('degree'),
                display    = section.getboolean('display'),
                fig        = tracefig,
                )
    # decorate trace fig and save to file
    tracefig.adjust_positions()
    tracefig.suptitle('Trace for all 3 CCDs', fontsize=15)
    figfile = os.path.join(report, 'trace.png')
    tracefig.savefig(figfile)

    trcfile = os.path.join(midproc, 'trace.trc')
    aperset.save_txt(trcfile)

    regfile = os.path.join(midproc, 'trace.reg')
    aperset.save_reg(regfile, transpose=True)

    # save mosaiced flat image
    trace_hdu_lst = fits.HDUList(
                        [fits.PrimaryHDU(allimage.T),
                         fits.ImageHDU(allmask.T),
                        ])
    trace_hdu_lst.writeto(config['reduce.trace'].get('file'), overwrite=True)

    ######################### Extract flat spectrum ############################

    spectra1d = extract_aperset(flatdata, flatmask,
                    apertureset = aperset,
                    lower_limit = 6,
                    upper_limit = 6,
                    )

    flatmap = get_slit_flat(flatdata, flatmask,
                apertureset = aperset,
                spectra1d   = spectra1d,
                lower_limit = 6,
                upper_limit = 6,
                deg         = 7,
                q_threshold = 20**2,
                figfile     = 'spec_%02d.png',
                )
    fits.writeto('flat_resp.fits', flatmap, overwrite=True)
