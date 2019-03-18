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
from ..utils import obslog
from .common import PrintInfo

print_columns = [
        ('frameid',    'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',     'str',   '{:^17s}', '{0[fileid]:17s}'),
        ('imgtype',    'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',     'str',   '{:^15s}', '{0[object]:15s}'),
        ('i2cell',     'bool',  '{:^6s}',  '{0[i2cell]!s: <6}'),
        ('exptime',    'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate',    'time',  '{:^23s}', '{0[obsdate]:}'),
        ('deckname',   'str',   '{:^8s}',  '{0[deckname]:^8s}'),
        ('filter1',    'str',   '{:^7s}',  '{0[filter1]:^7s}'),
        ('filter2',    'str',   '{:^7s}',  '{0[filter2]:^7s}'),
        ('saturation', 'int',   '{:^10s}', '{0[saturation]:10d}'),
        ('quantile95', 'int',   '{:^10s}', '{0[quantile95]:10d}'),

        ]

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
        ('object',  'S15'),  ('i2cell', 'bool'),  ('exptime', 'f4'),
        ('obsdate',  Time),
        ('deckname', 'S2'),  ('filter1', 'S5'),   ('filter2', 'S5'),
        ('saturation','i4'), ('quantile95', 'i4'),
        ])

    # prepare infomation to print
    pinfo = PrintInfo(print_columns)

    print(pinfo.get_title())
    print(pinfo.get_dtype())
    print(pinfo.get_separator())

    # start scanning the raw files
    prev_frameid = -1
    for fname in fname_lst:
        if not re.match(name_pattern, fname):
            continue
        fileid = fname[0:17]
        filename = os.path.join(path, fname)
        hdu_lst = fits.open(filename)
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
        objectname = head0.get('TARGNAME', '')
        itype      = head0.get('IMAGETYP')
        if itype.strip() == 'object':
            imgtype = 'sci'
        else:
            imgtype = 'cal'
            objectname = itype
        # get deck and filter information
        deckname = head0.get('DECKNAME', '')
        filter1  = head0.get('FIL1NAME', '')
        filter2  = head0.get('FIL2NAME', '')

        data1 = hdu_lst[1].data
        data2 = hdu_lst[2].data
        data3 = hdu_lst[3].data

        # determine the total number of saturated pixels
        mask_sat1 = (data1==0)
        mask_sat2 = (data2==0)
        mask_sat3 = (data3==0)
        saturation = mask_sat1.sum() + mask_sat2.sum() + mask_sat3.sum()

        # find the 95% quantile
        quantile95 = np.sort(data2.flatten())[int(data2.size*0.95)]

        hdu_lst.close()

        item = [frameid, fileid, imgtype, objectname, i2cell, exptime, obsdate,
                deckname, filter1, filter2, saturation, quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        print(pinfo.get_format().format(item))

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
    outfile = open(outfilename, 'w')
    outfile.write(pinfo.get_title()+os.linesep)
    outfile.write(pinfo.get_dtype()+os.linesep)
    outfile.write(pinfo.get_separator()+os.linesep)
    for row in logtable:
        outfile.write(pinfo.get_format().format(row)+os.linesep)
    outfile.close()


def reduce():
    """2D to 1D pipeline for Keck/HIRES
    """
    # read obs log
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    # load config files
    config_file_lst = []
    # find built-in config file
    config_path = os.path.join(os.path.dirname(__file__), '../data/config')
    config_file = os.path.join(config_path, 'HIRES.cfg')
    if os.path.exists(config_file):
        config_file_lst.append(config_file)

    # find local config file
    for fname in os.listdir(os.curdir):
        if fname[-4:]=='.cfg':
            config_file_lst.append(fname)

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # extract keywords from config file
    section     = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    section     = config['reduce']
    midproc     = section.get('midproc')
    result      = section.get('result')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format = section.get('fig_format')

    # create folders if not exist
    if not os.path.exists(report):  os.mkdir(report)
    if not os.path.exists(result):  os.mkdir(result)
    if not os.path.exists(midproc): os.mkdir(midproc)

    nccd = 3

    ################################ parse bias ################################
    section = config['reduce.bias']
    bias_file = section['bias_file']

    if os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        biasdata_lst = [hdu_lst[iccd+1].data for iccd in range(nccd)]
        hdu_lst.close()
        logger.info('Load bias from image: %s'%bias_file)
    else:
        # read each individual CCD
        bias_lst = [[] for iccd in range(nccd)]
        for item in log:
            if item.objectname[0].strip().lower()=='bias':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                hdu_lst = fits.open(filename)
                for iccd in range(nccd):
                    data = hdu_lst[iccd+1].data
                    bias_lst[iccd].append(np.float64(data))
                hdu_lst.close()

        has_bias = len(bias_lst[0])>0

        if has_bias:
            # there is bias frames
            head_lst     = [] # each bias has a head
            biasdata_lst = []
            hdu_lst = fits.HDUList([fits.PrimaryHDU()])

            # scan for each ccd
            for iccd in range(nccd):
                ### 3 CCDs loop begins here ###
                head = fits.Header()
                head['HIERARCH EDRS BIAS NFILE'] = len(bias_lst[iccd])
                bias = combine_images(bias_lst[iccd],
                        mode       = 'mean',
                        upper_clip = section.getfloat('cosmic_clip'),
                        maxiter    = section.getint('maxiter'),
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
                        
                        bias_smooth = gaussian_filter(bias,
                                        sigma=smooth_sigma, mode=smooth_mode)

                        # write information to FITS header
                        head['HIERARCH EDRS BIAS SMOOTH']        = True
                        head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                        head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                        head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode
                    else:
                        print('Unknown smooth method: ', smooth_method)
                        pass

                    bias = bias_smooth
                else:
                    # bias not smoothed
                    head['HIERARCH EDRS BIAS SMOOTH'] = False

                biasdata_lst.append(bias)
                hdu_lst.append(fits.ImageHDU(data=bias, header=head))
                ### 3 CCDs loop ends here ##

            hdu_lst.writeto(bias_file, overwrite=True)

    ########################## find flat groups #########################

    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)
    # initialize flat_groups
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}
