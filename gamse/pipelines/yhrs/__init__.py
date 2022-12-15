import os
import re
import datetime
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date
from ..common import load_obslog, load_config
from .common import print_wrapper
from .reduce import reduce_rawdata

def make_config():
    """Generate a config file for reducing the data taken with YHRS.

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

    statime_key = 'DATE-STA'
    exptime_key = 'EXPTIME'

    config.set('data', 'telescope',    'Lijiang2.4m')
    config.set('data', 'instrument',   'YHRS')
    config.set('data', 'rawpath',      'rawdata')
    config.set('data', 'statime_key',  statime_key)
    config.set('data', 'exptime_key',  exptime_key)

    config.add_section('reduce')
    config.set('reduce', 'midpath',     'midproc')
    config.set('reduce', 'figpath',     'images')
    config.set('reduce', 'odspath',     'onedspec')
    config.set('reduce', 'mode',        'normal')
    config.set('reduce', 'oned_suffix', 'ods')
    config.set('reduce', 'fig_format',  'png')
    config.set('reduce', 'ncores',      'max')

    # section of bias correction
    sectname = 'reduce.bias'
    config.add_section(sectname)
    config.set(sectname, 'bias_file',     '${reduce:midpath}/bias.fits')
    config.set(sectname, 'cosmic_clip',   str(10))
    config.set(sectname, 'maxiter',       str(5))
    config.set(sectname, 'smooth',        'yes')
    config.set(sectname, 'smooth_method', 'gaussian')
    config.set(sectname, 'smooth_sigma',  str(3))
    config.set(sectname, 'smooth_mode',   'nearest')

    # section of order trace
    sectname = 'reduce.trace'
    config.add_section(sectname)
    config.set(sectname, 'minimum',    str(8))
    config.set(sectname, 'scan_step',  str(100))
    config.set(sectname, 'separation', '500:19, 1500:29, 3500:52')
    config.set(sectname, 'filling',    str(0.3))
    config.set(sectname, 'align_deg',  str(2))
    config.set(sectname, 'display',    'no')
    config.set(sectname, 'degree',     str(3))

    # section of flat field correction
    sectname = 'reduce.flat'
    config.add_section(sectname)
    config.set(sectname, 'slit_step',       str(256))
    config.set(sectname, 'q_threshold',     str(50))
    config.set(sectname, 'mosaic_maxcount', str(50000))

    # section of wavelength calibration
    sectname = 'reduce.wlcalib'
    config.add_section(sectname)
    config.set(sectname, 'search_database',  'yes')
    config.set(sectname, 'linelist',         'thar.dat')
    config.set(sectname, 'use_prev_fitpar',  'yes')
    config.set(sectname, 'window_size',      str(13))
    config.set(sectname, 'xorder',           str(3))
    config.set(sectname, 'yorder',           str(3))
    config.set(sectname, 'maxiter',          str(5))
    config.set(sectname, 'clipping',         str(3))
    config.set(sectname, 'q_threshold',      str(10))
    config.set(sectname, 'auto_selection',   'yes')
    config.set(sectname, 'rms_threshold',    str(0.006))
    config.set(sectname, 'group_contiguous', 'yes')
    config.set(sectname, 'time_diff',        str(120))

    # section of background correction
    sectname = 'reduce.background'
    config.add_section(sectname)
    config.set(sectname, 'subtract',      'yes')
    config.set(sectname, 'ncols',         str(9))
    config.set(sectname, 'distance',      str(7))
    config.set(sectname, 'yorder',        str(7))

    # section of spectra extraction
    sectname = 'reduce.extract'
    config.add_section(sectname)
    #config.set(sectname, 'extract', 
    #        "lambda row: row['imgtype']=='sci' or row['object'].lower()=='i2'")
    #method = {'single':'optimal', 'double':'sum'}[fibermode]
    config.set(sectname, 'method',      'optimal')
    config.set(sectname, 'upper_limit', str(7))
    config.set(sectname, 'lower_limit', str(7))
    config.set(sectname, 'deblaze',     'no')

    # write to config file
    filename = 'YHRS.{}.cfg'.format(input_date)
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

def parse_objectstring(string):

    pattern = ('count(\d+) speed\-([a-zA-Z]+) gain(\d) '
                '([a-zA-Z]+)\-(\d+)s bin\-\-(\d)')
    mobj = re.match(pattern, string)
    count = int(mobj.group(1))
    speed = mobj.group(2)
    gain = int(mobj.group(3))
    mode = mobj.group(4)
    exptime = mobj.group(5)
    binning = int(mobj.group(6))
    return {'count': count, 'speed': speed, 'gain': gain, 'mode': mode,
            'exptime': exptime, 'binning': binning}

def make_obslog():
    """Scan the raw data, and generate a log file containing the detail
    information for each frame.

    An ascii file will be generated after running.
    The name of the ascii file is `YYYY-MM-DD.obslog`, where `YYYY-MM-DD` is the
    date of the *first* FITS image in the data folder.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.

    """
    # load config file
    config = load_config('YHRS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    statime_key = config['data'].get('statime_key')
    exptime_key = config['data'].get('exptime_key')

    # initialize logtable
    logtable = Table(dtype=[
                    ('frameid',     'i2'),
                    ('fileid',      'S12'),
                    ('imgtype',     'S3'),
                    ('object',      'S80'),
                    ('exptime',     'f4'),
                    ('obsdate',     'S23'),
                    ('gain',        'i2'),
                    ('speed',       'S6'),
                    ('binning',     'i2'),
                    ('nsat',        'i4'),
                    ('q95',         'i4'),
            ], masked=True)

    fmt_str = ('  - {:7} {:13s} {:5s} {:<12s} {:>7} {:^23s} '
                '{:4} {:5} {:7} {:>7} {:>5}')
    head_str = fmt_str.format('frameid', 'fileid', 'type', 'object', 'exptime',
                'obsdate', 'gain', 'speed', 'binning', 'nsat', 'q95')
    print(head_str)

    # start scanning the raw files
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match('(Y\d{6}[A-Z])(\d{4})([a-z])\.fits', fname)
        if not mobj:
            continue
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        datasetid = mobj.group(1)
        frameid   = int(mobj.group(2))
        filetype  = mobj.group(3)
        fileid    = '{:7s}{:04d}{:1s}'.format(datasetid, frameid, filetype)
        exptime   = head[exptime_key]
        info      = parse_objectstring(head['OBJECT'])

        # guess object name from filename
        if filetype=='b':
            objectname = 'Bias'
        elif filetype=='f':
            objectname = 'Flat'
        elif filetype=='c':
            objectname = 'ThAr'
        else:
            objectname = ''

        obsdate = head[statime_key]

        if filetype=='o':
            imgtype = 'sci'
        else:
            imgtype = 'cal'

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        item = [frameid, fileid, imgtype, objectname, exptime, obsdate,
                info['gain'], info['speed'], info['binning'],
                saturation, quantile95]
        logtable.add_row(item)

        item = logtable[-1]

        # print log item with colors
        string = fmt_str.format(frameid, fileid, '({:3s})'.format(imgtype),
                    objectname, exptime, obsdate,
                    info['gain'], info['speed'], info['binning'],
                    saturation, quantile95)
        print(print_wrapper(string, item))

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'][0:10]
    outname = 'log.{}.txt'.format(obsdate)
    if os.path.exists(outname):
        i = 0
        while(True):
            i += 1
            outname = 'log.{}.{}.txt'.format(obsdate, i)
            if not os.path.exists(outname):
                outfilename = outname
                break
    else:
        outfilename = outname

    # set display formats
    logtable['imgtype'].info.format = '^s'
    logtable['fileid'].info.format = '<s'
    logtable['object'].info.format = '<s'
    logtable['speed'].info.format = '<s'
    logtable['exptime'].info.format = 'g'

    # save the logtable
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

    message = 'observation log written in {}'.format(outfilename)
    print(message)
