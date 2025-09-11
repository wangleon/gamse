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
from ..xinglong216hrs import parse_idstring, parse_timestr


def make_config():
    """Generate a config file for reducing the data taken with LiRES.

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
    config.set('data', 'instrument',   'LiRES')
    config.set('data', 'rawpath',      'rawdata')
    config.set('data', 'statime_key',  statime_key)
    config.set('data', 'exptime_key',  exptime_key)
    config.set('data', 'direction',    'xr+')

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
    config.set(sectname, 'use_prev_fitpar',  'yes')
    config.set(sectname, 'window_size',      str(23))
    config.set(sectname, 'xorder',           str(4))
    config.set(sectname, 'yorder',           str(4))
    config.set(sectname, 'maxiter',          str(5))
    config.set(sectname, 'clipping',         str(1.5))
    config.set(sectname, 'q_threshold',      str(10))
    config.set(sectname, 'auto_selection',   'yes')
    config.set(sectname, 'rms_threshold',    str(0.008))
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
    filename = 'LiRES.{}.cfg'.format(input_date)
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
                '([a-zA-Z]+)\-(\d+)s bin\-\-?(\d)')
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
    # find date of data obtained
    current_pathname = os.path.basename(os.getcwd())
    datestr = extract_date(current_pathname)
    if datestr is not None:
        dt = datetime.datetime.strptime(datestr, '%Y-%m-%d')
        date = (dt.year, dt.month, dt.day)
    else:
        date = None


    # load config file
    config = load_config('LiRES\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    statime_key = config['data'].get('statime_key')
    exptime_key = config['data'].get('exptime_key')

    # initialize logtable
    logtable = Table(dtype=[
                #('frameid',     'i2'),
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

    fmt_str = ('  - {:13s} {:5s} {:<12s} {:>7} {:^23s} '
                '{:4} {:5} {:7} {:>7} {:>5}')
    head_str = fmt_str.format('fileid', 'type', 'object', 'exptime',
                'obsdate', 'gain', 'speed', 'binning', 'nsat', 'q95')
    print(head_str)

    # start scanning the raw files
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(r'(\S+)\.fits', fname)
        if not mobj:
            continue
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
    
        fileid  = mobj.group(1)
        exptime = head[exptime_key]
        info    = parse_objectstring(head['OBJECT'])
        obsdate = head[statime_key]

        if re.match(r'bias_?\d+', fileid):
            objectname = 'Bias'
            imgtype    = 'cal'
        elif re.match(r'flat_?\d+', fileid):
            objectname = 'Flat'
            imgtype    = 'cal'
        elif re.match(r'thar_?\d+', fileid):
            objectname = 'ThAr'
            imgtype    = 'cal'
        elif re.match(r'obj(\S*)', fileid):
            objectname = ''
            imgtype    = 'sci'
        else:
            objectname = ''
            imgtype = ''
    
        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()
    
        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))
    
        item = (fileid, imgtype, objectname, exptime, obsdate,
                info['gain'], info['speed'], info['binning'],
                saturation, quantile95)
        logtable.add_row(item)
    
        item = logtable[-1]
    
        # print log item with colors
        string = fmt_str.format(fileid,
                    '({:3s})'.format(imgtype),
                    objectname, exptime, obsdate,
                    info['gain'], info['speed'], info['binning'],
                    saturation, quantile95)
        print(print_wrapper(string, item))

    # resort logtable
    logtable.sort('obsdate')
    frameid_lst = np.arange(1, len(logtable)+1)
    logtable.add_column(frameid_lst, name='frameid', index=0)

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
