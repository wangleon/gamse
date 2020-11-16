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
    """Generate a config file for reducing the data taken with LAMOST-HR
    spectrograph.
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

    # general database path for this instrument
    dbpath = '~/.gamse/LAMOST.LHRS'

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    config.set('data', 'telescope',    'LAMOST')
    config.set('data', 'instrument',   'HRS')
    config.set('data', 'rawpath',      'rawdata')

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

    # section of order trace
    sectname = 'reduce.trace'
    config.add_section(sectname)
    config.set(sectname, 'trace_file', '${reduce:midpath}/trace.fits')
    config.set(sectname, 'minimum',    str(1e-3))
    config.set(sectname, 'scan_step',  str(100))

    if input_datetime > datetime.datetime(2020, 9, 30):
        separation = '500:38, 2000:55, 3800:95'
    else:
        separation = '100:95, 2000:55, 3700:24'
    config.set(sectname, 'separation', separation)
    config.set(sectname, 'filling',    str(0.2))
    config.set(sectname, 'align_deg',  str(3))
    config.set(sectname, 'display',    'no')
    config.set(sectname, 'degree',     str(3))

    # section of wavelength calibration
    sectname = 'reduce.wlcalib'
    config.add_section(sectname)
    config.set(sectname, 'search_database',  'yes')
    config.set(sectname, 'database_path',    os.path.join(dbpath, 'wlcalib'))
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

    # section of spectra extraction
    sectname = 'reduce.extract'
    config.add_section(sectname)
    config.set(sectname, 'upper_limit', str(7))
    config.set(sectname, 'lower_limit', str(7))

    # write to config file
    filename = 'LHRS.{}.cfg'.format(input_date)
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

def make_obslog():
    """Scan the raw data, and generate a log file containing the detail
    information for each frame.

    """
    # load config file
    config = load_config('LHRS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid',  'i2'),
                        ('fileid',   'S12'),
                        ('imgtype',  'S3'),
                        ('object',   'S12'),
                        ('exptime',  'f4'),
                        ('obsdate',  'S19'),
                        ('nsat',     'i4'),
                        ('q95',      'i4'),
                ])

    fmt_str = '  - {:11s} {:5s} {:<12s} {:>7} {:^23s} {:>7} {:>5}'
    head_str = fmt_str.format('fileid', 'type', 'object', 'exptime',
                                'obsdate', 'nsat', 'q95')
    print(head_str)

    # start scanning the raw files
    for fname in sorted(os.listdir(rawpath)):
        if not fname.endswith('.fit'):
            continue
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        frameid    = 0  # frameid will be determined later
        fileid     = fname[0:-4]
        exptime    = head['EXPOSURE']
        objectname = ""
        obsdate    = head['DATE-OBS']
        imgtype    = 'cal'

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        item = [frameid, fileid, imgtype, objectname, exptime, obsdate,
                saturation, quantile95]
        logtable.add_row(item)

        item = logtable[-1]

        # print log item with colors
        string = fmt_str.format(fileid,
                    '({:3s})'.format(imgtype), objectname, exptime,
                    obsdate, saturation, quantile95)
        print(print_wrapper(string, item))

    # sort by obsdate
    logtable.sort('obsdate')

    # allocate frameid
    prev_frameid = -1
    for logitem in logtable:
        frameid = prev_frameid + 1
        logitem['frameid'] = frameid
        prev_frameid = frameid

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'][0:10]
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

    # set display formats
    logtable['imgtype'].info.format = '^s'
    logtable['fileid'].info.format = '<s'
    logtable['object'].info.format = '<s'
    logtable['exptime'].info.format = 'g'

    # save the logtable
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

