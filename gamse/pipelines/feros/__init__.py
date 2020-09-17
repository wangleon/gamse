import os
import re
import datetime
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date
from ..common import load_obslog, load_config
from .common import print_wrapper, get_ccd_geometry
from .reduce import reduce_feros

def make_config():
    """Generate a config file for reducing the data taken with FEROS.

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

    direction = 'yr-'

    # general database path for this instrument
    dbpath = '~/.gamse/FEROS'

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')
    config.set('data', 'telescope',    'MPG/ESO-2.2m')
    config.set('data', 'instrument',   'FEROS')
    config.set('data', 'rawpath',      'rawdata')
    config.set('data', 'statime_key',  'OBS-DATE')
    config.set('data', 'exptime_key',  'EXPTIME')
    config.set('data', 'direction',    direction)

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
    config.set(sectname, 'separation', '500:20, 1500:30, 3500:52')
    config.set(sectname, 'filling',    str(0.3))
    config.set(sectname, 'align_deg',  str(2))
    config.set(sectname, 'display',    'no')
    config.set(sectname, 'degree',     str(3))

    # write to config file
    filename = 'FEROS.{}.cfg'.format(input_date)
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
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    """
    # load config file
    config = load_config('FEROS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # scan the raw files
    fname_lst = sorted(os.listdir(rawpath))

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid', 'i4'),
                        ('fileid',  'S23'),
                        ('imgtype', 'S3'),
                        ('datatype','S11'),
                        ('object',  'S15'),
                        ('exptime', 'f4'),
                        ('binning', 'S6'),
                        ('nsat',    'i4'),
                        ('q95',     'i4'),
                ])

    # filename pattern
    pattern = 'FEROS\.\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\.fits'

    # start scanning the raw files
    frameid = 0
    for fname in fname_lst:
        if not re.match(pattern, fname):
            continue
        fileid = fname[6:29]
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        
        obsdate = dateutil.parser.parse(head['DATE-OBS'])
        exptime = head['EXPTIME']

        objectname = head['OBJECT']

        datatype = head['ESO DPR TYPE']

        if datatype.split(',')[0]=='OBJECT':
            imgtype = 'sci'
        else:
            imgtype = 'cal'

        # find the binning factor
        _, _, binx, biny = get_ccd_geometry(head)
        binning = '({:d}, {:d})'.format(binx, binx)

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        item = [frameid, fileid, imgtype, datatype, objectname, exptime,
                binning, saturation, quantile95]
        logtable.add_row(item)
        item = logtable[-1]

        # print log item with colors
        string_lst = [
                '  {:>5s}'.format('[{:d}]'.format(frameid)),
                '  {:23s}'.format(fileid),
                '  ({:3s})'.format(imgtype),
                '  {:11s}'.format(datatype),
                '  {:15s}'.format(objectname),
                '  Texp = {:4g}'.format(exptime),
                '  Binning = {:5s}'.format(binning),
                '  Nsat = {:6d}'.format(saturation),
                '  Q95 = {:5d}'.format(quantile95),
                ]
        string = ''.join(string_lst)
        print(print_wrapper(string, item))

        frameid += 1

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['fileid'][0:10]
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
    logtable['datatype'].info.format = '<s'
    logtable['object'].info.format = '<s'
    logtable['exptime'].info.format = 'g'

    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

def reduce_rawdata():
    """2D to 1D pipeline for FEROS.
    """

    # read obslog and config
    config = load_config('FEROS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    reduce_feros(config, logtable)
