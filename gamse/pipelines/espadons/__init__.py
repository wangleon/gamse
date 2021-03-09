import os
import re
import sys
import datetime
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date

def make_config():
    """Generate a config file for reducing the data taken with CFHT/ESPaDOnS.


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
    dbpath = '~/.gamse/CFHT.ESPaDOnS'

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    config.set('data', 'telescope',    'CFHT')
    config.set('data', 'instrument',   'ESPaDOnS')
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
    config.set(sectname, 'bias_file',   '${reduce:midpath}/bias.fits')
    config.set(sectname, 'cosmic_clip', str(10))
    config.set(sectname, 'maxiter',     str(5))

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
    config = load_config('ESPaDOnS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # scan the raw files
    fname_lst = sorted(os.listdir(rawpath))

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid', 'i2'),
                ])

    for fname in fname_lst:
        if not fname.endswith('.fits'):
            continue
        fileid  = fname[0:-5]
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)


def reduce_rawdata():
    """2D to 1D pipeline for the CFHT/ESPaDOnS.
    """

    # read obslog and config
    config = load_config('ESPaDOnS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')
