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
from ..common import load_obslog, load_config
from .reduce import reduce_rawdata

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

    # write to config file
    filename = 'ESPaDOnS.{}.cfg'.format(input_date)
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

    An ascii file will be generated after running.
    The name of the ascii file is `YYYY-MM-DD.obslog`, where `YYYY-MM-DD` is the
    date of the *first* FITS image in the data folder.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.
    """
    # load config file
    config = load_config('ESPaDOnS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid',  'i2'),
                        ('fileid',   'S8'),
                        ('obstype',  'S10'),
                        ('object',   'S15'),
                        ('exptime',  'f4'),
                        ('obsdate',  'S23'),
                        ('nsat',     'i4'),
                        ('q95',      'i4'),
                        ('runid',    'S6'),
                        ('pi',       'S15'),
                        ('observer', 'S15'),
                ])

    prev_frameid = -1

    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match('(\d{7}[a-z])\.fits', fname)
        if not mobj:
            continue
        fileid  = mobj.group(1)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        frameid = prev_frameid + 1
        obstype    = head['OBSTYPE']
        objectname = head['OBJECT']
        exptime    = head['EXPTIME']
        if objectname == 'Nowhere':
            objectname = ''

        obsdate_str = '{}T{}'.format(head['DATE-OBS'], head['UTC-OBS'])
        obsdate = dateutil.parser.parse(obsdate_str)

        # determine the total number of saturated pixels
        saturation = (data>=head['SATURATE']).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))


        runid = head['RUNID']
        pi    = head['PI_NAME']
        observer = head['OBSERVER']

        item = [frameid, fileid, obstype, objectname, exptime,
                obsdate.isoformat()[0:23], saturation, quantile95,
                runid, pi, observer]
        logtable.add_row(item)
        
        item = logtable[-1]

        prev_frameid = frameid


    # sort by fileid
    logtable.sort('fileid')

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
    logtable['obstype'].info.format = '^s'
    logtable['object'].info.format = '<s'
    logtable['exptime'].info.format = 'g'

    # save the logtable
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

def reduce_rawdata():
    """2D to 1D pipeline for the CFHT/ESPaDOnS.
    """

    # read obslog and config
    config = load_config('ESPaDOnS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    reduce_rawdata(config, logtable)
