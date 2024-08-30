import os
import re
import datetime
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date
from ...utils.obslog import read_obslog, write_obslog
from ..common import load_obslog, load_config
from .common import get_metadata, print_wrapper, parse_3ccd_images
from .reduce_pre2004 import reduce_pre2004
from .reduce_post2004 import reduce_post2004


def make_metatable(rawpath, verbose=True):
    # prepare metatable
    metatable = Table(dtype=[
                        ('fileid',   str),
                        ('imgtype',  str),
                        ('frameno',  int),
                        ('target',   str),
                        ('ra',       str),
                        ('dec',      str),
                        ('equinox',  str),
                        ('exptime',  float),
                        ('i2',       str),
                        ('obsdate',  str),
                        ('deck',     str),
                        ('snr',      float),
                        ('progid',   str),
                        ('progpi',   str),
                ], masked=True)
    pattern = '(HI\.\d{8}\.\d{5}\.?\d?\d?)\.fits'
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(pattern, fname)
        if not mobj:
            continue

        fileid = mobj.group(1)
        filename = os.path.join(rawpath, fname)
        meta = get_metadata(filename)

        mask_ra  = meta['imgtype']!='object'
        mask_dec = meta['imgtype']!='object'
        mask_equ = meta['imgtype']!='object'

        item = [
                (fileid,            False),
                (meta['imgtype'],   False),
                (meta['frameno'],   False),
                (meta['target'],    False),
                (meta['ra'],        mask_ra),
                (meta['dec'],       mask_dec),
                (meta['equinox'],   mask_equ),
                (meta['exptime'],   False),
                (meta['i2'],        False),
                (meta['obsdate'],   False),
                (meta['deck'],      False),
                (meta['snr'],       False),
                (meta['progid'],    False),
                (meta['progpi'],    False),
                ]
        value, mask = list(zip(*item))

        metatable.add_row(value, mask=mask)

        if verbose:
            # print information
            string_lst = [
                fileid,
                '{:5d}'.format(meta['frameno']),
                '{:>12s}'.format(meta['imgtype']),
                # the longest word in imgtype is 'dark_lamp_on'
                '{:26s}'.format(meta['target']),
                ' '*11 if mask_ra  else '{:11s}'.format(meta['ra']),
                ' '*11 if mask_dec else '{:11s}'.format(meta['dec']),
                '{:6g}'.format(meta['exptime']),
                '{:>3s}'.format(meta['i2']),
                meta['obsdate'],
                meta['deck'],
                '{:6g}'.format(meta['snr']),
                '{:>6s}'.format(meta['progid']),
                meta['progpi'],
                ]
            print(' '.join(string_lst))

    format_metatable(metatable)

    return metatable


def format_metatable(metatable):
    #metatable['ra'].info.format='%10.6f'
    #metatable['dec'].info.format='%9.5f'

    maxlen = max([len(s) for s in metatable['imgtype']])
    metatable['imgtype'].info.format='%-{}s'.format(maxlen)
    #metatable['imgtype'].info.format='{:>8s}'

    maxlen = max([len(s) for s in metatable['target']])
    metatable['target'].info.format='%-{}s'.format(maxlen)

    metatable['exptime'].info.format='%6g'
    metatable['snr'].info.format='%6g'


def make_config():
    """Generate a config file for reducing the data taken with Keck/HIRES.


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
    config.set('data', 'rawpath',     'rawdata')
    #config.set('data', 'statime_key', statime_key)
    #config.set('data', 'exptime_key', exptime_key)

    config.add_section('reduce')
    config.set('reduce', 'midpath',     'midproc')
    config.set('reduce', 'figpath',     'images')
    config.set('reduce', 'odspath',     'onedspec')
    config.set('reduce', 'mode',        'normal')
    config.set('reduce', 'oned_suffix', 'ods')
    config.set('reduce', 'fig_format',  'png')
    config.set('reduce', 'ncores',      'max')
    
    config.add_section('reduce.bias')
    config.set('reduce.bias', 'bias_file',     '${reduce:midpath}/bias.fits')
    config.set('reduce.bias', 'cosmic_clip',   str(10))
    config.set('reduce.bias', 'maxiter',       str(5))
    config.set('reduce.bias', 'smooth',        'no')
    #config.set('reduce.bias', 'smooth_method', 'gaussian')
    #config.set('reduce.bias', 'smooth_sigma',  str(3))
    #config.set('reduce.bias', 'smooth_mode',   'nearest')

    config.add_section('reduce.trace')
    config.set('reduce.trace', 'minimum',    str(1e-3))
    config.set('reduce.trace', 'scan_step',  str(100))
    config.set('reduce.trace', 'separation', '100:84, 1500:45, 3000:14')
    config.set('reduce.trace', 'filling',    str(0.2))
    config.set('reduce.trace', 'align_deg',  str(2))
    config.set('reduce.trace', 'display',    'no')
    config.set('reduce.trace', 'degree',     str(4))
    config.set('reduce.trace', 'file',       '${reduce:midpath}/trace.fits')

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'flat_file', '${reduce:midpath}/flat.fits')

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
    


def make_obslog():
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.
    """

    # load config file
    config = load_config('HIRES\S*\.cfg$')

    rawpath = config['data']['rawpath']

    name_pattern = '^HI\.\d{8}\.\d{5}\.fits$'

    # scan the raw files
    fname_lst = sorted(os.listdir(rawpath))

    # prepare logtable
    logtable = Table(dtype=[
                    ('frameid',  'i2'),
                    ('fileid',   'S17'),
                    ('imgtype',  'S3'),
                    ('object',   'S20'),
                    ('i2',       'S1'),
                    ('exptime',  'f4'),
                    ('obsdate',  'S23'),
                    ('deckname', 'S2'),
                    ('filter1',  'S5'),
                    ('filter2',  'S5'),
                    ('nsat_1',   'i4'),
                    ('nsat_2',   'i4'),
                    ('nsat_3',   'i4'),
                    ('q95_1',    'i4'),
                    ('q95_2',    'i4'),
                    ('q95_3',    'i4'),
                ])
    fmt_str = ('  - {:>5s} {:17s} {:5s} {:<20s} {:1s}I2 {:>7} {:^23s}'
                ' {:2s} {:5s} {:5s}' # deckname, filter1, filter2
                ' \033[34m{:8}\033[0m' # nsat_1
                ' \033[32m{:8}\033[0m' # nsat_2
                ' \033[31m{:8}\033[0m' # nsat_3
                ' \033[34m{:5}\033[0m' # q95_1
                ' \033[32m{:5}\033[0m' # q95_2
                ' \033[31m{:5}\033[0m' # q95_3
                )
    head_str = fmt_str.format('FID', 'fileid', 'imgtype', 'object', '',
                'exptime', 'obsdate', 'deckname', 'filter1', 'filter2',
                'nsat_1', 'nsat_2', 'nsat_3', 'q95_1', 'q95_2', 'q95_3')

    print(head_str)

    # start scanning the raw files
    prev_frameid = -1
    for fname in fname_lst:
        if not re.match(name_pattern, fname):
            continue
        fileid = fname[0:17]
        filename = os.path.join(rawpath, fname)
        hdu_lst = fits.open(filename)

        # parse images
        if len(hdu_lst)==4:
            # post-2004 data
            data_lst, mask_lst = parse_3ccd_images(hdu_lst)
        else:
            # pre-2004 data
            pass


        head0 = hdu_lst[0].header

        frameid = prev_frameid + 1

        # get obsdate in 'YYYY-MM-DDTHH:MM:SS' format
        date = head0.get('DATE-OBS')
        utc  = head0.get('UTC', head0.get('UT'))
        obsdate = '{}T{}'.format(date, utc)

        exptime  = head0.get('ELAPTIME')
        i2in     = head0.get('IODIN', False)
        i2out    = head0.get('IODOUT', True)
        i2       = ('-', '+')[i2in]
        imagetyp = head0.get('IMAGETYP')
        targname = head0.get('TARGNAME', '')
        lampname = head0.get('LAMPNAME', '')

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
        q95_lst = [int(np.round(np.percentile(data, 95)))
                            for data in data_lst]
        q95_1, q95_2, q95_3 = q95_lst

        # close the fits file
        hdu_lst.close()

        item = [frameid, fileid, imgtype, objectname,
                i2, exptime, obsdate,
                deckname, filter1, filter2,
                nsat_1, nsat_2, nsat_3,
                q95_1, q95_2, q95_3]

        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string = fmt_str.format('[{:3d}]'.format(frameid),
                    fileid, '({:3s})'.format(imgtype),
                    objectname, i2, exptime, obsdate,
                    deckname, filter1, filter2,
                    nsat_1, nsat_2, nsat_3, q95_1, q95_2, q95_3,
                    )
        print(print_wrapper(string, item))

        prev_frameid = frameid

    #print(pinfo.get_separator())

    # sort by obsdate
    #logtable.sort('obsdate')

    # determine filename of logtable.
    # use the obsdate of the LAST frame.
    obsdate = logtable[-1]['obsdate'][0:10]
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

    logtable['imgtype'].info.format = '^s'
    logtable['object'].info.format = '<s'
    logtable['i2'].info.format = '^s'
    logtable['exptime'].info.format = 'g'
    logtable['deckname'].info.format = '^s'
    logtable['filter1'].info.format = '^s'
    logtable['filter2'].info.format = '^s'

    #outfile.write(loginfo.get_title()+os.linesep)
    #outfile.write(loginfo.get_dtype()+os.linesep)
    #outfile.write(loginfo.get_separator()+os.linesep)
    #for row in logtable:
    #    outfile.write(loginfo.get_format(has_esc=False).format(row)+os.linesep)
    #outfile.close()

    #write_obslog(logtable, outfilename, delimiter='|')

    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()


def reduce_rawdata():
    """2D to 1D pipeline for Keck/HIRES.
    """
    # read obslog and config
    config = load_config('HIRES\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    obsdate = dateutil.parser.parse(logtable[0]['obsdate'])

    if obsdate < datetime.datetime(2004, 8, 17, 0, 0, 0):
        reduce_pre2004(config, logtable)
    else:
        reduce_post2004(config, logtable)

