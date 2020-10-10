import os
import re
import datetime
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date
from ..common import load_config
from .common import get_region_lst, get_std_setup, print_wrapper
from .reduce import reduce_rawdata

def make_config():
    """Generate a config file for reducing the data taken with Subaru/HDS
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
    dbpath = '~/.gamse/Subaru.HDS'

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')
    config.set('data', 'telescope',    'Subaru')
    config.set('data', 'instrument',   'HDS')
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

    # write to config file
    filename = 'HDS.{}.cfg'.format(input_date)
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
    config = load_config('HDS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid',  'i2'),
                        ('fileid1',  'S12'),
                        ('fileid2',  'S12'),
                        ('objtype',  'S10'),
                        ('object',   'S20'),
                        ('i2',       'S1'),
                        ('exptime',  'f4'),
                        ('obsdate',  'S19'),
                        ('bin_1',    'S5'),
                        ('bin_2',    'S5'),
                        ('slitsize', 'S8'),
                        ('setup',    'S7'),
                        ('nsat_1',   'i4'),
                        ('nsat_2',   'i4'),
                        ('q95_1',    'i4'),
                        ('q95_2',    'i4'),
                ])

    fmt_str = ('  - {:>5s} {:12s} {:12s} {:<10s} {:<20s} {:1s}I2 {:>7} {:^23s}'
            ' {:5} {:5}' # bin_1, bin_2
            ' {:>8s} {:<7s}' # slit_wid, slit_len, setup
            ' {:>7} {:>7} {:>5} {:>5}' # nsat_1, nsat_2, q95_1, q95_2
            )
    head_str = fmt_str.format('FID', 'fileid1', 'fileid2', 'objtype', 'object',
                '', 'exptime', 'obsdate', 'bin_1',  'bin_2', 'slitsize',
                'setup', 'nsat_1', 'nsat_2', 'q95_1',  'q95_2')
    
    print(head_str)
    frameid = 0
    # start scanning the raw files
    for fname in sorted(os.listdir(rawpath)):
        if not re.match('HDSA\d{8}\.fits$', fname):
            continue
        # check the both CCD frames are exist
        framenum = int(fname[4:12])
        if framenum % 2 == 1:
            other_fname = 'HDSA{:08d}.fits'.format(framenum+1)
        else:
            other_fname = 'HDSA{:08d}.fits'.format(framenum-1)
        if not os.path.exists(os.path.join(rawpath, other_fname)):
            print('Warning: missing file: {}'.format(other_fname))

        if framenum % 2 == 0:
            continue

        frameid1 = int(fname[4:12])
        frameid2 = frameid1 + 1

        fileid1 = 'HDSA{:08d}'.format(frameid1)
        fileid2 = 'HDSA{:08d}'.format(frameid2)

        fname1 = '{}.fits'.format(fileid1)
        fname2 = '{}.fits'.format(fileid2)

        filename1 = os.path.join(rawpath, fname1)
        data1, head1 = fits.getdata(filename1, header=True)
        region1, region2 = get_region_lst(head1)
        x11, x12, y11, y12 = region1[0]
        x21, x22, y21, y22 = region2[0]
        data1 = np.concatenate(
                (data1[y11:y12, x11:x12], data1[y21:y22, x21:x22]),
                axis=1)

        filename2 = os.path.join(rawpath, fname2)
        data2, head2 = fits.getdata(filename2, header=True)
        region1, region2 = get_region_lst(head2)
        x11, x12, y11, y12 = region1[0]
        x21, x22, y21, y22 = region2[0]
        data2 = np.concatenate(
                (data2[y11:y12, x11:x12], data2[y21:y22, x21:x22]),
                axis=1)

        for key in ['DATA-TYP', 'OBJECT', 'EXPTIME', 'DATE-OBS', 'UT',
                    'SLIT', 'SLT-WID', 'SLT-LEN', 'FILTER01', 'FILTER02',
                    'H_I2CELL', 'H_COLLIM', 'H_CROSSD']:
            if head1[key] != head2[key]:
                print('Warning: {} of {} ({}) and {} ({}) does not match.'.format(
                    key, frameid1, head1[key], frameid2, head2[key]))

        frameid    = frameid + 1
        objtype    = head1['DATA-TYP']
        objectname = head1['OBJECT']
        exptime    = head1['EXPTIME']
        i2         = {'USE': '+', 'NOUSE': '-'}[head1['H_I2CELL']]
        obsdate    = '{}T{}'.format(head1['DATE-OBS'], head1['UT'])
        bin_1      = '({},{})'.format(head1['BIN-FCT1'], head1['BIN-FCT2'])
        bin_2      = '({},{})'.format(head2['BIN-FCT1'], head2['BIN-FCT2'])
        slitsize   = '{:4.2f}x{:3.1f}'.format(head1['SLT-WID'], head1['SLT-LEN'])
        setup1     = get_std_setup(head1)
        setup2     = get_std_setup(head2)
        if setup1 != setup2:
            print('Warning: setup of CCD1 ({}) and CCD2 ({})'
                  'does not match'.format(setup1, setup2))
        setup = setup1
        sat_mask1  = np.isnan(data1)
        sat_mask2  = np.isnan(data2)
        nsat_1     = sat_mask1.sum()
        nsat_2     = sat_mask2.sum()
        data1[sat_mask1] = 66535
        data2[sat_mask2] = 66535
        q95_1      = int(np.round(np.percentile(data1, 95)))
        q95_2      = int(np.round(np.percentile(data2, 95)))

        item = [frameid, fileid1, fileid2, objtype, objectname, i2, exptime,
                obsdate, bin_1, bin_2, slitsize, setup,
                nsat_1, nsat_2, q95_1, q95_2]
        logtable.add_row(item)

        item = logtable[-1]

        # print log item with colors
        string = fmt_str.format('[{:d}]'.format(frameid),
                    fileid1, fileid2, objtype, objectname, i2, exptime,
                    obsdate, bin_1, bin_2, slitsize, setup,
                    '\033[31m{:7d}\033[0m'.format(nsat_1),
                    '\033[34m{:7d}\033[0m'.format(nsat_2),
                    '\033[31m{:5d}\033[0m'.format(q95_1),
                    '\033[34m{:5d}\033[0m'.format(q95_2),
                    )
        print(print_wrapper(string, item))

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
    logtable['objtype'].info.format = '<s'
    logtable['object'].info.format = '<s'
    logtable['i2'].info.format = '^s'
    logtable['exptime'].info.format = 'g'

    # save the logtable
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()
