import re
import os
import sys
import datetime
import configparser

from astropy.table import Table

from ...utils.misc import extract_date
from ..common import load_obslog, load_config
from .common import get_metadata
from .reduce import reduce_data

def make_metatable(rawpath):

    # prepare metatable
    metatable = Table(dtype=[
                        ('expid',    'i4'),
                        ('fileid',   'S28'),
                        ('category', 'S8'),
                        ('imgtype',  'S13'),
                        ('object',   'S20'),
                        ('ra',       'f8'),
                        ('dec',      'f8'),
                        ('exptime',  'f4'),
                        ('obsdate',  'S23'),
                        ('mode',     'S9'),
                        ('binning',  'S7'),
                        #('biny',     'i4'),
                        #('gain_r',   'f4'),
                        #('gain_b',   'f4'),
                        #('ron_r',    'f4'),
                        #('ron_b',    'f4'),
                        ('progid',   'S30'),
                        ('pi',       'S50'),
                ], masked=True)
    pattern = '(ESPRE\.\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\.fits'
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(pattern, fname)
        if not mobj:
            continue
   
        fileid = mobj.group(1)

        filename = os.path.join(rawpath, fname)
        meta = get_metadata(filename)

        binning = '({}, {})'.format(meta['binx'], meta['biny'])
        #gain    = '{}, {}'.format(meta['gain'][0], meta['gain'][1])
        #ron     = '{}, {}'.format(meta['ron'][0],  meta['ron'][1])

        mask_ra  = (meta['category']=='CALIB' or meta['ra'] is None)
        mask_dec = (meta['category']=='CALIB' or meta['dec'] is None)

        item = [(meta['expid'],     False),
                (fileid,            False),
                (meta['category'],  False),
                (meta['imgtype'],   False),
                (meta['objname'],   False),
                (meta['ra'],        mask_ra),
                (meta['dec'],       mask_dec),
                (meta['exptime'],   False),
                (meta['obsdate'],   False),
                (meta['mode'],      False),
                (binning,           False),
                #(meta['gain'][0],  False),
                #(meta['gain'][1],  False),
                #(meta['ron'][0],   False),
                #(meta['ron'][1],   False),
                (meta['progid'],    False),
                (meta['piname'],    False),
                ]
        value, mask = list(zip(*item))

        metatable.add_row(value, mask=mask)

        # print information
        string_lst = [
                '{:5d}'.format(meta['expid']),
                fileid,
                '{:7s}'.format(meta['category']),
                '{:20s}'.format(meta['imgtype']),
                '{:20s}'.format(meta['objname']),
                ' '*9 if mask_ra  else '{:9.5f}'.format(meta['ra']),
                ' '*9 if mask_dec else '{:9.5f}'.format(meta['dec']),
                '{:6g}'.format(meta['exptime']),
                meta['obsdate'],
                '{:10s}'.format(meta['mode']),
                binning,
                '{:>15s}'.format(meta['progid']),
                meta['piname'],
                ]
        print(' '.join(string_lst))

    format_metatable(metatable)

    return metatable

def split_obslog():
    for fname in sorted(os.listdir(os.curdir)):
        mobj = re.match('ESPRESSO.(\d{4}\-\d{2}\-\d{2}).dat', fname)
        if mobj:
            logfilename = fname
            datestr     = mobj.group(1)
            break
    
    logtable = Table.read(logfilename, format='ascii.fixed_width_two_line')
    '''
    config_lst = {}
    for logitem in logtable:
        config = (logitem['mode'], logitem['binning'])
        if config in config_lst:
            config_lst[config].add_row(logitem)
        else:
            config_lst[config] = Table(logitem)

    for config, tab in config_lst.items():
        mode    = config[0]
        binning = config[1]
        mobj = re.match('\((\d), (\d)\)', binning)
        binx = mobj.group(1)
        biny = mobj.group(2)
        newname = 'ESPRESSO.{}.{}.{}x{}.dat'.format(datestr, mode, binx, biny)
        tab.write(newname, format='ascii.fixed_width_two_line', overwrite=True)
        # count number of calib files and science files
        ncal = (tab['category']=='CALIB').sum()
        nsci = (tab['category']=='SCIENCE').sum()
        print('mode={:12s} binning={}x{} N(Cal)={:3d}, N(Sci)={:3d}'.format(
                mode, binx, biny, ncal, nsci))
    '''
    config_lst = []
    for logitem in logtable:
        config = (logitem['mode'], logitem['binning'])
        if config not in config_lst:
            config_lst.append(config)

    for config in config_lst:
        mode    = config[0]
        binning = config[1]
        mobj = re.match('\((\d), (\d)\)', binning)
        binx = mobj.group(1)
        biny = mobj.group(2)
        newname = 'ESPRESSO.{}.{}.{}x{}.dat'.format(datestr, mode, binx, biny)

        m1 = logtable['mode']==mode
        m2 = logtable['binning']==binning
        m = m1*m2
        newtable = logtable[m]
        format_metatable(newtable)
        newtable.write(newname, format='ascii.fixed_width_two_line',
                overwrite=True)
        # count number of calib files and science files
        ncal = (newtable['category']=='CALIB').sum()
        nsci = (newtable['category']=='SCIENCE').sum()
        print('mode={:12s} binning={}x{} N(Cal)={:3d}, N(Sci)={:3d}'.format(
                mode, binx, biny, ncal, nsci))


def format_metatable(metatable):
    metatable['ra'].info.format='%10.6f'
    metatable['dec'].info.format='%9.5f'
    maxlen = max([len(s) for s in metatable['category']])
    metatable['category'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['imgtype']])
    metatable['imgtype'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['object']])
    metatable['object'].info.format='%-{}s'.format(maxlen)
    metatable['mode'].info.format='%9s'
    metatable['progid'].info.format='%15s'
    #metatable['gain_r'].info.format='%4.2f'
    #metatable['gain_b'].info.format='%4.2f'
    #metatable['ron_r'].info.format='%4.2f'
    #metatable['ron_b'].info.format='%4.2f'    


def make_config():
    """Generate a config file for reducing the data taken with ESPRESSO.

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
    config.set('data', 'telescope',    'VLT')
    config.set('data', 'instrument',   'ESPRESSO')
    config.set('data', 'rawpath',      'rawdata')
    config.set('data', 'statime_key',  'DATE-OBS')
    config.set('data', 'exptime_key',  'EXPTIME')
    config.set('data', 'direction',    'yr+')

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
    #config.set(sectname, 'bias_file',     '${reduce:midpath}/bias.fits')
    config.set(sectname, 'cosmic_clip',   str(10))
    config.set(sectname, 'maxiter',       str(5))
    config.set(sectname, 'smooth',        'yes')
    config.set(sectname, 'smooth_method', 'gaussian')
    config.set(sectname, 'smooth_sigma',  str(3))
    config.set(sectname, 'smooth_mode',   'nearest')

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


    # write to config file
    filename = 'ESPRESSO.{}.cfg'.format(input_date)
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

def reduce_rawdata():
    """2D to 1D pipeline for ESPRESSO."""

    # read obslog and config
    config = load_config('ESPRESSO\S*\.cfg$')
    logtablename = sys.argv[2]
    logtable = load_obslog(logtablename, fmt='astropy')

    reduce_data(config, logtable)
