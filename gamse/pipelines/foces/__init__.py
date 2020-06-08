import os
import re
import datetime
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.time  import Time
from astropy.table import Table

from ...utils.misc import extract_date
from ..common import load_obslog, load_config
from .common import print_wrapper
from .reduce_singlefiber import reduce_singlefiber
from .reduce_doublefiber import reduce_doublefiber

def make_config():
    """Generate a config file for reducing the data taken with FOCES.

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

    # determine the fiber mode
    while(True):
        string = input(
            'The data was obatined with Single fiber or Double fibers? [s/d]:')
        if string == 's':
            fibermode = 'single'
            break
        elif string == 'd':
            fibermode = 'double'
            break
        else:
            print('Invalid input: {}'.format(string))
            continue

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')
    config.set('data', 'telescope',   'Fraunhofer')
    config.set('data', 'instrument',  'FOCES')
    config.set('data', 'rawdata',     'rawdata')
    config.set('data', 'statime_key', 'FRAME')
    config.set('data', 'exptime_key', 'EXPOSURE')
    config.set('data', 'direction',   'xb+')
    config.set('data', 'fibermode',   fibermode)
    if fibermode == 'double':
        config.set('data', 'fiberoffset', str(9))

    config.add_section('reduce')
    config.set('reduce', 'midproc',     'midproc')
    config.set('reduce', 'report',      'report')
    config.set('reduce', 'onedspec',    'onedspec')
    config.set('reduce', 'mode',        'normal')
    config.set('reduce', 'oned_suffix', 'ods')
    config.set('reduce', 'fig_format',  'png')

    config.add_section('reduce.bias')
    config.set('reduce.bias', 'bias_file',     '${reduce:midproc}/bias.fits')
    config.set('reduce.bias', 'cosmic_clip',   str(10))
    config.set('reduce.bias', 'maxiter',       str(5))
    config.set('reduce.bias', 'smooth',        'yes')
    config.set('reduce.bias', 'smooth_method', 'gaussian')
    config.set('reduce.bias', 'smooth_sigma',  str(3))
    config.set('reduce.bias', 'smooth_mode',   'nearest')

    config.add_section('reduce.trace')
    config.set('reduce.trace', 'minimum',    str(8))
    config.set('reduce.trace', 'scan_step',  str(100))
    config.set('reduce.trace', 'separation', '500:26, 1500:15')
    config.set('reduce.trace', 'filling',    str(0.3))
    config.set('reduce.trace', 'align_deg',  str(2))
    config.set('reduce.trace', 'display',    'no')
    config.set('reduce.trace', 'degree',     str(3))

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'slit_step',       str(128))
    config.set('reduce.flat', 'q_threshold',     str(50))
    config.set('reduce.flat', 'param_deg',       str(7))
    config.set('reduce.flat', 'mosaic_maxcount', str(50000))

    config.add_section('reduce.wlcalib')
    config.set('reduce.wlcalib', 'search_database', 'yes')
    config.set('reduce.wlcalib', 'database_path',
                                    '~/.gamse/FOCES/wlcalib')
    config.set('reduce.wlcalib', 'linelist',        'thar.dat')
    config.set('reduce.wlcalib', 'use_prev_fitpar', 'no')
    config.set('reduce.wlcalib', 'window_size',     str(13))
    config.set('reduce.wlcalib', 'xorder',          str(3))
    config.set('reduce.wlcalib', 'yorder',          str(4))
    # in previous single fiber data, yorder = 4
    config.set('reduce.wlcalib', 'maxiter',         str(6))
    config.set('reduce.wlcalib', 'clipping',        str(2.3))
    config.set('reduce.wlcalib', 'q_threshold',     str(10))

    config.add_section('reduce.background')
    config.set('reduce.background', 'subtract', 'yes')
    config.set('reduce.background', 'ncols',    str(9))
    distance = {'single': 6, 'double': 2}[fibermode]
    config.set('reduce.background', 'distance', str(distance))
    config.set('reduce.background', 'yorder',   str(6))
    config.set('reduce.background', 'excluded_frameids', '')
    config.set('reduce.background', 'database_path',
                                    '~/.gamse/FOCES/background')
                                    
    config.add_section('reduce.extract')
    config.set('reduce.extract', 'upper_limit', str(6))
    config.set('reduce.extract', 'lower_limit', str(6))

    # write to config file
    filename = 'FOCES.{}.cfg'.format(input_date)
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

def make_obslog(path):
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.

    Args:
        path (str): Path to the raw FITS files.

    """

    # load config file
    config = load_config('FOCES\S*\.cfg$')

    fibermode = config['data']['fibermode']
    
    # standard naming convenction for fileid
    name_pattern1 = '^\d{8}_\d{4}_FOC\d{4}_[A-Za-z0-9]{4}$'
    name_pattern2 = '^fcs_\d{14}$'

    fname_lst = sorted(os.listdir(path))

    # find the maximum length of fileid
    maxlen_fileid = 0
    for fname in fname_lst:
        if fname[-5:] == '.fits':
            fileid = fname[0:-5]
            maxlen_fileid = max(maxlen_fileid, len(fileid))
    # now the maxlen_fileid is the maximum length of fileid

    # prepare infomation to print
    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid', 'S{:d}'.format(maxlen_fileid)),
        ('imgtype', 'S4'),  ('object', 'S21'),
        ('exptime', 'f4'),  ('obsdate', Time),
        ('nsat',   'i4'),   ('q95',     'i4'),
        ])
    display_columns = ['frameid', 'fileid', 'imgtype', 'object',
                        'exptime', 'obsdate', 'nsat', 'q95']
    pinfo = FormattedInfo(obslog_columns, display_columns)
            
    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    #print(pinfo.get_dtype())
    print(pinfo.get_separator())

    # start scanning the raw files
    prev_frameid = 0
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid = fname[0:-5]
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        # old FOCES data are 3-dimensional arrays
        if data.ndim == 3:
            scidata = data[0, 20:-20]
        else:
            scidata = data[:,20:-20]
            
        obsdate = Time(head['FRAME'])
        exptime = head['EXPOSURE']
        target  = 'Unknown'
        if 'PROJECT' in head: target = str(head['PROJECT'])[:10]
        if 'OBJECT'  in head: target = str(head['OBJECT'])[:10]

        if re.match(name_pattern1, fileid):
            # fileid matches the standard FOCES naming convention
            if fibermode == 'single':
                # in single-fiber mode
                if fileid[22:25]=='BIA':
                    imgtype, objectname = 'cal', 'Bias'
                elif fileid[22:25]=='FLA':
                    imgtype, objectname = 'cal', 'Flat'
                elif fileid[22:25]=='THA':
                    imgtype, objectname = 'cal', 'ThAr'
                elif fileid[22:25]=='SCI':
                    imgtype, objectname = 'sci', target
                else:
                    imgtype, objectname = 'cal', 'Unknown'
            elif fibermode == 'double':
                # in double-fiber mode
                if fileid[22:25]=='BIA':
                    imgtype, obj_lst = 'cal', ['Bias']
                elif fileid[22:25]=='FLS':
                    imgtype, obj_lst = 'cal', ['Flat', '']
                elif fileid[22:25]=='FLC':
                    imgtype, obj_lst = 'cal', ['', 'Flat']
                elif fileid[22:26]=='COCS':
                    imgtype, obj_lst = 'cal', ['Comb', 'Comb']
                elif fileid[22:26]=='COC0':
                    imgtype, obj_lst = 'cal', ['', 'Comb']
                elif fileid[22:26]=='COS0':
                    imgtype, obj_lst = 'cal', ['Comb', '']
                elif fileid[22:25]=='THS':
                    imgtype, obj_lst = 'cal', ['ThAr', 'ThAr']
                elif fileid[22:25]=='THC':
                    imgtype, obj_lst = 'cal', ['', 'ThAr']
                elif fileid[22:25]=='THS':
                    imgtype, obj_lst = 'cal', ['ThAr', '']
                elif fileid[22:26]=='SCI0':
                    imgtype, obj_lst = 'sci', [target, '']
                elif fileid[22:26]=='SCC2':
                    imgtype, obj_lst = 'sci', [target, 'Comb']
                elif fileid[22:26]=='SCT2':
                    imgtype, obj_lst = 'sci', [target, 'ThAr']
                else:
                    imgtype, obj_lst = 'cal', ['','']
            else:
                print('Unknown fiber mode: {}'.format(fibermode))
                raise ValueError

            frameid = int(fileid[9:13])
            has_frameid = True
        elif re.match(name_pattern2, fileid):
            frameid = prev_frameid + 1
            if fibermode == 'single':
                imgtype, objectname = 'cal', 'Unknown'
            elif fibermode == 'double':
                imgtype, obj_lst = 'cal', ['', '']
            else:
                print('Unknown fiber mode: {}'.format(fibermode))
                raise ValueError
            has_frameid = True
        else:
            # fileid does not follow the naming convetion
            if fibermode == 'single':
                imgtype, objectname = 'cal', 'Unknown'
            elif fibermode == 'double':
                imgtype, obj_lst = 'cal', ['', '']
            else:
                print('Unknown fiber mode: {}'.format(fibermode))
                raise ValueError
            frameid = 0
            has_frameid = False

        if fibermode == 'single':
            objectname = '{:21s}'.format(objectname)
        elif fibermode == 'double':
            if len(obj_lst)==1:
                objectname = '{:^21s}'.format(obj_lst[0])
            elif len(obj_lst)==2:
                objectname = '|'.join(['{:^10s}'.format(v) for v in obj_lst])
            else:
                print('Warning: length of object_lst ({}) excess the maximum '
                      'number of fibers (2)'.format(len(obj_lst)))
                objectname = '{:^21s}'.format('Error')
                pass
        else:
            print('Unknown fiber mode: {}'.format(fibermode))
            raise ValueError

        # determine the total number of saturated pixels
        saturation = (data>=63000).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        item = [frameid, fileid, imgtype, objectname, exptime, obsdate,
                saturation, quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string = pinfo.get_format(has_esc=False).format(item)
        print(print_wrapper(string, item))

        prev_frameid = frameid

    print(pinfo.get_separator())

    logtable.sort('obsdate')

    if not has_frameid:
        # allocate frameid
        prev_frameid = -1
        for item in logtable:
            frameid = prev_frameid + 1
            item['frameid'] = frameid
            prev_frameid = frameid

    # determine filename of logtable.
    # use the obsdate of the second frame. Here assume total number of files>2
    obsdate = logtable[1]['obsdate'].iso[0:10]
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
    loginfo = FormattedInfo(obslog_columns)
    outfile = open(outfilename, 'w')
    outfile.write(loginfo.get_title(delimiter='|')+os.linesep)
    outfile.write(loginfo.get_dtype(delimiter='|')+os.linesep)
    outfile.write(loginfo.get_separator(delimiter='+')+os.linesep)
    fmtstr = loginfo.get_format(has_esc=False, delimiter='|')
    for row in logtable:
        outfile.write(fmtstr.format(row)+os.linesep)
    outfile.close()


def reduce_rawdata():
    """2D to 1D pipeline for FOCES on the 2m Fraunhofer Telescope in Wendelstein
    Observatory.
    """

    # read obslog and config
    config = load_config('FOCES\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$')

    fibermode = config['data']['fibermode']

    if fibermode == 'single':
        reduce_singlefiber(config, logtable)
    elif fibermode == 'double':
        reduce_doublefiber(config, logtable)
    else:
        print('Invalid fibermode:', fibermode)

