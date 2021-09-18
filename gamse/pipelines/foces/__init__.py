import os
import re
import datetime
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.time  import Time
from astropy.table import Table

from ...utils.misc import extract_date
from ...utils.obslog import write_obslog
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
    config.set('reduce', 'midpath',     'midproc')
    config.set('reduce', 'figpath',     'images')
    config.set('reduce', 'odspath',     'onedspec')
    #config.set('reduce', 'midproc',     'midproc')   # old style
    #config.set('reduce', 'report',      'report')    # old style
    #config.set('reduce', 'onedspec',    'onedspec')   # old style
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
    config.set(sectname, 'separation', '500:26, 1500:15')
    config.set(sectname, 'filling',    str(0.3))
    config.set(sectname, 'align_deg',  str(2))
    config.set(sectname, 'display',    'no')
    config.set(sectname, 'degree',     str(3))

    # section of flat field correction
    sectname = 'reduce.flat'
    config.add_section(sectname)
    config.set(sectname, 'slit_step',       str(128))
    config.set(sectname, 'q_threshold',     str(50))
    config.set(sectname, 'param_deg',       str(7))
    config.set(sectname, 'mosaic_maxcount', str(50000))

    # section of wavelength calibration
    sectname = 'reduce.wlcalib'
    config.add_section(sectname)
    config.set(sectname, 'search_database',  'yes')
    config.set(sectname, 'linelist',         'thar.dat')
    config.set(sectname, 'use_prev_fitpar',  'no')
    config.set(sectname, 'window_size',      str(13))
    config.set(sectname, 'xorder',           str(3))
    config.set(sectname, 'yorder',           str(4))
    # in previous single fiber data, yorder = 4
    config.set(sectname, 'maxiter',          str(6))
    config.set(sectname, 'clipping',         str(2.3))
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
    distance = {'single': 6, 'double': 2}[fibermode]
    config.set(sectname, 'distance',      str(distance))
    config.set(sectname, 'yorder',        str(6))

    # section of spectra extraction
    sectname = 'reduce.extract'
    config.add_section(sectname)
    config.set(sectname, 'upper_limit', str(4.5))
    config.set(sectname, 'lower_limit', str(4.5))

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

def make_obslog():
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.

    """

    # load config file
    config = load_config('FOCES\S*\.cfg$')

    rawpath   = config['data']['rawdata']
    fibermode = config['data']['fibermode']
    
    # standard naming convenction for fileid
    name_pattern1 = '^\d{8}_\d{4}_FOC\d{4}_[A-Za-z0-9]{4}$'
    name_pattern2 = '^fcs_\d{14}$'

    fname_lst = sorted(os.listdir(rawpath))

    # find the maximum length of fileid
    maxlen_fileid = 0
    for fname in fname_lst:
        if fname.endswith('.fits'):
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

    # start scanning the raw files
    prev_frameid = 0
    for fname in fname_lst:
        if not fname.endswith('.fits'):
            continue
        fileid = fname[0:-5]
        filename = os.path.join(rawpath, fname)
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
                # for Bias
                if fileid[22:25]=='BIA':
                    imgtype, obj_lst = 'cal', ['Bias']
                # for Flats:
                elif fileid[22:25]=='FLS':
                    imgtype, obj_lst = 'cal', ['Flat', '']
                elif fileid[22:25]=='FLC':
                    imgtype, obj_lst = 'cal', ['', 'Flat']
                # for ThAr:
                elif fileid[22:26]=='THCS':
                    imgtype, obj_lst = 'cal', ['ThAr', 'ThAr']
                elif fileid[22:25]=='THS':
                    imgtype, obj_lst = 'cal', ['ThAr', '']
                elif fileid[22:25]=='THC':
                    imgtype, obj_lst = 'cal', ['', 'ThAr']
                # for Comb:
                elif fileid[22:26]=='COS0':
                    imgtype, obj_lst = 'cal', ['Comb', '']
                elif fileid[22:26]=='COC0':
                    imgtype, obj_lst = 'cal', ['', 'Comb']
                elif fileid[22:26]=='COCS':
                    imgtype, obj_lst = 'cal', ['Comb', 'Comb']
                # for Science:
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

        # generate the objectname
        # objectname is the string written in the .obslog file
        # screen_objectname is the string displayed in the terminal
        if fibermode == 'single':
            objectname = '{:23s}'.format(objectname)
            screen_objectname = objectname
        elif fibermode == 'double':

            if len(obj_lst)==1:
                # double fiber mode but BIAS
                objectname = '{:^23s}'.format(obj_lst[0])
                screen_objectname = objectname

            elif len(obj_lst)==2:
                objstr_lst = ['{:^10s}'.format(obj_lst[i]) for i in range(2)]
                objectname = '|'.join(objstr_lst)

                # generate the screen_objectname with style of
                # (A) XXXXXXX (B) -------
                objstr_lst = []
                for ifiber in range(2):
                    fiber = chr(ifiber+65)
                    if len(obj_lst[ifiber])==0:
                        objstr = '-'*7
                    else:
                        objstr = '{:7s}'.format(obj_lst[ifiber])
                    objstr = '({:s}) '.format(fiber) + objstr
                    objstr_lst.append(objstr)
                screen_objectname = ' '.join(objstr_lst)

            else:
                print('Warning: length of object_lst ({}) excess the maximum '
                      'number of fibers (2)'.format(len(obj_lst)))
                objectname = '{:^23s}'.format('Error')
                screen_objectnae = objectname
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
        string_lst = [
                '{:3d}'.format(frameid),
                '{:15s}'.format(fileid),
                '({:3s})'.format(imgtype),
                '{:s}'.format(screen_objectname),
                'Texp = {:5g}'.format(exptime),
                '{:23s}'.format(obsdate.isot),
                'Nsat = {:7d}'.format(saturation),
                'Q95 = {:5d}'.format(quantile95),
                ]
        string = '    '.join(string_lst)
        print(print_wrapper(string, item))

        prev_frameid = frameid

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

    logtable['imgtype'].info.format = '^s'
    logtable['exptime'].info.format = 'g'
    # save the logtable
    #logtable.write(filename=outfilename, format='ascii.fixed_width_two_line',
    #                delimiter='|')
    write_obslog(logtable, outfilename, delimiter='|')

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

