import os
import re
import sys
import datetime
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt

from ...utils.misc import extract_date
from ...utils.obslog import read_obslog, write_obslog
from ..common import load_obslog, load_config
from .common import get_sci_region, print_wrapper, plot_time_offset
from .reduce_singlefiber import reduce_singlefiber
from .reduce_doublefiber import reduce_doublefiber

def make_config():
    """Generate a config file for reducing the data taken with Xinglong 2.16m
    HRS.


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

    # select readout mode
    readout_mode_lst = [
                    'Left Top & Bottom',
                    'Left Bottom & Right Top',
                    ]
    default_readout_mode = 'Left Top & Bottom'
    for i, readout_mode in enumerate(readout_mode_lst):
        print(' [{:d}] {:s}'.format(i, readout_mode))
    while(True):
        string = input('Select CCD Readout Mode ({}): '.format(
                        default_readout_mode))
        if string.isdigit() and int(string)<len(readout_mode_lst):
            readout_mode = readout_mode_lst[int(string)]
            break
        elif len(string.strip())==0:
            readout_mode = default_readout_mode
        else:
            print('Invalid selection:', string)
            continue


    direction = {
            'Left Top & Bottom': 'xr-',
            'Left Bottom & Right Top': 'xr-',
            }[readout_mode]

    # general database path for this instrument
    dbpath = '~/.gamse/Xinglong216.HRS'

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    # determine the time-dependent keywords
    if input_datetime < datetime.datetime(2018, 12, 1):
        statime_key = 'DATE-STA'
        exptime_key = 'EXPTIME'
    else:
        # since 2019 there's another type of FITS header
        statime_key = 'DATE-OBS'
        exptime_key = 'EXPOSURE'

    config.set('data', 'telescope',    'Xinglong216')
    config.set('data', 'instrument',   'HRS')
    config.set('data', 'rawpath',      'rawdata')
    config.set('data', 'statime_key',  statime_key)
    config.set('data', 'exptime_key',  exptime_key)
    config.set('data', 'readout_mode', readout_mode)
    config.set('data', 'direction',    direction)
    #config.set('data', 'obsinfo_file', 'obsinfo.txt')
    config.set('data', 'fibermode',    fibermode)
    if fibermode == 'double':
        config.set('data', 'fiberoffset', str(-12))

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

    # section of background correction
    sectname = 'reduce.background'
    config.add_section(sectname)
    config.set(sectname, 'subtract',      'yes')
    config.set(sectname, 'ncols',         str(9))
    config.set(sectname, 'distance',      str(7))
    config.set(sectname, 'yorder',        str(7))
    config.set(sectname, 'database_path', os.path.join(dbpath, 'background'))

    # section of spectra extraction
    sectname = 'reduce.extract'
    config.add_section(sectname)
    config.set(sectname, 'extract', 
            "lambda row: row['imgtype']=='sci' or row['object'].lower()=='i2'")
    config.set(sectname, 'upper_limit', str(7))
    config.set(sectname, 'lower_limit', str(7))

    # write to config file
    filename = 'Xinglong216HRS.{}.cfg'.format(input_date)
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

def parse_idstring(idstring):
    def parse_idstr(_idstr):
        if   len(_idstr)>8: return int(_idstr[8:])
        elif len(_idstr)>4: return int(_idstr[4:])
        else: return int(_idstr)

    if '-' in idstring:
        g = idstring.split('-')
        id1, id2 = g[0], g[1]
        id1 = parse_idstr(id1)
        id2 = parse_idstr(id2)
        return range(id1, id2+1)
    else:
        id1 = parse_idstr(idstring)
        return [id1]

def parse_timestr(timestr, date):
    mobj = re.match('(\d{2}):(\d{2}):(\d{2})', timestr)
    yy, mm, dd = date
    h = int(mobj.group(1))
    m = int(mobj.group(2))
    s = int(mobj.group(3))
    if h >= 24:
        h = h - 24
        dt = datetime.timedelta(days=1)
    else:
        dt = datetime.timedelta(days=0)
    obstime = datetime.datetime(yy, mm, dd, h, m, s) + dt
    return obstime.isoformat()

def parse_logfile_singlefiber(filename, date):

    logtable = Table(dtype=[
                    ('frameid', 'i2'),
                    ('fileid',  'S11'),
                    ('imgtype', 'S3'),
                    ('object',  'S50'),
                    ('exptime', 'f4'),
                    ('obsdate', 'S23'),
            ], masked=True)

    ptn1 = '([a-zA-Z]?[\d\-]+)'                 # for id string
    ptn2 = '([a-zA-Z0-9+-_\[\]\s]+)'            # object name
    ptn3 = '(\d{2}:\d{2}:\d{2})'                # time string
    ptn4 = '([\.\d]+)'                          # exptime
    ptn5 = '(\d{2}:\d{2}:\d{2}\.?\d?\d?)'       # ra
    ptn6 = '([+-]\d{2}:\d{2}:\d{2}\.?\d?\d?)'   # dec

    yy, mm, dd = date
    file1 = open(filename, encoding='gbk')
    for row in file1:
        row = row.strip()

        # match Bias, Flat and ThAr
        is_match = False
        for objname in ['Bias', 'Flat', 'ThAr']:
            pattern = '{}\s*({})\s*{}\s*{}'.format(
                        ptn1, objname.lower(), ptn3, ptn4)
            mobj = re.match(pattern, row.lower())
            if mobj:
                id_lst = parse_idstring(mobj.group(1))
                obstime = parse_timestr(mobj.group(3), date)
                exptime = float(mobj.group(4))
                for iframe, frameid in enumerate(id_lst):
                    fileid  = '{:04d}{:02d}{:02d}{:03d}'.format(
                                yy, mm, dd, frameid)
                    if iframe==0:
                        item = (frameid, fileid, 'cal', objname,
                                    exptime, obstime)
                        mask = (False, False, False, False,
                                    False, False)
                    else:
                        item = (frameid, fileid, 'cal', objname,
                                    exptime, '')
                        mask = (False, False, False, False,
                                    False, True)
                    logtable.add_row(item)
                is_match = True
                break

        if is_match:
            continue

        # match science objects
        pattern = '{}\s*{}\s*{}\s*{}\s*{}\s*{}\s*2000'.format(
                    ptn1, ptn2, ptn3, ptn4, ptn5, ptn6)
        mobj = re.match(pattern, row)
        if mobj:
            id_lst = parse_idstring(mobj.group(1))
            objname = mobj.group(2).strip()
            obstime = parse_timestr(mobj.group(3), date)
            exptime = float(mobj.group(4))
            for iframe, frameid in enumerate(id_lst):
                fileid  = '{:04d}{:02d}{:02d}{:03d}'.format(
                                yy, mm, dd, frameid)
                if iframe==0:
                    item = (frameid, fileid, 'sci', objname,
                            exptime, obstime)
                    mask = (False, False, False, False,
                            False, False)
                else:
                    item = (frameid, fileid, 'sci', objname,
                            exptime, '')
                    mask = (False, False, False, False,
                            False, True)
                logtable.add_row(item)
            continue

        print('Match error:', row)
    file1.close()

    return logtable


def parse_logfile_doublefiber(filename, date):

    logtable = Table(dtype=[
                    ('frameid',     'i2'),
                    ('fileid',      'S11'),
                    ('imgtype',     'S3'),
                    ('object',      'S80'),
                    ('object_A',    'S50'),
                    ('object_B',    'S50'),
                    ('exptime',     'f4'),
                    ('obsdate',     'S23'),
            ], mased=True)

    ptn1 = '([a-zA-Z]?[\d\-]+)'                 # for id string
    ptn2 = '([a-zA-Z0-9+-_\s]+)'            # object name
    ptn3 = '(\d{2}:\d{2}:\d{2})'                # time string
    ptn4 = '([\.\d]+)'                          # exptime
    ptn5 = '(\d{2}:\d{2}:\d{2}\.?\d?\d?)'       # ra
    ptn6 = '([+-]\d{2}:\d{2}:\d{2}\.?\d?\d?)'   # dec

    yy, mm, dd = date
    file1 = open(logfile, encoding='gbk')
    for row in file1:
        row = row.strip()

        # match Bias
        pattern = '{}\s*(bias)\s*{}\s*{}'.format(ptn1, ptn3, ptn4)
        mobj = re.match(pattern, row.lower())
        if mobj:
            id_lst = parse_idstring(mobj.group(1))
            obstime = parse_timestr(mobj.group(3), date)
            exptime = float(mobj.group(4))
            for iframe, frameid in enumerate(id_lst):
                fileid  = '{:04d}{:02d}{:02d}{:03d}'.format(yy, mm, dd, frameid)
                if iframe==0:
                    item = (frameid, fileid, 'cal', 'Bias', '', '',
                            exptime, obstime)
                    mask = (False, False, False, False, False, False,
                            False, False)
                else:
                    item = (frameid, fileid, 'cal', 'Bias', '', '',
                            exptime, '')
                    mask = (False, False, False, False, False, False,
                            False, True)
                logtable.add_row(item)
            continue


        # match other frames
        pattern = '{}\s*\[A\]\s*{}\s*\[B\]\s*{}\s*{}\s*{}'.format(
                ptn1, ptn2, ptn2, ptn3, ptn4)
        mobj = re.match(pattern, row)
        if mobj:
            id_lst = parse_id_string(mobj.group(1))
            objname_A = mobj.group(2).strip()
            objname_B = mobj.group(3).strip()
            if objname_A.lower() in ['flat', 'thar', 'comb', '']:
                imgtype = 'cal'
            else:
                imgtype = 'sci'
            obstime = parse_timestr(mobj.group(4), date)
            exptime = float(mobj.group(5))
            for iframe, frameid in enumerate(id_lst):
                fileid  = '{:04d}{:02d}{:02d}{:03d}'.format(
                                yy, mm, dd, frameid)
                if iframe==0:
                    item = (frameid, fileid, imgtype, '', objname_A, objname_B,
                            exptime, obstime)
                    mask = (False, False, False, False, False, False,
                            False, False)
                else:
                    item = (frameid, fileid, imgtype, '', objname_A, objname_B,
                            exptime, '')
                    mask = (False, False, False, False, False, False,
                            False, True)
                logtable.add_row(item)
            continue

        print('Match error:', row)
    file1.close()

    maxlen_A = max([len(row['object_A']) for row in logtable])
    maxlen_B = max([len(row['object_B']) for row in logtable])
    for row in logtable:
        if row['object']!='Bias':
            row['object'] = '[A] {} [B] {}'.format(
                            row['object_A'].ljust(maxlen_A),
                            row['object_B'].ljust(maxlen_B))
    logtable.remove_column(['object_A','object_B'])

    return logtable



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
    config = load_config('Xinglong216HRS\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    cal_objects = ['bias', 'flat', 'dark', 'i2', 'thar', 'flat;', ';flat',
                    'thar;', ';thar']
    regular_names = ('Bias', 'Flat', 'ThAr', 'I2')

    # read original log file
    # search file in the current folder
    logfile = None
    for fname in os.listdir('./'):
        mobj = re.match('(\d{8})\.txt', fname)
        if mobj:
            logfile = fname
            datestr = mobj.group(1)
            date = (int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
            break
    # logfile not found
    if logfile is None:
        print('Error: could not find log file')
        exit()

    fibermode = config['data']['fibermode']
    if fibermode == 'single':
        logtable = parse_logfile_singlefiber(logfile, date)
    elif fibermode == 'double':
        logtable = parse_logfile_doublefiber(logfile, date)
    message = 'Read log from "{}"'.format(logfile)
    print(message)
    logfilename = 'log.{0:04d}-{1:02d}-{2:02d}.txt'.format(*date)
    logtable.write(logfilename, format='ascii.fixed_width_two_line',
                    overwrite=True)

    maxobjlen = max([len(row['object']) for row in logtable])

    # if the obsinfo file exists, read and pack the information
    addinfo_lst = {}
    obsinfo_file = config['data'].get('obsinfo_file', None)
    has_obsinfo = obsinfo_file is not None and os.path.exists(obsinfo_file)
    if has_obsinfo:
        # has obsinfo file
        # method 1 (deprecated)
        #io_registry.register_reader('obslog', Table, read_obslog)
        #addinfo_table = Table.read(obsinfo_file, format='obslog')

        # method 2 (for 3-line headers, deprecated)
        #addinfo_table = read_obslog(obsinfo_file)

        # method 3 (for normal table)
        addinfo_table = read_obsinfo(obsinfo_file)

        addinfo_lst = {row['frameid']:row for row in addinfo_table}
        # prepare the difference list between real observation time and FITS
        # time
        real_obsdate_lst = []
        delta_t_lst = []

    statime_key = config['data'].get('statime_key')
    exptime_key = config['data'].get('exptime_key')

    nsat_lst = []
    q95_lst  = []

    fmt_str = ('  - {:5s} {:11s} {:5s} ' + '{{:<{}s}}'.format(maxobjlen) +
                #' {:1s}I2 {:>7} {:23s} {:>7} {:>5}')
                ' {:>7} {:23s} {:>7} {:>5}')
    head_str = fmt_str.format('frameid', 'fileid', 'type', 'object', '',
                                'exptime', 'obsdate', 'nsat', 'q95')
    print(head_str)

    # start scanning the raw files
    for logitem in logtable:
        frameid    = logitem['frameid']
        fileid     = logitem['fileid']
        imgtype    = logitem['imgtype']
        objectname = logitem['object']
        exptime    = logitem['exptime']

        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
        data, head = fits.getdata(filename, header=True)

        # get science region
        y1, y2, x1, x2 = get_sci_region(head)
        data = data[y1:y2, x1:x2]

        # get obsdate from FITS header
        obsdate = dateutil.parser.parse(head[statime_key])

        # get exposure time from FITS header
        exptime_fits = head[exptime_key]
        if abs(exptime - exptime_fits)>1.0:
            print('Error: Exposure time do not match: {} - {}'.format(
                    exptime, exptime_fits))

        # for post 2019 data, there's another keyword "EXPOEND" to record
        # exposur end time.
        if 'EXPOEND' in head:
            obsdate2 = dateutil.parser.parse(head['EXPOEND'])
            calc_exptime = (obsdate2-obsdate).seconds

            if abs(calc_exptime - exptime)>5:
                # this usually caused by the error in start time
                print('Warning: TIME difference of {} error: '
                        'EXPOEND - {} = {:.2f}, exptime = {:.2f}'
                        ''.format(fileid, statime_key, calc_exptime, exptime))
                # re-calculate the start time according to the exposure
                # end time and exptime.
                obsdate = obsdate2 - datetime.timedelta(seconds=int(exptime))

        # parse obsdate, and calculate the time offset
        if (frameid in addinfo_lst and 'obsdate' in addinfo_table.colnames
            and addinfo_lst[frameid]['obsdate'] is not np.ma.masked):
            time_str = addinfo_lst[frameid]['obsdate']
            real_obsdate = dateutil.parser.parse(time_str)
            file_obsdate = obsdate
            delta_t = real_obsdate - file_obsdate
            real_obsdate_lst.append(real_obsdate)
            delta_t_lst.append(delta_t.total_seconds())

        # update obsdate into log file
        obsdatestr = obsdate.isoformat()[0:23]
        if len(obsdatestr)==19:
            obsdatestr += '.000'
        logitem['obsdate'] = obsdatestr

        # parse object name
        #if 'OBJECT' in head:
        #    objectname = head['OBJECT'].strip()
        #else:
        #    objectname = ''

        #if (frameid in addinfo_lst and 'object' in addinfo_table.colnames
        #    and addinfo_lst[frameid]['object'] is not np.ma.masked):
        #    objectname = addinfo_lst[frameid]['object']
        #elif frameid in loginfo:
        #    objectname = loginfo[frameid]['object']
        #else:
        #    pass

        ## change to regular name
        #for regname in regular_names:
        #    if objectname.lower() == regname.lower():
        #        objectname = regname
        #        break

        ## parse I2 cell
        #i2 = ('-', '+')[objectname.lower()=='i2']
        #if (frameid in addinfo_lst and 'i2' in addinfo_table.colnames
        #    and addinfo_lst[frameid]['i2'] is not np.ma.masked):
        #    i2 = addinfo_lst[frameid]['i2']

        #imgtype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        nsat_lst.append(saturation)
        q95_lst.append(quantile95)

        #item = [frameid, fileid, imgtype, objectname, i2, exptime,
        #        obsdate.isoformat()[0:23], saturation, quantile95]
        #logtable.add_row(item)
        ## get table Row object. (not elegant!)
        #item = logtable[-1]

        # print log item with colors
        string = fmt_str.format(
                    '[{:d}]'.format(frameid), fileid,
                    '({:3s})'.format(imgtype),
                    objectname, exptime, obsdate.isoformat()[0:23],
                    saturation, quantile95)
        print(print_wrapper(string, logitem))

    # sort by obsdate
    #logtable.sort('obsdate')
    logtable.add_column(nsat_lst, name='saturation')
    logtable.add_column(q95_lst, name='q95')
    logtable['object'].info.format = '%-{}s'.format(maxobjlen)
    logtable.write(logfilename, format='ascii.fixed_width_two_line',
                    overwrite=True)

    exit()
    if has_obsinfo and len(real_obsdate_lst)>0:
        # determine the time offset as median value
        time_offset = np.median(np.array(delta_t_lst))
        time_offset_dt = datetime.timedelta(seconds=time_offset)
        # plot time offset

        # find the filename of time-offset figure
        figpath = config['reduce'].get('figpath')
        if not os.path.exists(figpath):
            os.mkdir(figpath)
        figname = os.path.join(figpath, 'obsdate_offset.png')

        plot_time_offset(real_obsdate_lst, delta_t_lst, time_offset, figname)

        # correct time offset
        for row in logtable:
            # convert to datetime.Datetime object
            dt = dateutil.parser.parse(row['obsdate'])
            # add offset and convert back to string
            obsdate_str = (dt + time_offset_dt).isoformat()[0:23]
            if len(obsdate_str)==19:
                obsdate_str += '.000'
            row['obsdate'] = obsdate_str

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
    #logtable['imgtype'].info.format = '^s'
    #logtable['object'].info.format = '<s'
    #logtable['i2'].info.format = '^s'
    #logtable['exptime'].info.format = 'g'

    # save the logtable
    # method 1: deprecated
    #write_obslog(logtable, outfilename, delimiter='|')
    # method 2: deprecated because cannot specify the formats of each column
    #logtable.write(outfilename, format='ascii.fixed_width_two_line',
    #                delimiter='|')
    # method 3
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

def read_logfile(filename):
    """Read ascii log file.

    Args:
        filename (str): Name of log file.
    Returns:
        dict: 

    """

    loginfo = {}

    pattern1 = '([\d\-]+)'                  # pattern for fileid
    pattern2 = '([a-zA-Z0-9+-_\s]+)'        # pattern for object name
    pattern3 = '(\d{2}:\d{2}:\d{2})'        # pattern for begin time
    pattern4 = '([\d\.]+)'                  # pattern for exptime
    pattern5 = '(\d{2}:\d{2}:\d{2}\.?\d?\d?)'       # pattern for ra
    pattern6 = '([+-]\d{2}:\d{2}:\d{2}\.?\d?\d?)'   # pattern for dec
    pattern_bias = '{}\s*(bias)\s*{}\s*{}'.format(
            pattern1, pattern3, pattern4)
    pattern_thar = '{}\s*(thar)\s*{}\s*{}'.format(
            pattern1, pattern3, pattern4)
    pattern_flat = '{}\s*(flat)\s*{}\s*{}'.format(
            pattern1, pattern3, pattern4)
    pattern_star = '{}\s*{}\s*{}\s*{}\s*{}\s*{}\s*2000'.format(
            pattern1, pattern2, pattern3, pattern4, pattern5, pattern6)

    infile = open(filename, encoding='gbk')
    for row in infile:
        row = row.strip()

        mobj_bias = re.match(pattern_bias, row.lower())
        mobj_thar = re.match(pattern_thar, row.lower())
        mobj_flat = re.match(pattern_flat, row.lower())
        mobj_star = re.match(pattern_star, row)

        if mobj_bias is None and mobj_thar is None and \
            mobj_flat is None and mobj_star is None:
            continue

        for mobj in [mobj_bias, mobj_thar, mobj_flat, mobj_star]:
            if mobj is not None:
                idstring   = mobj.group(1)
                timestr    = mobj.group(3)
                exptimestr = mobj.group(4)
                break

        # parse object name
        if mobj_bias:
            objname = 'Bias'
        elif mobj_thar:
            objname = 'ThAr'
        elif mobj_flat:
            objname = 'Flat'
        elif mobj_star:
            objname = mobj.group(2)
        else:
            raise ValueError

        # parse exptime
        if '.' in exptimestr:
            exptime = float(exptimestr)
        else:
            exptime = int(exptimestr)

        # parse fileid
        id_prefix = int(idstring[0:8])
        yy = int(idstring[0:4])
        mm = int(idstring[4:6])
        dd = int(idstring[6:8])
        # get the range of frameids if there is '-' in idstring
        if '-' in idstring[8:]:
            group = idstring[8:].split('-')
            frameid1 = int(group[0])
            frameid2 = int(group[1])
        else:
            frameid1 = int(idstring[8:])
            frameid2 = frameid1

        # parse begin time of exposure
        m1 = re.match('(\d{2}):(\d{2}):(\d{2})', timestr)
        hour   = int(m1.group(1))
        minute = int(m1.group(2))
        second = int(m1.group(3))
        if hour >= 24:
            hour = hour-24
            dt   = datetime.timedelta(days=1)
        else:
            dt   = datetime.timedelta(days=0)
        obstime = datetime.datetime(yy, mm, dd, hour, minute, second) + dt

        if mobj_star:
            # convert ra string to float in degree
            rastr = mobj.group(5)
            m2 = re.match('(\d{2}):(\d{2}):(\d{2}\.?\d?\d?)', rastr)
            rah = int(m2.group(1))
            ram = int(m2.group(2))
            ras = float(m2.group(3))
            ra = (rah + ram/60 + ras/3600)*15

            # convert dec string to float in degree
            decstr = mobj.group(6)
            m3 = re.match('([+-])(\d{2}):(\d{2}):(\d{2}\.?\d?\d?)', decstr)
            ded = int(m3.group(2))
            dem = int(m3.group(3))
            des = float(m3.group(4))
            dec = ded + dem/60 + des/3600
            if m3.group(1)=='-':
                dec = -dec
        else:
            ra, dec = None, None

        for frameid in range(frameid1, frameid2+1):
            #fileid = int('{}{:03d}'.format(id_prefix, frameid))
            loginfo[frameid] = {
                    'object': objname,
                    'obstime': obstime if frameid==frameid1 else None,
                    'exptime': exptime,
                    'ra': ra,
                    'dec': dec,
                    }

    infile.close()
    return loginfo



def read_obsinfo(filename):
    """Read obsinfo file and convert it to a Astropy table.

    Args:
        filename (str): Filename of obsinfo.
    Returns:
        :class:`astropy.table.Table`: A new logtable.
        
    """
    table1 = Table.read(filename, format='ascii.fixed_width_two_line')

    # create a new table but the datatype of the first column is int
    newdtype = [(item[0], 'i4' if item[0]=='frameid' else item[1])
                    for item in table1.dtype.descr]
    newtable = Table(dtype=newdtype, masked=True)

    for row in table1:
        frameid = row['frameid']
        mask = [v is np.ma.masked for v in row]
        if '-' in frameid:
            # the first element is a range of numbers
            g = frameid.split('-')
            id1 = int(g[0])
            id2 = int(g[1])
            for i in range(id1, id2+1):
                # copy the row and set the first element to an integer
                newrow = [v for v in row]
                newrow[0] = i
                newtable.add_row(newrow, mask=mask)
        else:
            newrow = [v for v in row]
            newrow[0] = int(newrow[0])
            newtable.add_row(newrow, mask=mask)

    return newtable

def reduce_rawdata():
    """2D to 1D pipeline for the High Resolution spectrograph on Xinglong 2.16m
    telescope.
    """

    # read obslog and config
    config = load_config('Xinglong216HRS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    fibermode = config['data']['fibermode']

    if fibermode == 'single':
        reduce_singlefiber(config, logtable)
    elif fibermode == 'double':
        reduce_doublefiber(config, logtable)
    else:
        print('Invalid fibermode:', fibermode)

def plot_spectra1d():
    filename_lst = sys.argv[2:]

    for filename in filename_lst:
        hdu_lst = fits.open(filename)
        spec = hdu_lst[1].data
        hdu_lst.close()

        for row in spec:
            fig = plt.figure(dpi=200, figsize=(12, 8))
            ax = fig.gca()
            ax.plot(row['wavelength'], row['flux_sum'], lw=0.7)
            ax.set_xlim(row['wavelength'].min(), row['wavelength'].max())
            ax.grid(True, lw=0.5, ls='--')
            ax.set_axisbelow(True)
            ax.set_xlabel(u'Wavelength (\xc5)')
            ax.set_ylabel('Flux')
            fig.savefig('{}_order_{:03d}.png'.format(os.path.splitext(filename)[0],
                        row['order']))
            plt.close()

