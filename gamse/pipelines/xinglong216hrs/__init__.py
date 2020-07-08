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

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    # determine the time-dependent keywords
    if input_datetime < datetime.datetime(2019, 1, 1):
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
    config.set('data', 'obsinfo_file', 'obsinfo.txt')
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

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'slit_step',       str(256))
    config.set('reduce.flat', 'q_threshold',     str(50))
    config.set('reduce.flat', 'mosaic_maxcount', str(50000))

    # section of wavelength calibration
    sectname = 'reduce.wlcalib'
    config.add_section(sectname)
    config.set(sectname, 'search_database',  'yes')
    config.set(sectname, 'database_path',
                                    '~/.gamse/Xinglong216.HRS/wlcalib')
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
    config.set(sectname, 'group_continuous', 'yes')
    config.set(sectname, 'time_diff',        str(120))

    config.add_section('reduce.background')
    config.set('reduce.background', 'ncols',    str(9))
    config.set('reduce.background', 'distance', str(7))
    config.set('reduce.background', 'yorder',   str(7))

    config.add_section('reduce.extract')
    config.set('reduce.extract', 'extract', 
            "lambda row: row['imgtype']=='sci' or row['object'].lower()=='i2'")
    config.set('reduce.extract', 'upper_limit', str(7))
    config.set('reduce.extract', 'lower_limit', str(7))

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
    

def make_obslog():
    """Scan the raw data, and generated a log file containing the detail
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

    # scan filenames and determine the maximum length of fileid
    maxlen_fileid = max([len(fname[0:-5])
                    for fname in os.listdir(rawpath)
                    if fname.endswith('.fits')])

    cal_objects = ['bias', 'flat', 'dark', 'i2', 'thar']
    regular_names = ('Bias', 'Flat', 'ThAr', 'I2')

    # if the obsinfo file exists, read and pack the information
    addinfo_lst = {}
    obsinfo_file = config['data'].get('obsinfo_file')
    has_obsinfo = os.path.exists(obsinfo_file)
    if has_obsinfo:
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

        # find maximum length of object
        maxobjlen = max([len(row['object']) for row in addinfo_table])

    # scan the raw files
    fname_lst = sorted(os.listdir(rawpath))

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid', 'i2'),
                        ('fileid',  'S{:d}'.format(maxlen_fileid)),
                        ('imgtype', 'S3'),
                        ('object',  'S{:d}'.format(maxobjlen)),
                        ('i2',      'S1'),
                        ('exptime', 'f4'),
                        ('obsdate', 'S19'),
                        ('nsat',    'i4'),
                        ('q95',     'i4'),
                ])

    prev_frameid = -1
    # start scanning the raw files
    for fname in fname_lst:
        if not fname.endswith('.fits'):
            continue
        fileid  = fname[0:-5]
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        # get science region
        y1, y2, x1, x2 = get_sci_region(head)
        data = data[y1:y2, x1:x2]

        # find frameid for different name conventions
        if re.match('^\d{11}$', fileid):
            frameid = int(fileid[8:])
        elif re.match('^\d{12}$', fileid):
            frameid = int(fileid[8:])
        elif re.match('^HRS\d{11}$', fileid):
            frameid = int(fileid[11:])
        else:
            print('Error: unknown FileID: {}'.format(fileid))
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(
                    frameid, prev_frameid))

        # get obsdate from FITS header
        statime_key = config['data'].get('statime_key')
        obsdate = dateutil.parser.parse(head[statime_key])
        # parse obsdate, and calculate the time offset
        if (frameid in addinfo_lst and 'obsdate' in addinfo_table.colnames
            and addinfo_lst[frameid]['obsdate'] is not np.ma.masked):
            time_str = addinfo_lst[frameid]['obsdate']
            real_obsdate = dateutil.parser.parse(time_str)
            file_obsdate = obsdate
            delta_t = real_obsdate - file_obsdate
            real_obsdate_lst.append(real_obsdate)
            delta_t_lst.append(delta_t.total_seconds())

        # get exposure time from FITS header
        exptime_key = config['data'].get('exptime_key')
        exptime = head[exptime_key]

        # parse object name
        if 'OBJECT' in head:
            objectname = head['OBJECT'].strip()
        else:
            objectname = ''
        if (frameid in addinfo_lst and 'object' in addinfo_table.colnames
            and addinfo_lst[frameid]['object'] is not np.ma.masked):
            objectname = addinfo_lst[frameid]['object']

        # change to regular name
        for regname in regular_names:
            if objectname.lower() == regname.lower():
                objectname = regname
                break

        # parse I2 cell
        i2 = ('-', '+')[objectname.lower()=='i2']
        if (frameid in addinfo_lst and 'i2' in addinfo_table.colnames
            and addinfo_lst[frameid]['i2'] is not np.ma.masked):
            i2 = addinfo_lst[frameid]['i2']

        imgtype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = int(np.round(np.percentile(data, 95)))

        item = [frameid, fileid, imgtype, objectname, i2, exptime, obsdate,
                saturation, quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string_lst = [
                '  {:>5s}'.format('[{:d}]'.format(frameid)),
                '  {:11s}'.format(fileid),
                '  ({:3s})'.format(imgtype),
                '  {:s}'.format(objectname.ljust(maxobjlen)),
                '  {:1s}I2'.format(i2),
                '  Texp = {:4g}'.format(exptime),
                '  {:23s}'.format(obsdate.isoformat()),
                '  Nsat = {:6d}'.format(saturation),
                '  Q95 = {:5d}'.format(quantile95),
                ]
        string = ''.join(string_lst)
        print(print_wrapper(string, item))

        prev_frameid = frameid

    # sort by obsdate
    #logtable.sort('obsdate')

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
            row['obsdate'] = (dt + time_offset_dt).isoformat()

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
    logtable['object'].info.format = '<s'
    logtable['i2'].info.format = '^s'
    logtable['exptime'].info.format = 'g'

    # save the logtable
    # method 1: deprecated
    #write_obslog(logtable, outfilename, delimiter='|')
    # method 2: deprecated because cannot specify the formats of each column
    #logtable.write(outfilename, format='ascii.fixed_width_two_line',
    #                delimiter='|')
    # methhod 3
    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()

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
    logtable = load_obslog('\S*\.obslog$', format='astropy')

    fibermode = config['data']['fibermode']

    if fibermode == 'single':
        reduce_singlefiber(config, logtable)
    elif fibermode == 'double':
        reduce_doublefiber(config, logtable)
    else:
        print('Invalid fibermode:', fibermode)
