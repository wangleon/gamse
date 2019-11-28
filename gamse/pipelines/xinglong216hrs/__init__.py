import os
import re
import datetime
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time
from astropy.table import Table

from ...echelleutils.misc import extract_date
from ...utils.obslog import read_obslog
from ..common import FormattedInfo
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

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')

    # determine the time-dependent keywords
    if input_datetime > datetime.datetime(2009, 1, 1):
        # since 2019 there's another type of FITS header
        statime_key = 'DATE-OBS'
        exptime_key = 'EXPOSURE'
    else:
        statime_key = 'DATE-STA'
        exptime_key = 'EXPTIME'

    config.set('data', 'telescope',   'Xinglong216')
    config.set('data', 'instrument',  'HRS')
    config.set('data', 'rawdata',     'rawdata')
    config.set('data', 'statime_key', statime_key)
    config.set('data', 'exptime_key', exptime_key)
    config.set('data', 'direction',   'xr-')
    config.set('data', 'fibermode',   fibermode)
    if fibermode == 'double':
        config.set('data', 'fiberoffset', str(-12))

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
    config.set('reduce.trace', 'separation', '500:21, 3500:52')
    config.set('reduce.trace', 'filling',    str(0.3))
    config.set('reduce.trace', 'align_deg',  str(2))
    config.set('reduce.trace', 'display',    'no')
    config.set('reduce.trace', 'degree',     str(3))

    config.add_section('reduce.flat')
    config.set('reduce.flat', 'slit_step',       str(256))
    config.set('reduce.flat', 'q_threshold',     str(50))
    config.set('reduce.flat', 'mosaic_maxcount', str(50000))

    config.add_section('reduce.wlcalib')
    config.set('reduce.wlcalib', 'search_database', 'yes')
    config.set('reduce.wlcalib', 'database_path',
                                    '~/.gamse/Xinglong216.HRS/wlcalib')
    config.set('reduce.wlcalib', 'linelist',        'thar.dat')
    config.set('reduce.wlcalib', 'use_prev_fitpar', 'yes')
    config.set('reduce.wlcalib', 'window_size',     str(13))
    config.set('reduce.wlcalib', 'xorder',          str(3))
    config.set('reduce.wlcalib', 'yorder',          str(3))
    config.set('reduce.wlcalib', 'maxiter',         str(5))
    config.set('reduce.wlcalib', 'clipping',        str(3))
    config.set('reduce.wlcalib', 'q_threshold',     str(10))

    config.add_section('reduce.background')
    config.set('reduce.background', 'ncols',    str(9))
    config.set('reduce.background', 'distance', str(7))
    config.set('reduce.background', 'yorder',   str(7))

    config.add_section('reduce.extract')
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
    

def make_obslog(path):
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running.
    The name of the ascii file is `YYYY-MM-DD.obslog`, where `YYYY-MM-DD` is the
    date of the *first* FITS image in the data folder.
    If the file name already exists, `YYYY-MM-DD.1.obslog`,
    `YYYY-MM-DD.2.obslog` ... will be used as substituions.

    Args:
        path (str): Path to the raw FITS files.

    """
    cal_objects = ['bias', 'flat', 'dark', 'i2', 'thar']
    regular_names = ('Bias', 'Flat', 'ThAr', 'I2')

    # if the obsinfo file exists, read and pack the information
    addinfo_lst = {}
    obsinfo_file = 'obsinfo.txt'
    has_obsinfo = os.path.exists(obsinfo_file)
    if has_obsinfo:
        #io_registry.register_reader('obslog', Table, read_obslog)
        #addinfo_table = Table.read(obsinfo_file, format='obslog')
        addinfo_table = read_obslog(obsinfo_file)
        addinfo_lst = {row['frameid']:row for row in addinfo_table}
        # prepare the difference list between real observation time and FITS
        # time
        real_obsdate_lst = []
        delta_t_lst = []

    # scan the raw files
    fname_lst = sorted(os.listdir(path))

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid', 'S12'),  ('imgtype', 'S3'),
        ('object',  'S12'), ('i2cell', 'bool'), ('exptime', 'f4'),
        ('obsdate', Time),  ('nsat',   'i4'),   ('q95',     'i4'),
        ])

    # prepare infomation to print
    pinfo = FormattedInfo(all_columns,
            ['frameid', 'fileid', 'imgtype', 'object', 'i2cell', 'exptime',
            'obsdate', 'nsat', 'q95'])

    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    #print(pinfo.get_dtype())
    print(pinfo.get_separator())

    prev_frameid = -1
    # start scanning the raw files
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid  = fname[0:-5]
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        # determine the science and overscan regions
        naxis1 = head['NAXIS1']
        naxis2 = head['NAXIS2']
        x1 = head.get('CRVAL1', 0)
        y1 = head.get('CRVAL2', 0)
        # get science region along x axis
        cover = head.get('COVER')
        if cover is None:
            if naxis1 >= 4096:
                cover = naxis1 - 4096
        # get science region along y axis
        rover = head.get('ROVER')
        if rover is None:
            if naxis2 >= 4136:
                rover = naxis2 - 4136

        # get start and end indices of science region
        y2 = y1 + naxis2 - rover
        x2 = x1 + naxis1 - cover
        data = data[y1:y2,x1:x2]

        # find frame-id
        frameid = int(fileid[8:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(
                    frameid, prev_frameid))

        # parse obsdate
        if 'DATE-STA' in head:
            obsdate = Time(head['DATE-STA'])
        else:
            obsdate = Time(head['DATE-OBS'])
        if (frameid in addinfo_lst and 'obsdate' in addinfo_table.colnames
            and addinfo_lst[frameid]['obsdate'] is not np.ma.masked):
            real_obsdate = addinfo_lst[frameid]['obsdate'].datetime
            file_obsdate = obsdate.datetime
            delta_t = real_obsdate - file_obsdate
            real_obsdate_lst.append(real_obsdate)
            delta_t_lst.append(delta_t.total_seconds())

        if 'EXPTIME' in head:
            exptime = head['EXPTIME']
        else:
            exptime = head['EXPOSURE']

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
        i2cell = objectname.lower()=='i2'
        if (frameid in addinfo_lst and 'i2cell' in addinfo_table.colnames
            and addinfo_lst[frameid]['i2cell'] is not np.ma.masked):
            i2cell = addinfo_lst[frameid]['i2cell']

        imgtype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        item = [frameid, fileid, imgtype, objectname, i2cell, exptime, obsdate,
                saturation, quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string = pinfo.get_format(has_esc=False).format(item)
        print(print_wrapper(string, item))

        prev_frameid = frameid

    print(pinfo.get_separator())
    
    # sort by obsdate
    #logtable.sort('obsdate')

    if has_obsinfo and len(real_obsdate_lst)>0:
        # determine the time offset as median value
        time_offset = np.median(np.array(delta_t_lst))
        time_offset_dt = datetime.timedelta(seconds=time_offset)
        # plot time offset
        fig = plt.figure(figsize=(9, 6), dpi=100)
        ax = fig.add_axes([0.12,0.16,0.83,0.77])
        xdates = mdates.date2num(real_obsdate_lst)
        ax.plot_date(xdates, delta_t_lst, 'o-', ms=6)
        ax.axhline(y=time_offset, color='k', ls='--', alpha=0.6)
        ax.set_xlabel('Log Time', fontsize=12)
        ax.set_ylabel('Log Time - FTIS Time (sec)', fontsize=12)
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.text(0.95*x1+0.05*x2, 0.1*y1+0.9*y2,
                'Time offset = %d seconds'%time_offset, fontsize=14)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.grid(True, ls='-', color='w')
        ax.set_facecolor('#eaeaf6')
        ax.set_axisbelow(True)
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        for t in ax.xaxis.get_ticklines():
            t.set_color('none')
        for t in ax.yaxis.get_ticklines():
            t.set_color('none')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        #plt.setp(ax.get_xticklabels(), rotation=30)i
        fig.autofmt_xdate()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        fig.suptitle('Time Offsets Between Log and FITS', fontsize=15)
        fig.savefig('obsdate_offset.png')
        plt.close(fig)

        # correct time offset
        for row in logtable:
            row['obsdate'] = row['obsdate'] + time_offset_dt

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'].iso[0:10]
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
    outfile = open(outfilename, 'w')
    outfile.write(pinfo.get_title()+os.linesep)
    outfile.write(pinfo.get_dtype()+os.linesep)
    outfile.write(pinfo.get_separator()+os.linesep)
    for row in logtable:
        outfile.write(pinfo.get_format().format(row)+os.linesep)
    outfile.close()


def reduce_rawdata():
    """2D to 1D pipeline for the High Resolution spectrograph on Xinglong 2.16m
    telescope.
    """

    # find obs log
    logname_lst = [fname for fname in os.listdir(os.curdir)
                        if fname[-7:]=='.obslog']
    if len(logname_lst)==0:
        print('No observation log found')
        exit()
    elif len(logname_lst)>1:
        print('Multiple observation log found:')
        for logname in sorted(logname_lst):
            print('  '+logname)
    else:
        pass

    # read obs log
    logtable = read_obslog(logname_lst[0])

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )

    # find local config file
    for fname in os.listdir(os.curdir):
        if re.match ('Xinglong216HRS\S*.cfg', fname):
            config.read(fname)
            print('Load Congfile File: {}'.format(fname))
            break

    fibermode = config['data']['fibermode']

    if fibermode == 'single':
        reduce_singlefiber(logtable, config)
    elif fibermode == 'double':
        reduce_doublefiber(logtable, config)
    else:
        print('Invalid fibermode:', fibermode)
