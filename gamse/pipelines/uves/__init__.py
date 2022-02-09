import os
import re
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date, get_date_from_cmd
from ..common import load_obslog, load_config

def make_config():
    
    # find date of data obtained
    current_pathname = os.path.basename(os.getcwd())
    guess_date = extract_date(current_pathname)
    input_date = get_date_from_cmd(guess_date)

    # create config object
    config = configparser.ConfigParser()

    config.add_section('data')
    config.set('data', 'telescope',    'VLT')
    config.set('data', 'instrument',   'UVES')
    config.set('data', 'rawpath',      'rawdata')

    # write to config file
    filename = 'UVES.{}.cfg'.format(input_date.strftime('%Y-%m-%d'))
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
    # load config file
    config = load_config('UVES\S*\.cfg$')
    rawpath = config['data'].get('rawpath')

    # scan the raw files
    fname_lst = sorted(os.listdir(rawpath))

    # prepare logtable
    logtable = Table(dtype=[
                        ('frameid', 'i2'),
                        ('fileid',  'S28'),
                        ('imgtype', 'S13'),
                        ('object',  'S20'),
                        ('exptime', 'f4'),
                        ('obsdate', 'S23'),
                        ('lst',     'f4'),
                        ('opath',   'S4'),
                        ('mode',    'S10')
                ])

    fmt_str = ('  - {:7s} {:28s} {:<13s} {:20s} {:>9} {:23s} {:>9} {:>4} {:>10}')
    head_str = fmt_str.format('frameid', 'fileid', 'imgtype', 'object',
                'exptime', 'obsdate', 'lst', 'opath', 'mode')
    print(head_str)

    prev_frameid = -1
    pattern = '(UVES\.\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\.fits'
    for fname in fname_lst:
        mobj = re.match(pattern, fname)
        if not mobj:
            continue

        fileid = mobj.group(1)

        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        imgtype = head['ESO DPR TYPE']
        if imgtype=='SLIT':
            continue

        lst = head['LST']


        objectname = head['OBJECT']
        obsdate = dateutil.parser.parse(head['DATE-OBS'])
        exptime = head['EXPTIME']
        if abs(exptime - round(exptime))<1e-2:
            exptime = int(round(exptime))

        #opath = head['ESO INS PATH']
        mode  = head.get('ESO INS MODE', '')
        ccd   = head['ESO DET CHIPS']

        frameid = prev_frameid + 1

        item = [frameid, fileid, imgtype, objectname, exptime,
                obsdate.isoformat()[0:23], lst,
                str(ccd), mode]
        logtable.add_row(item)
        item = logtable[-1]

        # print log item
        string = fmt_str.format(
                    '[{:d}]'.format(frameid),
                    fileid, imgtype, objectname,
                    '{:9.4f}'.format(exptime),
                    obsdate.isoformat()[0:23],
                    '{:9.3f}'.format(lst),
                    ccd, mode,
                )
        print(string)

        prev_frameid = frameid


    logtable.sort('obsdate')

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['fileid'][5:15]

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

    outfile = open(outfilename, 'w')
    for row in logtable.pformat_all():
        outfile.write(row+os.linesep)
    outfile.close()
