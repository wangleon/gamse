import os
import re
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from ...utils.misc import extract_date, get_date_from_cmd
from ..common import load_obslog, load_config
from .common import get_metadata

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


def make_metatable(rawpath):

    # prepare metatable
    metatable = Table(dtype=[
                        ('expoid',  'i4'),
                        ('det',     'S3'),
                        ('expid',   'i4'),
                        ('fileid',  'S28'),
                        ('category','S8'),
                        ('imgtype', 'S13'),
                        ('object',  'S20'),
                        ('ra',      'f8'),
                        ('dec',     'f8'),
                        ('exptime', 'f4'),
                        ('obsdate', 'S23'),
                        ('mode',    'S10'),
                        ('slitwid', 'f4'),
                        ('slitlen', 'f4'),
                        ('binning', 'S7'),
                        ('progid',  'S30'),
                        ('pi',      'S50'),
                ], masked=True)
    pattern = '(UVES\.\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\.fits'
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(pattern, fname)
        if not mobj:
            continue

        fileid = mobj.group(1)

        filename = os.path.join(rawpath, fname)
        meta = get_metadata(filename)

        mask_ra  = (meta['category']=='CALIB' and meta['imgtype']!='STD')
        mask_dec = (meta['category']=='CALIB' and meta['imgtype']!='STD')
        mask_slitwid = (meta['slitwid'] is None)
        mask_slitlen = (meta['slitlen'] is None)

        if len(meta['targname'])==0:
            objname = meta['objname']
        else:
            objname = meta['targname']

        binning = '({0[0]}, {0[1]})'.format(meta['binning'])
        item = [
                (meta['expoid'],    False),
                (meta['detector'],  False),
                (meta['expid'],     False),
                (fileid,            False),
                (meta['category'],  False),
                (meta['imgtype'],   False),
                (objname,           False),
                (meta['ra'],        mask_ra),
                (meta['dec'],       mask_dec),
                (meta['exptime'],   False),
                (meta['obsdate'],   False),
                (meta['mode'],      False),
                (meta['slitwid'],   mask_slitwid),
                (meta['slitlen'],   mask_slitlen),
                (binning,           False),
                (meta['progid'],    False),
                (meta['piname'],    False),
                ]
        value, mask = list(zip(*item))

        metatable.add_row(value, mask=mask)

        # print information
        string_lst = [
                '{:5d}'.format(meta['expoid']),
                '{:1s}'.format(meta['detector']),
                '{:5d}'.format(meta['expid']),
                fileid,
                '{:7s}'.format(meta['category']),
                '{:20s}'.format(meta['imgtype']),
                '{:26s}'.format(objname),
                ' '*9 if mask_ra  else '{:9.5f}'.format(meta['ra']),
                ' '*9 if mask_dec else '{:9.5f}'.format(meta['dec']),
                '{:7g}'.format(meta['exptime']),
                meta['obsdate'],
                '{:10s}'.format(meta['mode']),
                ' '*4 if mask_slitwid else '{:4g}'.format(meta['slitwid']),
                ' '*4 if mask_slitlen else '{:4g}'.format(meta['slitlen']),
                binning,
                '{:>15s}'.format(meta['progid']),
                meta['piname'],
                ]
        print(' '.join(string_lst))

    format_metatable(metatable)

    return metatable

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


def format_metatable(metatable):
    metatable['det'].info.format='%-3s'
    metatable['ra'].info.format='%10.6f'
    metatable['dec'].info.format='%9.5f'
    maxlen = max([len(s) for s in metatable['category']])
    metatable['category'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['imgtype']])
    metatable['imgtype'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['object']])
    metatable['object'].info.format='%-{}s'.format(maxlen)
    metatable['exptime'].info.format='%7g'
    metatable['mode'].info.format='%9s'
    metatable['progid'].info.format='%15s'
