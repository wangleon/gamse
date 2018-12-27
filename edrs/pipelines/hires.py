import os
import re
import configparser

import numpy as np
import astropy.io.fits as fits

from ..utils import obslog

def make_log(path):
    '''
    Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (str): Path to the raw FITS files.

    '''

    # scan the raw files
    fname_lst = sorted(os.listdir(path))
    log = obslog.Log()
    for fname in fname_lst:
        if not re.match('HI\.\d{8}\.\d{5}\.fits', fname):
            continue
        fileid = fname[0:17]
        filepath = os.path.join(path, fname)
        hdu_lst = fits.open(filepath)
        head0 = hdu_lst[0].header

        # get obsdate in 'YYYY-MM-DDTHH:MM:SS' format
        date = head0.get('DATE-OBS')
        utc  = head0.get('UTC', head0.get('UT'))
        obsdate = '%sT%s'%(date, utc)

        exptime    = head0.get('ELAPTIME')
        i2in       = head0.get('IODIN', False)
        i2out      = head0.get('IODOUT', True)
        objectname = head0.get('TARGNAME', '')
        _type      = head0.get('IMAGETYP')
        if _type.strip() == 'object':
            imagetype = 'sci'
        else:
            imagetype = 'cal'
            objectname = _type

        data1 = hdu_lst[1].data
        data2 = hdu_lst[2].data
        data3 = hdu_lst[3].data

        # determine the fraction of saturated pixels permillage
        mask_sat1 = (data1==0)
        mask_sat2 = (data2==0)
        mask_sat3 = (data3==0)
        prop = (mask_sat1.sum() + mask_sat2.sum() + mask_sat3.sum())/(
                data1.size + data2.size + data3.size)*1e3

        # find the brightness index in the central region
        h, w = data2.shape
        d = data2[h//2-2:h//2+3, int(w*0.2):int(w*0.8)]
        bri_index = np.median(d, axis=1).mean()

        hdu_lst.close()

        item = obslog.LogItem(
                   fileid     = fileid,
                   obsdate    = obsdate,
                   exptime    = exptime,
                   imagetype  = imagetype,
                   i2         = i2in,
                   objectname = objectname,
                   saturation = prop,
                   brightness = bri_index,
                   )
        log.add_item(item)

    log.sort('obsdate')

    # make info list
    all_info_lst = []
    column_lst = [('frameid',    'i'), ('fileid',     's'), ('imagetype',  's'),
                  ('objectname', 's'), ('i2',         'i'), ('exptime',    'f'),
                  ('obsdate',    's'), ('saturation', 'f'), ('brightness', 'f'),
                 ]
    columns = ['%s (%s)'%(_name, _type) for _name, _type in column_lst]

    prev_frameid = -1
    for logitem in log:
        frameid = prev_frameid + 1
        info_lst = [
                    str(frameid),
                    str(logitem.fileid),
                    logitem.imagetype,
                    str(logitem.objectname),
                    int(logitem.i2),
                    '%8.3f'%logitem.exptime,
                    str(logitem.obsdate),
                    '%.3f'%logitem.saturation,
                    '%.1f'%logitem.brightness,
                ]
        prev_frameid = frameid
        all_info_lst.append(info_lst)

    # find the maximum length of each column
    length = []
    for info_lst in all_info_lst:
        length.append([len(info) for info in info_lst])
    length = np.array(length)
    maxlen = length.max(axis=0)

    # find the output format for each column
    for info_lst in all_info_lst:
        for i, info in enumerate(info_lst):
            if columns[i] in ['fileid (s)','objectname (s)']:
                fmt = '%%-%ds'%maxlen[i]
            else:
                fmt = '%%%ds'%maxlen[i]
            info_lst[i] = fmt%(info_lst[i])

    # write the obslog into an ascii file
    #date = log[0].fileid.split('_')[0]
    #outfilename = '%s-%s-%s.log'%(date[0:4],date[4:6],date[6:8])
    #outfile = open(outfilename,'w')
    string = '% columns = '+', '.join(columns)
    #outfile.write(string+os.linesep)
    print(string)
    for info_lst in all_info_lst:
        string = ' | '.join(info_lst)
        string = ' '+string
        #outfile.write(string+os.linesep)
        print(string)
    #outfile.close()
