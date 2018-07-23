import os
import astropy.io.fits as fits

from ..utils import obslog

def make_log(path):
    '''
    Args:
        path (string): Path to the raw FITS files.
    '''
    cal_objects = ['bias', 'wideflat', 'narrowflat', 'flat', 'dark', 'iodine',
                    'thar']
    log = obslog.Log()
    for fname in sorted(os.listdir(path)):
        if fname[-5:] != '.fits':
            continue
        f = fits.open(os.path.join(path, fname))
        head = f[0].header
        data = f[0].data

        fileid     = fname[0:-5]
        obstype    = head['OBSTYPE']
        exptime    = head['EXPTIME']
        objectname = head['OBJECT']
        dateobs    = head['DATE-OBS']

        f.close()

        if objectname.lower().strip() in cal_objects:
            imgtype = 'cal'
        else:
            imgetype = 'sci'

        # determine the fraction of saturated pixels permillage
        mask_sat = (data>=65535)
        prop = float(mask_sat.sum())/data.size*1e3

        # find the brightness index in the central region
        h, w = data.shape
        data1 = data[h//2-2:h//2+3, int(w*0.3):int(w*0.7)]
        bri_index = np.median(data1,axis=1).mean()

        item = obslog.LotItem(
                fileld     = fileid,
                obsdate    = obsdate,
                exptime    = exptime,
                imagetype  = imagetype,
                objectname = objectname,
                obstype    = obstype,
                saturation = pop,
                brightness = bri_index,
                )
        log.add_item(item)

    log.sort('obsdata')

    # make info_lst
    all_info_lst = []
    columns = ['frameid (i)', 'fileid (s)', 'imagetype (s)', 'obstype (s)',
               'objectname (s)', 'i2 (i)', 'exptime (f)', 'obsdate (s)',
               'saturation (f)', 'brightness (f)']
    prev_frameid = -1
    for logitem in log:
        frameid = int(logitem.fileid[-4:])
        info_lst = [
                str(frameid),
                logitem.fileid,
                logitem.imagetype,
                logitem.obstype,
                str(logitem.i2),
                '%8.3f'%logitem.exptime,
                str(logitem.obsdate),
                '%.3f'%logitem.saturation,
                '%.1f'%logitem.brightness,
                ]
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
            if columns[i] in ['filename','object']:
                fmt = '%%-%ds'%maxlen[i]
            else:
                fmt = '%%%ds'%maxlen[i]
            info_lst[i] = fmt%(info_lst[i])

    string = '% columns = '+', '.join(columns)
    print(string)
    for info_lst in all_info_lst:
        string = ' | '.join(info_lst)
        string = ' '+string
        print(string)

