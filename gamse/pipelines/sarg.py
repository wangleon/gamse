import os
import numpy as np
import astropy.io.fits as fits

from ..utils import obslog

def make_log(path):
    """

    Args:
        path (string): path to the raw FITS files
    """
    log = obslog.Log()
    for fname in sorted(os.listdir(path)):
        if fname[-4:] != '.fts':
            continue
        f = fits.open(os.path.join(path, fname))
        head = f[0].header
        data = f[0].data
        
        fileid = fname[0:-4]
        objectname = head['IDENT']
        object_alt = head['OBJCAT']
        exptime    = head['EXPTIME']
        obstype    = head['OBS-TYPE']
        obsdate    = head['DATE-OBS'] + 'T' + head['EXPSTART']
        slit       = head['SLIT_ID']
        grism      = head['GRM_ID']
        program    = head['PROGRAM']
        binx       = int(round(head['CRDELT1']))
        biny       = int(round(head['CRDELT2']))
        binning    = '%d, %d'%(biny, binx)
        if head['FLT2_ID']=='Iodine Cell' and head['IODINE_S']=='Iodine Cell ON':
            i2cell = 1
        else:
            i2cell = 0
        f.close()

        imagetype = ('cal', 'sci')[obstype=='OBJECT']

        mask_sat = (data>=65535)
        prop = float(mask_sat.sum())/data.size*1e3

        h, w = data.shape
        data1 = data[h//2-2:h//2+3, int(w*0.3):int(w*0.7)]
        bri_index = np.median(data1,axis=1).mean()

        item = obslog.LogItem(
                fileid     = fileid,
                obsdate    = obsdate,
                exptime    = exptime,
                imagetype  = imagetype,
                objectname = objectname,
                object_alt = object_alt,
                obstype    = obstype,
                i2cell     = i2cell,
                slit       = slit,
                grism      = grism,
                binning    = binning,
                program    = program,
                saturation = prop,
                brightness = bri_index,
                )
        log.add_item(item)

    log.sort('obsdate')

    # make info_lst
    all_info_lst = []
    columns = ['fileid (s)', 'imagetype (s)', 'obstype (s)', 'objectname (s)',
               'objectname_alt (s)',
               'i2cell (i)', 'exptime (f)', 'obsdate (s)', 'slit (s)',
               'grism (s)', 'binning', 'program (s)',
               'saturation (f)', 'brightness (f)']
    for logitem in log:
        info_lst = [
                logitem.fileid,
                logitem.imagetype,
                logitem.obstype,
                logitem.objectname,
                logitem.object_alt,
                str(logitem.i2cell),
                '%g'%logitem.exptime,
                str(logitem.obsdate),
                '%s'%logitem.slit,
                '%s'%logitem.grism,
                '%s'%logitem.binning,
                '%s'%logitem.program,
                '%.3f'%logitem.saturation,
                '%.1f'%logitem.brightness,
                ]
        all_info_lst.append(info_lst)

    length = []
    for info_lst in all_info_lst:
        length.append([len(info) for info in info_lst])
    length = np.array(length)
    maxlen = length.max(axis=0)

    # find the output format for each column
    for info_lst in all_info_lst:
        for i, info in enumerate(info_lst):
            if columns[i].split()[0] in ['obstype','objectname','object_alt',
                                        'slit','grism','program']:
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
