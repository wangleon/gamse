import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ..utils import obslog
from ..utils.config import read_config
from ..echelle.imageproc import combine_images

def parse_bias_data(filename):
    '''Parse singl bias data by fitting the data with polynomial to remove the
    cosmic rays'''

    data = fits.getdata(filename)
    h, w = data.shape

    ysum = data[:,::3].mean(axis=1)
    x = np.arange(ysum.size)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, ysum, alpha=0.3)

    mask = np.ones_like(ysum, dtype=np.bool)
    niter = 0
    maxiter = 5
    while(niter < maxiter):
        niter += 1
        coeff = np.polyfit(x[mask], ysum[mask], deg=6)
        yfit = np.polyval(coeff, x)
        res  = ysum - yfit
        std = res[mask].std()
        new_mask = ysum < yfit + 5.*std
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
        ax.plot(x, yfit, label='Iter %d'%niter)

    leg = ax.legend(loc='upper right')
    leg.get_frame().set_alpha(0.1)
    fig.savefig('bias.%s.ymean.png'%os.path.basename(filename))

    ygrid, xgrid = np.mgrid[0:h, 0:w]
    bias = np.polyval(coeff, ygrid)
    return bias

def reduce():
    '''Reduce the APF/Levy spectra.

    '''
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    config = read_config('Levy')

    rawdata = config['data']['rawdata']

    # parse bias
    bias_lst = []
    for item in log:
        if item.objectname[0]=='Dark' and abs(item.exptime-1)<1e-3:
            filename = os.path.join(rawdata, '%s.fits'%item.fileid)
            bias = parse_bias_data(filename)
            bias_lst.append(bias)

    bias = np.array(bias_lst).mean(axis=0)
    fits.writeto('bias.fits', bias, overwrite=True)

    # trace the order
    trace_lst = [fits.getdata(os.path.join(rawdata, '%s.fits'%item.fileid))
                 for item in log if item.objectname[0]=='NarrowFlat']
    trace = combine_images(trace_lst, mode='mean', upper_clip=10, maxiter=5)
    fits.writeto('trace.fits', trace, overwrite=True)


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
        obsdate    = head['DATE-OBS']
        i2cell     = {'In': 1, 'Out': 0}[head['ICELNAM']]

        f.close()

        imagetype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the fraction of saturated pixels permillage
        mask_sat = (data>=65535)
        prop = float(mask_sat.sum())/data.size*1e3

        # find the brightness index in the central region
        h, w = data.shape
        data1 = data[h//2-2:h//2+3, int(w*0.3):int(w*0.7)]
        bri_index = np.median(data1,axis=1).mean()

        item = obslog.LogItem(
                fileid     = fileid,
                obsdate    = obsdate,
                exptime    = exptime,
                imagetype  = imagetype,
                objectname = objectname,
                obstype    = obstype,
                i2cell     = i2cell,
                saturation = prop,
                brightness = bri_index,
                )
        log.add_item(item)

    log.sort('obsdate')

    # make info_lst
    all_info_lst = []
    columns = ['frameid (i)', 'fileid (s)', 'imagetype (s)', 'obstype (s)',
               'objectname (s)', 'i2cell (i)', 'exptime (f)', 'obsdate (s)',
               'saturation (f)', 'brightness (f)']
    prev_frameid = -1
    for logitem in log:
        frameid = int(logitem.fileid[-4:])
        info_lst = [
                str(frameid),
                logitem.fileid,
                logitem.objectname,
                logitem.imagetype,
                logitem.obstype,
                str(logitem.i2cell),
                '%g'%logitem.exptime,
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

