import os
import numpy as np
import astropy.io.fits as fits
import scipy.signal as sg
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from ..utils import obslog
from ..utils.config import read_config
from ..echelle.imageproc import combine_images, table_to_array, array_to_table
from ..echelle.trace import find_apertures, load_aperture_set
from ..echelle.flat import get_slit_flat
from ..echelle.extract import extract_aperset

def correct_overscan(data):
    if data.shape==(4608, 2080):
        overmean = data[:,2049:2088].mean(axis=1)
        oversmooth = sg.savgol_filter(overmean, window_length=1201, polyorder=3)
        #coeff = np.polyfit(np.arange(overmean.size), overmean, deg=7)
        #oversmooth2 = np.polyval(coeff, np.arange(overmean.size))
        res = (overmean - oversmooth).std()
        #fig = plt.figure(dpi=150)
        #ax = fig.gca()
        #ax.plot(overmean)
        #ax.plot(oversmooth)
        #ax.plot(oversmooth2)
        #plt.show()
        #plt.close(fig)
        overdata = np.tile(oversmooth, (2048, 1)).T
        corrdata = data[:,0:2048] - overdata
        return corrdata

def reduce():
    '''Reduce the APF/Levy spectra.
    '''
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    config = read_config('Levy')

    rawdata = config['data']['rawdata']

    # parse bias
    if True:
        bias_lst = []
        for item in log:
            if item.objectname[0]=='Dark' and abs(item.exptime-1)<1e-3:
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data = fits.getdata(filename)
                data = correct_overscan(data)
                bias_lst.append(data)

        bias = combine_images(bias_lst, mode='mean', upper_clip=10, maxiter=5)
        bias = gaussian_filter(bias, 3, mode='nearest')
        fits.writeto('bias.fits', bias, overwrite=True)
    else:
        bias = fits.getdata('bias.fits')
    exit()

    # trace the orders
    if False:
        trace_lst = []
        for item in log:
            if item.objectname[0]=='NarrowFlat':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data = fits.getdata(filename)
                trace_lst.append(data - bias)
        trace = combine_images(trace_lst, mode='mean', upper_clip=10, maxiter=5)
        trace = trace.T
        fits.writeto('trace.fits', trace, overwrite=True)
    else:
        trace = fits.getdata('trace.fits')

    if False:
        mask = np.zeros_like(trace, dtype=np.int8)

        aperset = find_apertures(trace, mask,
            scan_step  = 50,
            minimum    = 8,
            seperation = 15,
            sep_der    = 3,
            filling    = 0.3,
            degree     = 3,
            display    = True,
            filename   = 'trace.fits',
            fig_file   = 'trace.png',
            trace_file = 'trace.trc',
            reg_file   = 'trace.reg',
            )
    else:
        aperset = load_aperture_set('trace.trc')


    # combine flat images
    flat_groups = {}
    for item in log:
        if item.objectname[0]=='WideFlat':
            flatname = 'flat_%d'%item.exptime
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(item.fileid)
    # print how many flats in each flat name
    for flatname in flat_groups:
        n = len(flat_groups[flatname])
        print('%3d images in %s'%(n, flatname))

    flat_data_lst = {}
    flat_mask_lst = {}
    if True:
        for flatname, fileids in flat_groups.items():
            data_lst = []
            for ifile, fileid in enumerate(fileids):
                filename = os.path.join(rawdata, '%s.fits'%fileid)
                data = fits.getdata(filename)
                mask = (data==65535)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += mask
                data = data - bias
                data_lst.append(data)
            nflat = len(data_lst)
            print('combine images for', flatname)
            flat_data = combine_images(data_lst, mode='mean',
                        upper_clip=10, maxiter=5)
            fits.writeto('%s.fits'%flatname, flat_data, overwrite=True)

            sat_mask = allmask>nflat/2.
            mask_array = np.int16(sat_mask)*4
            mask_table = array_to_table(mask_array)
            fits.writeto('%s_msk.fits'%flatname, mask_table, overwrite=True)
            flat_data_lst[flatname] = flat_data
            flat_mask_lst[flatname] = mask_array
    else:
        for flatname in flat_groups:
            data = fits.getdata('%s.fits'%flatname)
            mask_table = fits.getdata('%s_msk.fits'%flatname)
            mask_array = table_to_array(mask_table, data.shape)
            flat_data_lst[flatname] = data
            flat_mask_lst[flatname] = mask_array

    # extract flat spectrum
    if True:
        flat_spectra1d_lst = {}
        for flatname in flat_groups:
            data = flat_data_lst[flatname]
            mask = flat_mask_lst[flatname]
            spectra1d = extract_aperset(data.T, mask.T,
                            apertureset = aperset,
                            lower_limit = 5,
                            upper_limit = 5,
                            )
            flat_spectra1d_lst[flatname] = spectra1d
            for aper in aperset:
                print(flatname, aper, spectra1d[aper]['mask_sat'].sum())

            flatmap = get_slit_flat(data.T, mask.T,
                                apertureset = aperset,
                                spectra1d   = spectra1d,
                                lower_limit = 6,
                                upper_limit = 5,
                                deg         = 7,
                                q_threshold = 20**2,
                                )
            fits.writeto('%s_resp.fits'%flatname, flatmap, overwrite=True)

    # mosaic flats
    #data = fits.getdata('flatmap_40_resp.fits')
    #shape = data.shape
    #maskdata_lst = {flatname: np.zeros(shape, dtype=np.bool)
    #                for flatname in flat_groups}
    #lower_limit = 5
    #upper_limit = 5
    #for iaper, (aper, aper_loc) in enumerate(sorted(aperset.items())):

    

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

