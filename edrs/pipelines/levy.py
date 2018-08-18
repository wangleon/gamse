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
from ..echelle.wvcalib import (wvcalib, recalib, select_calib_from_database, 
                               self_reference_singlefiber,
                               wv_reference_singlefiber, get_time_weight)
from ..echelle.background import correct_background

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
    if os.path.exists('bias.fits'):
        bias = fits.getdata('bias.fits')
    else:
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

    # trace the orders
    if os.path.exists('trace.fits'):
        trace = fits.getdata('trace.fits')
    else:
        trace_lst = []
        for item in log:
            if item.objectname[0]=='NarrowFlat':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data = fits.getdata(filename)
                data = correct_overscan(data)
                trace_lst.append(data - bias)
        trace = combine_images(trace_lst, mode='mean', upper_clip=10, maxiter=5)
        trace = trace.T
        fits.writeto('trace.fits', trace, overwrite=True)


    if os.path.exists('trace.trc'):
        aperset = load_aperture_set('trace.trc')
    else:
        mask = np.zeros_like(trace, dtype=np.int8)

        aperset = find_apertures(trace, mask,
            scan_step  = 50,
            minimum    = 8,
            seperation = 14,
            sep_der    = 3,
            filling    = 0.2,
            degree     = 3,
            display    = True,
            filename   = 'trace.fits',
            fig_file   = 'trace.png',
            trace_file = 'trace.trc',
            reg_file   = 'trace.reg',
            )

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
    for flatname, fileids in flat_groups.items():
        flat_filename = '%s.fits'%flatname
        mask_filename = '%s_msk.fits'%flatname
        if os.path.exists(flat_filename) and os.path.exists(mask_filename):
            flat_data = fits.getdata(flat_filename)
            mask_table = fits.getdata(mask_filename)
            mask_array = table_to_array(mask_table, flat_data.shape)
        else:
            data_lst = []
            for ifile, fileid in enumerate(fileids):
                filename = os.path.join(rawdata, '%s.fits'%fileid)
                data = fits.getdata(filename)
                mask = (data[:,0:2048]==65535)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += mask
                data = correct_overscan(data)
                data = data - bias
                data_lst.append(data)
            nflat = len(data_lst)
            print('combine images for', flatname)
            flat_data = combine_images(data_lst, mode='mean',
                        upper_clip=10, maxiter=5)
            fits.writeto(flat_filename, flat_data, overwrite=True)
            
            sat_mask = allmask>nflat/2.
            mask_array = np.int16(sat_mask)*4
            mask_table = array_to_table(mask_array)
            fits.writeto(mask_filename, mask_table, overwrite=True)
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = mask_array

    # extract flat spectrum
    flatmap_lst = {}
    flat_spectra1d_lst = {}
    for flatname in flat_groups:
        resp_filename = '%s_resp.fits'%flatname
        data = flat_data_lst[flatname]
        mask = flat_mask_lst[flatname]
        spectra1d = extract_aperset(data.T, mask.T,
                        apertureset = aperset,
                        lower_limit = 5,
                        upper_limit = 5,
                        )
        flat_spectra1d_lst[flatname] = spectra1d

        if os.path.exists(resp_filename):
            flatmap = fits.getdata(resp_filename)
        else:
            flatmap = get_slit_flat(data.T, mask.T,
                            apertureset = aperset,
                            spectra1d   = spectra1d,
                            lower_limit = 6,
                            upper_limit = 5,
                            deg         = 7,
                            q_threshold = 20**2,
                            )
            fits.writeto(resp_filename, flatmap, overwrite=True)
        flatmap_lst[flatname] = flatmap


    # mosaic flats
    mosaic_resp_filename = 'flat_resp.fits'
    if os.path.exists(mosaic_resp_filename):
        flatmap = fits.getdata(mosaic_resp_filename)
    else:
        mosaic_mask_lst = {flatname:np.zeros_like(flat_data_lst[flatname].T,dtype=np.bool)
                           for flatname in flat_groups}
        maxcount = 55000
        h, w = flat_data_lst[list(flat_groups.keys())[0]].T.shape
        yy, xx = np.mgrid[:h:,:w:]
        for iaper, (aper, aper_loc) in enumerate(sorted(aperset.items())):
    
            # find the maximum count and its belonging flatname of this aperture
            cmax = -999
            maxflatname = None
            for flatname in flat_groups:
                nsat = flat_spectra1d_lst[flatname][aper]['mask_sat'].sum()
                cmaxi = np.sort(flat_spectra1d_lst[flatname][aper]['flux_mean'])[-10]
                if nsat > 0 or cmaxi > maxcount:
                    continue
                if cmaxi > cmax:
                    cmax = cmaxi
                    maxflatname = flatname
    
            print(aper, maxflatname, cmax)
            #domain = aper_loc.position.domain
            #d1, d2 = int(domain[0]), int(domain[1])+1
            #newx = np.arange(d1, d2)
            newx = np.arange(aper_loc.shape[1])
            position = aper_loc.position(newx)
            if iaper==0:
                mosaic_mask_lst[maxflatname][:,:] = True
            else:
                boundary = (position + prev_position)/2
                _m = yy > boundary
                for flatname in flat_groups:
                    if flatname == maxflatname:
                        mosaic_mask_lst[flatname][_m] = True
                    else:
                        mosaic_mask_lst[flatname][_m] = False
            prev_position = position
    
        flatname0 = list(flat_groups.keys())[0]
        flatmap0 = flatmap_lst[flatname0]
        flatmap = np.zeros_like(flatmap0, dtype=flatmap0.dtype)
        for flatname, maskdata in mosaic_mask_lst.items():
            #fits.writeto('mask_%s.fits'%flatname, np.int16(maskdata), overwrite=True)
            flatmap += flatmap_lst[flatname]*maskdata
        fits.writeto(mosaic_resp_filename, flatmap, overwrite=True)

    # extract ThAr
    h, w = bias.shape
    spectype = np.dtype({
                'names':  ('aperture', 'order', 'points', 'wavelength', 'flux'),
                'formats':('i',       'i',     'i',      '(%d,)f8'%h,  '(%d,)f'%h),
                })

    if True:
        calib_lst = {}
        count_thar = 0
        for item in log:
            if item.objectname[0]=='ThAr':
                count_thar += 1
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                mask = np.int16(data == 65535)*4
                data = correct_overscan(data)
                spectra1d = extract_aperset(data.T, mask.T,
                                apertureset = aperset,
                                lower_limit = 5,
                                upper_limit = 5,
                                )
                head = aperset.to_fitsheader(head, channel=None)
    
                spec = [(aper, 0, item['flux_sum'].size,
                        np.zeros_like(item['flux_sum'].size, dtype=np.float64),
                        item['flux_sum'])
                        for aper, item in sorted(spectra1d.items())]
                spec = np.array(spec, dtype=spectype)
                
                if count_thar == 1:
                    ref_spec, ref_calib, ref_aperset = select_calib_from_database('Levy',
                                                        'DATE-OBS',
                                                        head['DATE-OBS'],
                                                        channel=None)
                    aper_offset = ref_aperset.find_aper_offset(aperset)
                    print(aper_offset)
    
                if ref_spec is None or ref_calib is None:
                    calib = wvcalib(spec,
                                    filename      = '%s.fits'%item.fileid,
                                    identfilename = 'a.idt',
                                    figfilename   = 'wvcalib_%s.png'%item.fileid,
                                    channel       = None,
                                    linelist      = 'thar.dat',
                                    window_size   = 13,
                                    xorder        = 5,
                                    yorder        = 4,
                                    maxiter       = 10,
                                    clipping      = 3,
                                    snr_threshold = 10,
                                    )
                    aper_offset = 0
                else:
                    calib = recalib(spec,
                                    filename      = '%s.fits'%item.fileid,
                                    figfilename   = 'wvcalib_%s.png'%item.fileid,
                                    ref_spec      = ref_spec,
                                    channel       = None,
                                    linelist      = 'thar.dat',
                                    identfilename = '',
                                    aperture_offset = aper_offset,
                                    coeff         = ref_calib['coeff'],
                                    npixel        = ref_calib['npixel'],
                                    window_size   = ref_calib['window_size'],
                                    xorder        = ref_calib['xorder'],
                                    yorder        = ref_calib['yorder'],
                                    maxiter       = ref_calib['maxiter'],
                                    clipping      = ref_calib['clipping'],
                                    snr_threshold = ref_calib['snr_threshold'],
                                    k             = ref_calib['k'],
                                    offset        = ref_calib['offset'],
                                    )
    
                if count_thar == 1:
                    ref_calib = calib
                    ref_spec  = spec
                    aper_offset = 0
    
                hdu_lst = self_reference_singlefiber(spec, head, calib)
                hdu_lst.writeto('%s_wlc.fits'%item.fileid, overwrite=True)

                # add more infos in calib
                calib['fileid']   = item.fileid
                calib['date-obs'] = head['DATE-OBS']
                calib['exptime']  = head['EXPTIME']
                # pack to calib_lst
                calib_lst[item.frameid] = calib

        for frameid, calib in sorted(calib_lst.items()):
            print(' [%3d] %s - %4d/%4d r.m.s = %7.5f'%(frameid,
                  calib['fileid'], calib['nuse'], calib['ntot'],calib['std']))

        # print promotion and read input frameid list
        string = input('select references: ')
        ref_frameid_lst = [int(s) for s in string.split(',')
                                    if len(s.strip())>0 and
                                    s.strip().isdigit() and
                                    int(s) in calib_lst]
        ref_calib_lst    = [calib_lst[frameid]
                                for frameid in ref_frameid_lst]
        ref_datetime_lst = [calib_lst[frameid]['date-obs']
                                for frameid in ref_frameid_lst]

    # extract science frames
    for item in log:
        if item.imagetype=='sci':
            filename = os.path.join(rawdata, '%s.fits'%item.fileid)
            data, head = fits.getdata(filename, header=True)
            mask = np.int16(data == 65535)*4

            data = correct_overscan(data)

            # write order locations to header
            head = aperset.to_fitsheader(head, channel=None)

            # flat fielding correction
            data = data.T/flatmap
            mask = mask.T

            # background correction
            stray = correct_background(data, mask,
                        channels        = ['A'],
                        apertureset_lst = {'A': aperset},
                        scale           = 'log',
                        block_mask      = 4,
                        scan_step       = 200,
                        xorder          = 2,
                        yorder          = 3,
                        maxiter         = 5,
                        upper_clip      = 3,
                        lower_clip      = 3,
                        extend          = True,
                        display         = True,
                        fig_file        = 'background_%s.png'%item.fileid,
                        )
            data = data - stray
            # 1d spectra extraction
            spectra1d = extract_aperset(data, mask,
                        apertureset = aperset,
                        lower_limit = 6,
                        upper_limit = 5,
                        )
            spec = [(aper, 0, item['flux_sum'].size,
                    np.zeros_like(item['flux_sum'].size, dtype=np.float64),
                    item['flux_sum'])
                    for aper, item in sorted(spectra1d.items())]
            spec = np.array(spec, dtype=spectype)

            weight_lst = get_time_weight(ref_datetime_lst, head['DATE-OBS'])
            spec, head = wv_reference_singlefiber(spec, head, ref_calib_lst, weight_lst)

            # pack and save wavelength referenced spectra
            pri_hdu = fits.PrimaryHDU(header=head)
            tbl_hdu = fits.BinTableHDU(spec)
            hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
            hdu_lst.writeto('%s_wlc.fits'%item.fileid, overwrite=True)


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
                logitem.imagetype,
                logitem.obstype,
                logitem.objectname,
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
