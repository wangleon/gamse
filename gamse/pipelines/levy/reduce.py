import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat import get_slit_flat
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, select_calib_from_database, 
                               #self_reference_singlefiber,
                               #wl_reference,
                               get_calib_weight_lst)
from ...echelle.background import find_background
from ...utils.config import read_config
from ...utils.obslog import read_obslog
from ..common import load_obslog, load_config
from .common import (correct_overscan, TraceFigure)

def reduce_rawdata():
    """Reduce the APF/Levy spectra.
    """

    # read obslog and config
    config = load_config('Levy\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    # extract keywords from config file
    section = config['data']
    rawpath     = section.get('rawpath')
    statime_key = 'DATE-OBS'
    exptime_key = 'EXPTIME'

    section = config['reduce']
    midpath     = section.get('midpath')
    odspath     = section.get('odspath')
    figpath     = section.get('figpath')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')
    ncores      = section.get('ncores')

    # create folders if not exist
    if not os.path.exists(figpath): os.mkdir(figpath)
    if not os.path.exists(odspath): os.mkdir(odspath)
    if not os.path.exists(midpath): os.mkdir(midpath)

    # determine number of cores to be used
    if ncores == 'max':
        ncores = os.cpu_count()
    else:
        ncores = min(os.cpu_count(), int(ncores))

    ########################### find & trace the orders ######################
    section = config['reduce.trace']
    trace_file = section['trace_file']

    if mode=='debug' and os.path.exists(trace_file):
        # load trace image from existing file
        has_trace = True
        trace = fits.getdata(trace_file)
    else:
        # combine trace file from narrow flats
        trace_data_lst = []

        fmt_str = ('  - {:5s} {:11s} {:5s} {:<12s} {:1s}I2 {:>7} {:^23s}'
                ' {:>7} {:>5} {:7}')
        head_str = fmt_str.format('FID', 'fileid', 'type', 'object', '',
                                 'exptime', 'obsdate', 'nsat', 'q95', 'ovrmean')

        for logitem in logtable:
            if logitem['object'].strip()=='NarrowFlat':
                fname = '{}.fits'.format(logitem['fileid'])
                filename = os.path.join(rawpath, fname)
                data, head = fits.getdata(filename, header=True)
                data, card_lst, overmean = correct_overscan(data, head)

                trace_data_lst.append(data)

                # print info
                if len(trace_data_lst) == 1:
                    # first trace data
                    print('* Combine Images for Order Tracing: %s'%trace_file)
                    print(head_str)
                message = fmt_str.format(
                            '[{:d}]'.format(logitem['frameid']),
                            logitem['fileid'],
                            '({:3s})'.format(logitem['imgtype']),
                            logitem['object'],
                            logitem['i2'],
                            logitem['exptime'],
                            logitem['obsdate'],
                            logitem['nsat'],
                            logitem['q95'],
                            '{:>7.2f}'.format(overmean),
                            )
                print(message)


        n_trace = len(trace_data_lst)  # number of trace images
        has_trace = n_trace > 0

        if has_trace:
            # there is trace frames

            # combine trace images
            trace_data_lst = np.array(trace_data_lst)

            trace = combine_images(trace_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('upper_clip'),
                    maxiter    = section.getint('maxiter'),
                    maskmode   = (None, 'max')[n_trace>=3],
                    ncores     = ncores,
                    )

            fits.writeto(trace_file, trace, overwrite=True)

        else:
            # no trace image found
            pass

    # find the name of .trc file
    traceid = os.path.basename(trace_file)[0:-4]
    trac_file = os.path.join(midpath, '{}.trc'.format(traceid))
    treg_file = os.path.join(midpath, '{}.reg'.format(traceid))

    if mode=='debug' and os.path.exists(trac_file):
        # load apertures from existing file
        aperset = load_aperture_set(trac_file)
    else:
        mask = np.zeros_like(trace, dtype=np.int8)

        # create the trace figure
        tracefig = TraceFigure(datashape=trace.shape)

        aperset = find_apertures(trace, mask,
                    transpose  = True,
                    scan_step  = section.getint('scan_step'),
                    minimum    = section.getfloat('minimum'),
                    separation = section.get('separation'),
                    align_deg  = section.getint('align_deg'),
                    filling    = section.getfloat('filling'),
                    degree     = section.getint('degree'),
                    display    = section.getboolean('display'),
                    fig        = tracefig,
                    )
        # save the trace figure
        tracefig.adjust_positions()
        title = 'Trace for {}'.format(trace_file)
        tracefig.suptitle(title)
        figname = '{}.{}'.format(traceid, fig_format)
        figfile = os.path.join(figpath, figname)
        tracefig.savefig(figfile)
        tracefig.close()

        aperset.save_txt(trac_file)
        aperset.save_reg(treg_file)

    ######################### find flat groups #################################
    ########################### Combine flat images ############################
    exit()
    flat_groups = {}
    for logitem in logtable:
        if logitem['object']=='WideFlat':
            flatname = 'flat_{%d}'.format(logitem['exptime'])
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem['fileid'])
    # print how many flats in each flat name
    for flatname, item_lst in sorted(flat_groups.items()):
        n = len(item_lst)
        print('{} images in {}'.format(n, flatname))

    flat_data_lst = {}
    flat_mask_lst = {}
    for flatname, logitem_lst in flat_groups.items():
        flat_filename = '{}.fits'.format(flatname)
        mask_filename = '_msk.fits'%flatname
        if os.path.exists(flat_filename) and os.path.exists(mask_filename):
            flat_data = fits.getdata(flat_filename)
            mask_table = fits.getdata(mask_filename)
            mask_array = table_to_array(mask_table, flat_data.shape)
        else:
            data_lst = []
            for ifile, fileid in enumerate(fileids):
                filename = os.path.join(rawpath, '%s.fits'%fileid)
                data, head = fits.getdata(filename, header=True)
                mask = (data[:,0:2048]==65535)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += mask
                data, head = correct_overscan(data, head)
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

    ######################### Extract flat spectrum ############################
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


    ############################ Mosaic flats ##################################
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

    ############################# extract ThAr #################################
    ny, nx = bias.shape
    spectype = np.dtype({
                'names':  ('aperture', 'order', 'points', 'wavelength', 'flux'),
                'formats':('i',       'i',     'i',      '(%d,)f8'%ny,  '(%d,)f'%ny),
                })

    if True:
        calib_lst = {}
        count_thar = 0
        for logitem in logtable:
            if logitem.objectname[0]=='ThAr':
                count_thar += 1
                filename = os.path.join(rawpath, logitem['fileid']+'.fits')
                data, head = fits.getdata(filename, header=True)
                mask = np.int16(data == 65535)*4
                data, head, overmean = correct_overscan(data, head)
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
    
                if ref_spec is None or ref_calib is None:
                    calib = wvcalib(spec,
                                    filename      = '%s.fits'%logitem['fileid'],
                                    identfilename = 'a.idt',
                                    figfilename   = 'wvcalib_%s.png'%logitem['fileid'],
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
                    aper_offset = ref_aperset.find_aper_offset(aperset)
                    calib = recalib(spec,
                                    filename      = '%s.fits'%logitem['fileid'],
                                    figfilename   = 'wvcalib_%s.png'%logitem['fileid'],
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
                  calib['fileid'], calib['nuse'], calib['ntot'], calib['std']))

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

    ###################### Extract science spectra #############################
    for logitem in logtable:
        if logitem['imgtype']=='sci':
            filename = os.path.join(rawpath, item['fileid']+'.fits')
            data, head = fits.getdata(filename, header=True)
            mask = np.int16(data == 65535)*4

            data, head, overmean = correct_overscan(data, head)

            # write order locations to header
            head = aperset.to_fitsheader(head, channel=None)

            # flat fielding correction
            data = data.T/flatmap
            mask = mask.T

            # background correction
            stray = find_background(data, mask,
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
            spec, head = wv_reference_singlefiber(spec, head,
                            ref_calib_lst, weight_lst)

            # pack and save wavelength referenced spectra
            pri_hdu = fits.PrimaryHDU(header=head)
            tbl_hdu = fits.BinTableHDU(spec)
            hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
            hdu_lst.writeto('%s_wlc.fits'%item.fileid, overwrite=True)

all_columns = [
        ('frameid', 'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',  'str',   '{:^12s}', '{0[fileid]:12s}'),
        ('imgtype', 'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',  'str',   '{:^12s}', '{0[object]:12s}'),
        ('exptime', 'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate', 'time',  '{:^23s}', '{0[obsdate]:}'),
        ('nsat',    'int',   '{:^7s}',  '{0[nsat]:7d}'),
        ('q95',     'int',   '{:^6s}',  '{0[q95]:6d}'),
        ]

