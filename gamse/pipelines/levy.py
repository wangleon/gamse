import os
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import scipy.signal as sg
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from ..utils import obslog
from ..utils.config import read_config
from ..echelle.imageproc import combine_images, table_to_array, array_to_table
from ..echelle.trace import find_apertures, load_aperture_set
from ..echelle.flat import get_slit_flat
from ..echelle.extract import extract_aperset
from ..echelle.wlcalib import (wlcalib, recalib, select_calib_from_database, 
                               self_reference_singlefiber,
                               wl_reference_singlefiber, get_time_weight)
from ..echelle.background import find_background

def correct_overscan(data, head):
    """Correct the overscan of CCD image.

    Args:
        data (:class:`numpy.dtype`): Input data image.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        tuple: A tuple containing:

            * **corrdata** (:class:`numpy.dtype`) – Output image with overscan 
              corrected.
            * **head** (:class:`astropy.io.fits.Header`) – Updated FITS header.
            * **overmean** (*float*) – Average of overscan values.
    """
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
        overmean = overdata.mean()

        # update fits header
        head['HIERARCH EDRS OVERSCAN']        = True
        head['HIERARCH EDRS OVERSCAN METHOD'] = 'smooth'
        head['HIERARCH EDRS OVERSCAN AXIS-1'] = '2049:2088'
        head['HIERARCH EDRS OVERSCAN AXIS-2'] = '0:4608'
        head['HIERARCH EDRS OVERSCAN MEAN']   = overmean

        return corrdata, head, overmean

def reduce():
    """Reduce the APF/Levy spectra.
    """

    # read obs log
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    # load config files
    config_file_lst = []
    # find built-in config file
    config_path = os.path.join(os.path.dirname(__file__), '../data/config')
    config_file = os.path.join(config_path, 'Levy.cfg')
    if os.path.exists(config_file):
        config_file_lst.append(config_file)

    # find local config file
    for fname in os.listdir(os.curdir):
        if fname[-4:]=='.cfg':
            config_file_lst.append(fname)

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # extract keywords from config file
    section     = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    section     = config['reduce']
    midproc     = section.get('midproc')
    result      = section.get('result')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')

    # create folders if not exist
    if not os.path.exists(report):  os.mkdir(report)
    if not os.path.exists(result):  os.mkdir(result)
    if not os.path.exists(midproc): os.mkdir(midproc)

    ################################ parse bias ################################
    section = config['reduce.bias']
    bias_file = section['bias_file']

    if os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        bias = fits.getdata(bias_file)
        logger.info('Load bias from image: %s'%bias_file)
    else:
        bias_data_lst = []

        # prepare print info
        columns = [
                ('fileid',   '{0:10s}', '{0.fileid:10s}'),
                ('exptime',  '{1:7s}',  '{0.exptime:7g}'),
                ('obsdate',  '{2:25s}', '{0.obsdate:25s}'),
                ('overscan', '{3:8s}',  '{1:8.2f}'),
                ('mean',     '{4:8s}',  '{2:8.2f}'),
                ]
        title, fmt_title, fmt_item = zip(*columns)
        fmt_title = ' '.join(fmt_title)
        fmt_item  = ' '.join(fmt_item)

        for item in log:
            if item.objectname[0]=='Bias' and abs(item.exptime)<1e-3:
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                # correct overscan here
                data, head, overmean = correct_overscan(data, head)

                # print info
                if len(bias_data_lst) == 0:
                    print('* Combine Bias Images: %s'%bias_file)
                    print(' '*2 + fmt_title.format(*title))
                print(' '*2 + fmt_item.format(item, overmean, data.mean()))

                bias_data_lst.append(data)

        n_bias = len(bias_data_lst)         # number of bias images
        has_bias = n_bias > 0

        if has_bias:
            # there is bias frames

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            bias = combine_images(bias_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('cosmic_clip'),
                    maxiter    = section.getint('maxiter'),
                    mask       = (None, 'max')[n_bias>=3],
                    )

            # create new FITS Header for bias
            head = fits.Header()
            head['HIERARCH EDRS BIAS NFILE'] = n_bias

            ############## bias smooth ##################
            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                if smooth_method in ['gauss','gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')

                    bias_smooth = gaussian_filter(bias,
                                    sigma=smooth_sigma, mode=smooth_mode)

                    # write information to FITS header
                    head['HIERARCH EDRS BIAS SMOOTH']        = True
                    head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                    head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                    head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth
            else:
                # bias not smoothed
                head['HIERARCH EDRS BIAS SMOOTH'] = False

            fits.writeto(bias_file, bias, header=head, overwrite=True)
            logger.info('Bias image written to "%s"'%bias_file)

        else:
            # no bias found
            pass

    ############################# trace the orders ############################
    section = config['reduce.trace']
    trace_file = section['trace_file']

    if os.path.exists(trace_file):
        # load trace image from existing file
        has_trace = True
        trace = fits.getdata(trace_file)
    else:
        # combine trace file from narrow flats
        trace_data_lst = []

        # prepare print info
        columns = [
                ('fileid',   '{0:10s}', '{0.fileid:10s}'),
                ('exptime',  '{1:7s}',  '{0.exptime:7g}'),
                ('obsdate',  '{2:25s}', '{0.obsdate:25s}'),
                ('overscan', '{3:8s}',  '{1:8.2f}'),
                ('mean',     '{4:8s}',  '{2:8.2f}'),
                ]
        title, fmt_title, fmt_item = zip(*columns)
        fmt_title = ' '.join(fmt_title)
        fmt_item  = ' '.join(fmt_item)

        for item in log:
            if item.objectname[0]=='NarrowFlat':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                data, head, overmean = correct_overscan(data, head)
                if has_bias:
                    data = data - bias

                # print info
                if len(trace_data_lst) == 0:
                    print('* Combine Images for Order Tracing: %s'%trace_file)
                    print(' '*2 + fmt_title.format(*title))
                print(' '*2 + fmt_item.format(item, overmean, data.mean()))

                trace_data_lst.append(data)

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
                    mask       = (None, 'max')[n_trace>=3],
                    )
            trace = trace.T
            fits.writeto(trace_file, trace, overwrite=True)

        else:
            # no trace image found
            pass

    # find the name of .trc file
    trc_file = '.'.join(trace_file.split('.')[:-1])+'.trc'
    trc_reg  = '.'.join(trace_file.split('.')[:-1])+'.reg'
    trace_filename = os.path.basename(trace_file)
    trace_fileid = '.'.join(trace_filename.split('.')[:-1])
    fig_file = os.path.join(report, '%s.%s'%(trace_fileid, fig_format))

    if os.path.exists(trc_file):
        # load apertures from existing file
        aperset = load_aperture_set(trc_file)
    else:
        mask = np.zeros_like(trace, dtype=np.int8)

        aperset = find_apertures(trace, mask,
                    scan_step  = section.getint('scan_step'),
                    minimum    = section.getfloat('minimum'),
                    separation = section.getfloat('separation'),
                    sep_der    = section.getfloat('sep_der'),
                    filling    = section.getfloat('filling'),
                    degree     = section.getint('degree'),
                    display    = section.getboolean('display'),
                    filename   = trace_file,
                    fig_file   = fig_file,
                    )
        aperset.save_txt(trc_file)
        aperset.save_reg(trc_reg)

    ######################### find flat groups #################################
    ########################### Combine flat images ############################
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
                    aper_offset = ref_aperset.find_aper_offset(aperset)
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
    for item in log:
        if item.imagetype=='sci':
            filename = os.path.join(rawdata, '%s.fits'%item.fileid)
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


def make_obslog(path):
    """Print the observing log.

    Args:
        path (str): Path to the raw FITS files.
    """
    cal_objects = ['bias', 'wideflat', 'narrowflat', 'flat', 'dark', 'iodine',
                    'thar']

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid',     'S12'),  ('imagetype',  'S3'),
        ('obstype', 'S8'),
        ('object',  'S12'), ('i2',         'bool'), ('exptime',    'f4'),
        ('obsdate', 'S23'), ('saturation', 'i4'),   ('quantile95', 'i4'),
        ])

    # prepare infomation to print
    columns = [
            ('fileid',     '{:^12s}', '{:12s}'),
            ('object',     '{:^12s}', '{:12s}'),
            ('exptime',    '{:^7s}',  '{:7g}'),
            ('obsdate',    '{:^25s}', '{:25s}'),
            ('saturation', '{:^10s}', '{:10d}'),
            ('quantile95', '{:^10s}', '{:10d}'),
            ]
    titles, fmt_title, fmt_item = zip(*columns)
    fmt_title = ' '.join(fmt_title)
    fmt_item  = ' '.join(fmt_item)
    # print titles and a set of lines
    print(fmt_title.format(*titles))
    print(' '.join(['-'*len(fmt.format(title)) for title, fmt, _ in columns]))

    # start scanning the raw files
    for fname in sorted(os.listdir(path)):
        if fname[-5:] != '.fits':
            continue
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        fileid     = fname[0:-5]
        obstype    = head['OBSTYPE']
        exptime    = head['EXPTIME']
        objectname = head['OBJECT']
        obsdate    = head['DATE-OBS']
        i2cell     = {'In': 1, 'Out': 0}[head['ICELNAM']]

        imagetype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        item = [0, fileid, imagetype, obstype, objectname, i2cell, exptime,
                obsdate, saturation, quantile95]
        logtable.add_row(item)

        print(fmt_item.format(fileid, objectname, exptime, obsdate,
                saturation, quantile95))

    logtable.sort('obsdate')

    # allocate frameid
    prev_frameid = -1
    for item in logtable:
        frameid = int(item['fileid'][-4:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(
                    frameid, prev_frameid))

        item['frameid'] = frameid

        prev_frameid = frameid

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'][0:10]
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
    logtable.write(outfilename, format='ascii.fixed_width_two_line')

    return True
