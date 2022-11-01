import os
import re
import shutil
import datetime
import dateutil.parser
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import median_filter
import scipy.interpolate as intp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat  import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                                mosaic_spec,
                                save_crossprofile, read_crossprofile,
                                )
from ...echelle.extract import extract_aperset, extract_aperset_optimal
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_self_wavelength,
                                select_calib_auto, select_calib_manu,
                                FWHMMapFigure, ResolutionFigure,
                                )
from .common import (get_bias, get_mask, correct_overscan,
                     TraceFigure, AlignFigure, BackgroundFigure,
                     SpatialProfileFigure,
                     select_calib_from_database)
from .common import get_interorder_background, get_tharpollution_lst
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg, get_flat)

def reduce_singlefiber_phase3(config, logtable):
    """Reduce the single fiber data of Xinglong 2.16m HRS.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of observing log.

    """
    # extract keywords from config file
    section      = config['data']
    rawpath      = section.get('rawpath')
    statime_key  = section.get('statime_key')
    exptime_key  = section.get('exptime_key')
    direction    = section.get('direction')

    section     = config['reduce']
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

    # initialize general card list
    general_card_lst = {}

    ### Parse bias

    bias, bias_card_lst = get_bias(config, logtable)

    ### define dtype of 1-d spectra
    if bias is None:
        ndisp = 4096
    else:
        ncros, ndisp = bias.shape

    types = [
            ('aperture',    np.int16),
            ('order',       np.int16),
            ('points',      np.int16),
            ('wavelength',  (np.float64, ndisp)),
            ('flux',        (np.float32, ndisp)),
            ('error',       (np.float32, ndisp)),
            ('background',  (np.float32, ndisp)),
            ('mask',        (np.int16,   ndisp)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    ### Combine the flats and trace the orders

    # fiter flat frames
    filterfunc = lambda item: item['object'].lower()=='flat'
    logitem_lst = list(filter(filterfunc, logtable))

    nflat = len(logitem_lst)

    flat_filename    = os.path.join(midpath, 'flat.fits')
    aperset_filename = os.path.join(midpath, 'trace.trc')
    aperset_regname  = os.path.join(midpath, 'trace.reg')
    trace_figname    = os.path.join(figpath, 'trace.{}'.format(fig_format))
    align_figname    = os.path.join(figpath, 'align.{}'.format(fig_format))
    profile_filename = os.path.join(midpath, 'profile.fits')


    if mode=='debug' and os.path.exists(flat_filename) \
        and os.path.exists(aperset_filename) \
        and os.path.exists(profile_filename):
        # read flat data and mask array
        hdu_lst = fits.open(flat_filename)
        flat_data = hdu_lst[0].data
        flat_mask = hdu_lst[1].data
        flat_norm = hdu_lst[2].data
        flat_sens = hdu_lst[3].data
        flat_spec = hdu_lst[4].data
        exptime   = hdu_lst[0].header[exptime_key]
        hdu_lst.close()
        aperset = load_aperture_set(aperset_filename)
        disp_x_lst, profile_x, profile_lst = read_crossprofile(profile_filename)
    else:
        data_lst = []
        head_lst = []
        exptime_lst = []
        obsdate_lst = []

        print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
        fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^23s} {:^8} {:^6}'
        head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                    'obsdate', 'N(sat)', 'Q95')

        for iframe, logitem in enumerate(logitem_lst):
            # read each individual flat frame
            fname = '{}.fits'.format(logitem['fileid'])
            filename = os.path.join(rawpath, fname)
            data, head = fits.getdata(filename, header=True)
            exptime_lst.append(head[exptime_key])
            obsdate_lst.append(dateutil.parser.parse(head[statime_key]))
            mask = get_mask(data, head)
            sat_mask = (mask&4>0)
            bad_mask = (mask&2>0)
            if iframe == 0:
                allmask = np.zeros_like(mask, dtype=np.int16)
            allmask += sat_mask

            # correct overscan for flat
            data, card_lst = correct_overscan(data, head, logitem['amp'])
            for key, value in card_lst:
                head.append((key, value))

            # correct bias for flat, if has bias
            if bias is None:
                message = 'No bias. skipped bias correction'
            else:
                data = data - bias
                message = 'Bias corrected'
            logger.info(message)

            # print info
            if iframe == 0:
                print(head_str)
            message = fmt_str.format(
                        '[{:d}]'.format(logitem['frameid']),
                        logitem['fileid'], logitem['object'],
                        logitem['exptime'], logitem['obsdate'],
                        logitem['nsat'], logitem['q95'])
            print(message)

            data_lst.append(data)

        if nflat == 1:
            flat_data = data_lst[0]
        else:
            data_lst = np.array(data_lst)
            flat_data = combine_images(data_lst,
                            mode       = 'mean',
                            upper_clip = 10,
                            maxiter    = 5,
                            maskmode   = (None, 'max')[nflat>3],
                            ncores     = ncores,
                            )

        # determine flat name (??sec or ??-??sec)
        if len(set(exptime_lst))==1:
            flatname = '{:g}sec'.format(exptime_lst[0])
        else:
            flatname = '{:g}-{:g}sec'.format(
                    min(exptime_lst), max(exptime_lst))

        # get mean exposure time and write it to header
        flat_head = fits.Header()

        # calculat the mean exposure time and write it to the new header
        exptime = np.mean(exptime_lst)
        flat_head[exptime_key] = exptime

        # calculate the mean start time and write it to th new header
        delta_t_lst = [(t-obsdate_lst[0]).total_seconds() for t in obsdate_lst]
        mean_delta_t = datetime.timedelta(seconds=np.mean(delta_t_lst))
        mean_obsdate = obsdate_lst[0] + mean_delta_t
        flat_head[statime_key] = mean_obsdate.isoformat()

        # find saturation mask
        sat_mask = allmask > nflat/2.
        flat_mask = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

        # get exposure time normalized flats
        flat_norm = flat_data/exptime

        tracefig = TraceFigure()    # create the trace figure
        alignfig = AlignFigure()    # create the align figure

        section = config['reduce.trace']
        aperset = find_apertures(flat_data, flat_mask,
                    scan_step  = section.getint('scan_step'),
                    minimum    = section.getfloat('minimum'),
                    separation = section.get('separation'),
                    align_deg  = section.getint('align_deg'),
                    filling    = section.getfloat('filling'),
                    degree     = section.getint('degree'),
                    conv_core  = 10,
                    fill       = True,
                    fill_tol   = 10,
                    display    = section.getboolean('display'),
                    fig_trace  = tracefig,
                    fig_align  = alignfig,
                    )

        # save the trace figure
        tracefig.adjust_positions()
        title = 'Trace for {}'.format(flat_filename)
        tracefig.suptitle(title, fontsize=15)
        tracefig.savefig(trace_figname)
        tracefig.close()

        # save the alignment figure
        alignfig.adjust_axes()
        title = 'Order Alignment for {}'.format(flat_filename)
        alignfig.suptitle(title, fontsize=12)
        alignfig.savefig(align_figname)
        alignfig.close()

        aperset.save_txt(aperset_filename)
        aperset.save_reg(aperset_regname)

        # do the flat fielding
        # prepare the output mid-prococess figures in debug mode
        if mode=='debug':
            figname = 'flat_aperpar_{}_%03d.{}'.format(
                        flatname, fig_format)
            fig_aperpar = os.path.join(figpath, figname)
        else:
            fig_aperpar = None

        # prepare the name for slit figure
        figname = 'slit.{}'.format(fig_format)
        fig_slit = os.path.join(figpath, figname)

        # prepare the name for slit file
        fname = 'slit.dat'
        slit_file = os.path.join(midpath, fname)

        section = config['reduce.flat']

        p1, p2, pstep = -8, 8, 0.1
        profile_x = np.arange(p1, p2+1e-4, pstep)
        disp_x_lst = np.arange(48, ndisp, 500)

        fig_spatial = SpatialProfileFigure()
        flat_sens, flatspec_lst, profile_lst = get_flat(
                data            = flat_data,
                mask            = flat_mask,
                apertureset     = aperset,
                nflat           = nflat,
                q_threshold     = section.getfloat('q_threshold'),
                smooth_A_func   = smooth_aperpar_A,
                smooth_c_func   = smooth_aperpar_c,
                smooth_bkg_func = smooth_aperpar_bkg,
                mode            = 'debug',
                fig_spatial     = fig_spatial,
                flatname        = flatname,
                profile_x       = profile_x,
                disp_x_lst      = disp_x_lst,
                )
        figname = os.path.join(figpath, 'spatial_profile.png')
        title = 'Spatial Profile of Flat'
        fig_spatial.suptitle(title)
        fig_spatial.savefig(figname)
        fig_spatial.close()

        # pack 1-d spectra of flat
        flat_spec = []
        for aper, flatspec in sorted(flatspec_lst.items()):
            n = flatspec.size

            # get the indices of not NaN values in flatspec
            idx_notnan = np.nonzero(~np.isnan(flatspec))[0]
            # use interpolate to fill the NaN values
            f = intp.InterpolatedUnivariateSpline(
                        idx_notnan, flatspec[idx_notnan], k=3)
            # get the first and last not NaN values
            i1 = idx_notnan[0]
            i2 = idx_notnan[-1]+1
            newx = np.arange(i1, i2)
            newspec = f(newx)
            # smooth the spec with savgol filter
            newspec = savgol_filter(newspec,
                            window_length = 101,
                            polyorder     = 3,
                            mode          = 'mirror',
                            )
            row = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    newspec,                        # flux
                    np.zeros(n, dtype=np.float32),  # error
                    np.zeros(n, dtype=np.float32),  # background
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            flat_spec.append(row)
        flat_spec = np.array(flat_spec, dtype=spectype)

        # save cross-profiles
        save_crossprofile(profile_filename, disp_x_lst,
                            p1, p2, pstep, profile_lst)

        # pack results and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(flat_data, header=flat_head),
                    fits.ImageHDU(flat_mask),
                    fits.ImageHDU(flat_norm),
                    fits.ImageHDU(flat_sens),
                    fits.BinTableHDU(flat_spec),
                    ])
        hdu_lst.writeto(flat_filename, overwrite=True)


    #################### Extract ThAr #######################

    # get the data shape
    ny, nx = flat_sens.shape

    calib_lst = {}

    # filter ThAr frames
    filter_thar = lambda item: item['object'].lower() == 'thar'

    thar_items = list(filter(filter_thar, logtable))

    for ithar, logitem in enumerate(thar_items):
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        objname = logitem['object']
        imgtype = logitem['imgtype']
        exptime = logitem['exptime']
        amp     = logitem['amp']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fmt_str = 'FileID: {} ({}) OBJECT: {} - wavelength identification'
        message = fmt_str.format(fileid, imgtype, objname)
        logger.info(message)
        print(message)

        fname = '{}.fits'.format(fileid)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst = correct_overscan(data, head, amp)
        for key, value in card_lst:
            head.append((key, value))
        message = 'Overscan corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct bias for ThAr, if has bias
        if bias is None:
            message = 'No bias'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        head.append(('HIERARCH GAMSE BACKGROUND CORRECTED', False))

        # extract ThAr spectra
        lower_limit = 7
        upper_limit = 7
        spectra1d = extract_aperset(data, mask,
                    apertureset = aperset,
                    lower_limit = lower_limit,
                    upper_limit = upper_limit,
                    )
        head = aperset.to_fitsheader(head)
        message = '1D spectra extracted for {:d} orders'.format(len(spectra1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # pack to a structured array
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size

            # pack to table
            row = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux
                    np.zeros(n, dtype=np.float32),  # error
                    np.zeros(n, dtype=np.float32),  # background
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=spectype)

        figname = 'wlcalib_{}.{}'.format(fileid, fig_format)
        wlcalib_fig = os.path.join(figpath, figname)

        section = config['reduce.wlcalib']

        title = 'Wavelength Indentification for {}.fits'.format(fileid)

        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                index_file = os.path.join(os.path.dirname(__file__),
                                '../../data/calib/wlcalib_xinglong216hrs.dat')

                message = ('Searching for archive wavelength calibration'
                           'file in "{}"'.format(
                               os.path.basename(index_file)))
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

                ref_spec, ref_calib = select_calib_from_database(
                            index_file, head[statime_key])

                if ref_spec is None or ref_calib is None:

                    message = ('Did not find nay archive wavelength'
                               'calibration file')
                    logger.info(logger_prefix + message)
                    print(screen_prefix + message)

                    # if failed, pop up a calibration window and identify
                    # the wavelengths manually
                    calib = wlcalib(spec,
                        figfilename = wlcalib_fig,
                        title       = title,
                        linelist    = section.get('linelist'),
                        window_size = section.getint('window_size'),
                        xorder      = section.getint('xorder'),
                        yorder      = section.getint('yorder'),
                        maxiter     = section.getint('maxiter'),
                        clipping    = section.getfloat('clipping'),
                        q_threshold = section.getfloat('q_threshold'),
                        )
                else:
                    # if success, run recalib
                    # determien the direction
                    message = 'Found archive wavelength calibration file'
                    logger.info(message)
                    print(screen_prefix + message)

                    ref_direction = ref_calib['direction']

                    if direction[1] == '?':
                        aperture_k = None
                    elif direction[1] == ref_direction[1]:
                        aperture_k = 1
                    else:
                        aperture_k = -1

                    if direction[2] == '?':
                        pixel_k = None
                    elif direction[2] == ref_direction[2]:
                        pixel_k = 1
                    else:
                        pixel_k = -1

                    result = find_caliblamp_offset(ref_spec, spec,
                                aperture_k  = aperture_k,
                                pixel_k     = pixel_k,
                                pixel_range = (-50, 50),
                                mode        = mode,
                                )
                    aperture_koffset = (result[0], result[1])
                    pixel_koffset    = (result[2], result[3])

                    message = 'Aperture offset = {}; Pixel offset = {}'
                    message = message.format(aperture_koffset,
                                             pixel_koffset)
                    logger.info(logger_prefix + message)
                    print(screen_prefix + message)

                    use = section.getboolean('use_prev_fitpar')
                    xorder      = (section.getint('xorder'), None)[use]
                    yorder      = (section.getint('yorder'), None)[use]
                    maxiter     = (section.getint('maxiter'), None)[use]
                    clipping    = (section.getfloat('clipping'), None)[use]
                    window_size = (section.getint('window_size'), None)[use]
                    q_threshold = (section.getfloat('q_threshold'), None)[use]

                    calib = recalib(spec,
                        figfilename      = wlcalib_fig,
                        title            = title,
                        ref_spec         = ref_spec,
                        linelist         = section.get('linelist'),
                        aperture_koffset = aperture_koffset,
                        pixel_koffset    = pixel_koffset,
                        ref_calib        = ref_calib,
                        xorder           = xorder,
                        yorder           = yorder,
                        maxiter          = maxiter,
                        clipping         = clipping,
                        window_size      = window_size,
                        q_threshold      = q_threshold,
                        direction        = direction,
                        )
            else:
                message = 'No database searching. Identify lines manually'
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

                # do not search the database
                calib = wlcalib(spec,
                    figfilename   = wlcalib_fig,
                    title         = title,
                    identfilename = section.get('ident_file', None),
                    linelist      = section.get('linelist'),
                    window_size   = section.getint('window_size'),
                    xorder        = section.getint('xorder'),
                    yorder        = section.getint('yorder'),
                    maxiter       = section.getint('maxiter'),
                    clipping      = section.getfloat('clipping'),
                    q_threshold   = section.getfloat('q_threshold'),
                    )
                message = ('Wavelength calibration finished.'
                            '(k, offset) = ({}, {})'.format(
                            calib['k'], calib['offset']))
                logger.info(logger_prefix + message)


            # then use this thar as reference
            ref_calib = calib
            ref_spec  = spec
            message = 'Reference calib and spec are selected'
            logger.info(logger_prefix + message)
        else:
            message = 'Use reference calib and spec'
            logger.info(logger_prefix + message)
            # for other ThArs, no aperture offset
            calib = recalib(spec,
                figfilename      = wlcalib_fig,
                title            = title,
                ref_spec         = ref_spec,
                linelist         = section.get('linelist'),
                ref_calib        = ref_calib,
                aperture_koffset = (1, 0),
                pixel_koffset    = (1, 0),
                xorder           = ref_calib['xorder'],
                yorder           = ref_calib['yorder'],
                maxiter          = ref_calib['maxiter'],
                clipping         = ref_calib['clipping'],
                window_size      = ref_calib['window_size'],
                q_threshold      = ref_calib['q_threshold'],
                direction        = direction,
                )

        # add more infos in calib
        calib['fileid']   = fileid
        calib['date-obs'] = head[statime_key]
        calib['exptime']  = head[exptime_key]
        message = 'Add more info in calib of {}'.format(fileid)
        logger.info(logger_prefix + message)

        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(spec, calib)
        message = 'Wavelength solution added'
        logger.info(logger_prefix + message)

        prefix = 'HIERARCH GAMSE WLCALIB '
        for key, value in card_lst:
            head.append((prefix+key, value))

        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    fits.BinTableHDU(identlist),
                    ])

        # save in midproc path as a wlcalib reference file
        fname = 'wlcalib_{}.fits'.format(fileid)
        filename = os.path.join(midpath, fname)
        hdu_lst.writeto(filename, overwrite=True)
        message = 'Wavelength calibrated spectra written to {}'.format(filename)
        logger.info(logger_prefix + message)

        # save in onedspec path
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)
        message = 'Wavelength calibrated spectra written to {}'.format(filename)
        logger.info(logger_prefix + message)

        # plot fwhm map
        fwhm_fig = FWHMMapFigure(spec, identlist, aperset,
                fwhm_range=(3,8))
        title = 'FWHM Map for {}'.format(fileid)
        fwhm_fig.suptitle(title, fontsize=13)
        figname = 'fwhm_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        fwhm_fig.savefig(figfilename)
        fwhm_fig.close()

        # plot resolution map
        reso_fig = ResolutionFigure(spec, identlist, aperset,
                resolution_range=(4e4, 6e4))
        title = 'Resolution Map for {}'.format(fileid)
        reso_fig.suptitle(title, fontsize=13)
        figname = 'resolution_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        reso_fig.savefig(figfilename)
        reso_fig.close()
    
        # pack to calib_lst
        calib_lst[frameid] = calib


    # print fitting summary
    fmt_string = ' [{:3d}] {} - ({:4g} sec) - {:4d}/{:4d} RMS = {:7.5f}'
    section = config['reduce.wlcalib']
    auto_selection = section.getboolean('auto_selection')

    if auto_selection:
        rms_threshold    = section.getfloat('rms_threshold', 0.005)
        group_contiguous = section.getboolean('group_contiguous', True)
        time_diff        = section.getfloat('time_diff', 120)

        ref_calib_lst = select_calib_auto(calib_lst,
                            rms_threshold    = rms_threshold,
                            group_contiguous = group_contiguous,
                            time_diff        = time_diff,
                        )
        ref_fileid_lst = [calib['fileid'] for calib in ref_calib_lst]

        # print ThAr summary and selected calib
        for frameid, calib in sorted(calib_lst.items()):
            string = fmt_string.format(frameid, calib['fileid'],
                        calib['exptime'], calib['nuse'], calib['ntot'],
                        calib['std'])
            if calib['fileid'] in ref_fileid_lst:
                string = '\033[91m{} [selected]\033[0m'.format(string)
            print(string)
    else:
        # print the fitting summary
        for frameid, calib in sorted(calib_lst.items()):
            string = fmt_string.format(frameid, calib['fileid'],
                        calib['exptime'], calib['nuse'], calib['ntot'],
                        calib['std'])
            print(string)

        promotion = 'Select References: '
        ref_calib_lst = select_calib_manu(calib_lst,
                            promotion = promotion)


    ######## Extract 1d spectra of flat fielding ######
    extract_flat = False
    if extract_flat:

        # prepar message prefix
        logger_prefix = 'Flat - '
        screen_prefix = '    - '
        # correct flat for flat
        data = flat_norm/flat_sens
        message = 'Flat field corrected for flat.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # get background light for flat field
        ny, nx = data.shape
        allx = np.arange(nx)
        background = get_interorder_background(data, flat_mask, aperset)
        for y in np.arange(ny):
            m = flat_mask[y,:]==0
            f = intp.InterpolatedUnivariateSpline(
                        allx[m], background[y,:][m], k=3)
            background[y,:][~m] = f(allx[~m])
        background = median_filter(background, size=(9,5), mode='nearest')
        background = savitzky_golay_2d(background, window_length=(21, 101),
                        order=3, mode='nearest')

        # plot stray light
        figname = 'bkg2d_{}.{}'.format('flat', fig_format)
        figfilename = os.path.join(figpath, figname)
        fig_bkg = BackgroundFigure(data, background,
                    title   = 'Background Correction for {}'.format('flat'),
                    figname = figfilename,
                    )
        fig_bkg.close()

        # remove background light
        data = data - background
        message = 'Background corrected. Max = {:.2f}; Mean = {:.2f}'.format(
                    background.max(), background.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectrum
        result = extract_aperset_optimal(data, flat_mask,
                    background      = background,
                    apertureset     = aperset,
                    gain            = 1.02,
                    ron             = 3.29,
                    profilex        = profile_x,
                    disp_x_lst      = disp_x_lst,
                    main_disp       = 'x',
                    upper_clipping  = 25,
                    recenter        = True,
                    mode            = mode,
                    profile_lst     = profile_lst,
                    plot_apertures  = [19],
                )
        flux_opt_lst = result[0]
        flux_err_lst = result[1]
        back_opt_lst = result[2]
        flux_sum_lst = result[3]
        back_sum_lst = result[4]

        # pack spectrum
        spec = []
        for aper in sorted(flux_opt_lst.keys()):
            n = flux_opt_lst[aper].size
            row = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_opt_lst[aper],             # flux
                    flux_err_lst[aper],             # error
                    back_opt_lst[aper],             # background
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=spectype)
        # pack and save spectra
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(head=flat_head),
                    fits.BinTableHDU(spec),
                    ])
        fname = '{}_{}.fits'.format('flat', oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)
    
        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)


    ########### Extract Science Spectrum ##########
    # filter science items in logtable
    #extr_filter = config['reduce.extract'].get('extract',
    #                    'lambda row: row["imgtype"]=="sci"')
    #extr_filter = eval(extr_filter)
    extr_filter = lambda row: row['imgtype']=='sci'
    extr_items = list(filter(extr_filter, logtable))


    for logitem in extr_items:

        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objname = logitem['object']
        exptime = logitem['exptime']
        amp     = logitem['amp']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        filename = os.path.join(rawpath, '{}.fits'.format(fileid))

        message = 'FileID: {} ({}) OBJECT: {}'.format(
                    fileid, imgtype, objname)
        logger.info(message)
        print(message)

        # read raw data
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data, card_lst = correct_overscan(data, head, amp)
        for key, value in card_lst:
            head.append((key, value))
        message = 'Overscan corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct bias
        if bias is None:
            message = 'No bias'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct flat
        data = data/flat_sens
        message = 'Flat field corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        ny, nx = data.shape
        allx = np.arange(nx)
        # get background lights
        background = get_interorder_background(data, mask, aperset)
        for y in np.arange(ny):
            m = mask[y,:]==0
            f = intp.InterpolatedUnivariateSpline(
                    allx[m], background[y,:][m], k=3)
            background[y,:][~m] = f(allx[~m])
        background = median_filter(background, size=(9,5), mode='nearest')
        background = savitzky_golay_2d(background, window_length=(21, 101),
                        order=3, mode='nearest')

        # plot stray light
        figname = 'bkg2d_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        fig_bkg = BackgroundFigure(data, background,
                    title   = 'Background Correction for {}'.format(fileid),
                    figname = figfilename,
                    )
        fig_bkg.close()

        data = data - background
        message = 'Background corrected. Max = {:.2f}; Mean = {:.2f}'.format(
                    background.max(), background.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectrum
        section = config['reduce.extract']
        method  = section.get('method')
        deblaze = section.getboolean('deblaze', False)

        if method == 'optimal':
            result = extract_aperset_optimal(data, mask,
                        background      = background,
                        apertureset     = aperset,
                        gain            = 1.02,
                        ron             = 3.29,
                        profilex        = profile_x,
                        disp_x_lst      = disp_x_lst,
                        main_disp       = 'x',
                        upper_clipping  = 5,
                        recenter        = True,
                        mode            = mode,
                        profile_lst     = profile_lst,
                        plot_apertures  = [],
                    )
            flux_opt_lst = result[0]
            flux_err_lst = result[1]
            back_opt_lst = result[2]
            flux_sum_lst = result[3]
            back_sum_lst = result[4]

            # pack spectrum
            spec = []
            for aper in sorted(flux_opt_lst.keys()):
                n = flux_opt_lst[aper].size

                row = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux_opt_lst[aper],             # flux
                        flux_err_lst[aper],             # error
                        back_opt_lst[aper],             # background
                        np.zeros(n, dtype=np.int16),    # mask
                        )
                spec.append(row)
            spec = np.array(spec, dtype=spectype)

        elif method == 'sum':
            lower_limit = section.getfloat('lower_limit')
            upper_limit = section.getfloat('upper_limit')

            # extract 1d spectra of the object
            spectra1d = extract_aperset(data, mask,
                            apertureset = aperset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                        )
            norder = len(spectra1d)
            message = '1D spectra of {} orders extracted'.format(norder)
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            # extract 1d spectra for straylight/background light
            background1d = extract_aperset(background, mask,
                            apertureset = aperset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                        )
            message = '1D straylight of {} orders extracted'.format(
                        len(background1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            prefix = 'HIERARCH GAMSE EXTRACTION '
            head.append((prefix + 'LOWER LIMIT', lower_limit))
            head.append((prefix + 'UPPER LIMIT', upper_limit))

            # pack spectrum
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                n = flux_sum.size
                # background 1d flux
                back_flux = background1d[aper]['flux_sum']
            
                row = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux_sum,                       # flux
                        np.zeros(n, dtype=np.float32),  # error
                        back_flux,                      # background
                        np.zeros(n, dtype=np.int16),    # mask
                        )
                spec.append(row)
            spec = np.array(spec, dtype=spectype)

        # wavelength calibration
        weight_lst = get_calib_weight_lst(ref_calib_lst,
                        obsdate = head[statime_key],
                        exptime = head[exptime_key],
                        )

        message_lst = ['Wavelength calibration:']
        for i, calib in enumerate(ref_calib_lst):
            string = ' '*len(screen_prefix)
            string =  string + '{} ({:4g} sec) {} weight = {:5.3f}'.format(
                        calib['fileid'], calib['exptime'], calib['date-obs'],
                        weight_lst[i])
            message_lst.append(string)
        message = os.linesep.join(message_lst)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        spec, card_lst = reference_spec_wavelength(spec,
                            ref_calib_lst, weight_lst)
        prefix = 'HIERARCH GAMSE WLCALIB '
        for key, value in card_lst:
            head.append((prefix + key, value))

        # pack and save wavelength referenced spectra
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    ])

        # correct the blaze function
        if deblaze:
            spec2 = []
            for row in spec:
                aper  = row['aperture']
                order = row['order']
                n     = row['points']
                wave  = row['wavelength']
                flux  = row['flux']
                error = row['error']
                back  = row['background']
                mask  = row['mask']
                #newflux = np.array([np.NaN]*flux.size)

                m = flat_spec['aperture']==aper
                cont = flat_spec[m][0]['flux']
                flux2 = flux/cont
                normc = np.percentile(flux2[n//4:n//4*3], 98)
                newflux = flux2/normc

                row2 = (aper, order, n,
                        wave,       # wavelength
                        newflux,    # flux
                        error,      # error
                        back,       # background
                        mask,       # mask
                        )
                spec2.append(row2)

            spec2 = np.array(spec2, dtype=spectype)
            # append the deblazed spectra into fits
            hdu_lst.append(fits.BinTableHDU(spec2))

        # write 1d spectra to fits file
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
