import os
import re
import shutil
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import median_filter

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures, load_aperture_set, ApertureSet
from ...echelle.flat  import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                                mosaic_spec)
from ...echelle.background import (find_background, get_interorder_background,
                                   get_xdisp_profile, BackgroundLight,
                                   find_best_background,
                                   select_background_from_database,
                                   )
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, get_calib_from_header,
                                get_calib_weight_lst, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_pixel_wavelength,
                                reference_self_wavelength,
                                select_calib_auto, select_calib_manu,
                                )
from .common import (get_bias, get_mask, correct_overscan, 
                     select_calib_from_database,
                     TraceFigure, BackgroundFigure, BrightnessProfileFigure,
                     )
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg)

def get_fiberobj_lst(string):
    ptn_obj = '[a-zA-Z0-9+-_\s\/]*'
    pattern = '\[A\]\s*({})\s*\[B\]\s*({})'.format(ptn_obj, ptn_obj)
    mobj = re.match(pattern, string)
    if mobj:
        objname_A = mobj.group(1).strip()
        objname_B = mobj.group(2).strip()
        return (objname_A, objname_B)
    else:
        return (None, None)

def reduce_doublefiber_phase3(config, logtable):
    """Reduce the multi-fiber data of Xinglong 2.16m HRS.

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
    readout_mode = section.get('readout_mode')

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

    ############################# parse bias ###################################

    bias, bias_card_lst = get_bias(config, logtable)

    ny, nx = bias.shape

    ######### find flat images that used for order trace #########
    trace_lst = {'A': [], 'B': []}
    for logitem in logtable:
        objectname = logitem['object']
        fileid     = logitem['fileid']

        # skip bias frames
        if objectname.strip().lower() == 'bias':
            continue

        # parse objectnames
        obj_A, obj_B = get_fiberobj_lst(objectname)

        if obj_A.lower() == 'flat' and obj_B == '':
            trace_lst['A'].append(logitem)
        elif obj_A == '' and obj_B.lower() == 'flat':
            trace_lst['B'].append(logitem)

    #print(trace_lst)
    master_aperset = {}

    for fiber in ['A', 'B']:
        # find the one with maximum q95 value for each fiber
        logitem_lst = sorted(trace_lst[fiber],
                        key=lambda logitem:logitem['q95'])
        logitem = logitem_lst[-1]

        #print(fiber, logitem)

        fname = '{}.fits'.format(logitem['fileid'])
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)
        sat_mask = (mask&4>0)
        bad_mask = (mask&2>0)

        # correct overscan for flat
        data, card_lst = correct_overscan(data, head, readout_mode)
        # correct bias for flat, if has bias
        if bias is None:
            message = 'No bias. skipped bias correction'
        else:
            data = data - bias
            message = 'Bias corrected'
        logger.info(message)


        # create the trace figure
        tracefig = TraceFigure()
        section = config['reduce.trace']
        aperset = find_apertures(data, mask,
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
        title = 'Trace for Fiber {}'.format(fiber)
        tracefig.suptitle(title, fontsize=15)
        trace_figname = os.path.join(figpath,
                        'trace_{}.{}'.format(fiber, fig_format))
        tracefig.savefig(trace_figname)

        aperset_filename = os.path.join(midpath,
                        'trace_{}.trc'.format(fiber))
        aperset.save_txt(aperset_filename)

        aperset_regname = os.path.join(midpath,
                        'trace_{}.reg'.format(fiber))
        aperset.save_reg(aperset_regname, fiber=fiber,
                        color={'A':'green','B':'yellow'}[fiber])

        newaperset = ApertureSet(shape=(ny, nx))
        count = 0
        for aper, aperloc in sorted(aperset.items()):
            ypos = aperloc.position(nx//2)
            if ypos > 154:
                newaperset[count] = aperloc
                count += 1

        master_aperset[fiber] = newaperset

        
    ################### Extract ThAr ####################

    # define dtype of 1-d spectra for all fibers
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, nx)),
            ('flux',       (np.float32, nx)),
            ('mask',       (np.int32,   nx)),
            ]
    names, formats = list(zip(*types))
    wlcalib_spectype = np.dtype({'names': names, 'formats': formats})


    calib_lst = {'A': {}, 'B': {}}

    def filter_thar(logitem):
        fiberobj_lst = get_fiberobj_lst(logitem['object'])
        return fiberobj_lst in [('ThAr',''), ('','ThAr'), ('ThAr','ThAr')]

    thar_items = list(filter(filter_thar, logtable))

    count_thar = 0
    for logitem in thar_items:
        # logitem alias
        frameid = logitem['frameid']
        imgtype = logitem['imgtype']
        fileid  = logitem['fileid']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fiberobj_lst = get_fiberobj_lst(logitem['object'])
        fiberobj_str = '[A] {0} [B] {1}'.format(*fiberobj_lst)

        message = ('FileID: {} ({}) OBJECT: {}'
                   ' - wavelength identification'.format(
                    fileid, imgtype, fiberobj_str))
        logger.info(message)
        print(message)

        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst = correct_overscan(data, head, readout_mode)
        for key, value in card_lst:
            head.append((key, value))

        message = 'Overscan corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct bias for ThAr, if has bias
        if bias is None:
            message = 'No Bias'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract ThAr
        section = config['reduce.extract']
        lower_limit = section.getfloat('lower_limit')
        upper_limit = section.getfloat('upper_limit')
        for ifiber in range(2):

            if fiberobj_lst[ifiber]=='':
                continue

            fiber = chr(ifiber+65)
            spec1d = extract_aperset(data, mask,
                        apertureset = master_aperset[fiber],
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                        )
            message = 'Fiber {}: 1D spectra of {} orders extracted'.format(
                       fiber, len(spec1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            # pack to a structured array
            spec = []
            for aper, item in sorted(spec1d.items()):
                flux = item['flux_sum']
                n = flux.size
                # pack to table
                item = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux,                           # flux
                        np.zeros(n),                    # mask
                        )
                spec.append(item)
            spec = np.array(spec, dtype=wlcalib_spectype)

            figname = 'wlcalib_{}_{}.{}'.format(fileid, fiber, fig_format)
            wlcalib_fig = os.path.join(figpath, figname)

            section = config['reduce.wlcalib']

            title = '{}.fits - Fiber {}'.format(fileid, fiber)

            if count_thar == 0:
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
                '''
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
                '''
                # temporarily added
                calib = ref_calib

            # add more infos in calib
            calib['fileid']   = fileid
            calib['date-obs'] = head[statime_key]
            calib['exptime']  = head[exptime_key]
            message = 'Add more info in calib of {}'.format(fileid)
            logger.info(logger_prefix + message)
            count_thar += 1

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
            fname = 'wlcalib_{}_{}.fits'.format(fileid, fiber)
            filename = os.path.join(midpath, fname)
            hdu_lst.writeto(filename, overwrite=True)
            message = ('Wavelength calibrated spectra'
                       ' written to {}').format(filename)
            logger.info(logger_prefix + message)

            # save in onedspec path
            fname = '{}_{}_{}.fits'.format(fileid, fiber, oned_suffix)
            filename = os.path.join(odspath, fname)
            hdu_lst.writeto(filename, overwrite=True)
            message = ('Wavelength calibrated spectra'
                       ' written to {}').format(filename)
            logger.info(logger_prefix + message)

            # pack to calib_lst
            calib_lst[fiber][frameid] = calib

    # print fitting summary
    fmt_string = ' [{:3d}] {} - ({:4g} sec) - {:4d}/{:4d} RMS = {:7.5f}'
    section = config['reduce.wlcalib']
    auto_selection = section.getboolean('auto_selection')

    if auto_selection:
        rms_threshold    = section.getfloat('rms_threshold', 0.005)
        group_contiguous = section.getboolean('group_contiguous', True)
        time_diff        = section.getfloat('time_diff', 120)

        ref_calib_lst = {'A': calib, 'B': calib}
    else:
        pass


    # define dtype of 1-d spectra
    types = [
            ('aperture',    np.int16),
            ('order',       np.int16),
            ('points',      np.int16),
            ('wavelength',  (np.float64, nx)),
            ('flux',        (np.float32, nx)),
            ('error',       (np.float32, nx)),
            ('background',  (np.float32, nx)),
            ('mask',        (np.int16,   nx)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})


    # filter science items in logtable
    extr_filter = lambda logitem: logitem['imgtype']=='sci'
    extr_items = list(filter(extr_filter, logtable))

    for logitem in extr_items:
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objects = logitem['object']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fiberobj_lst = get_fiberobj_lst(logitem['object'])
        fiberobj_str = '[A] {0} [B] {1}'.format(*fiberobj_lst)

        message = ('FileID: {} ({}) OBJECT: {}'
                   ' - wavelength identification'.format(
                    fileid, imgtype, fiberobj_str))
        logger.info(message)
        print(message)

        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst = correct_overscan(data, head, readout_mode)
        for key, value in card_lst:
            head.append((key, value))

        message = 'Overscan corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct bias for ThAr, if has bias
        if bias is None:
            message = 'No Bias'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        for ifiber in range(2):

            if fiberobj_lst[ifiber]=='':
                continue

            fiber = chr(ifiber+65)
            spec1d = extract_aperset(data, mask,
                        apertureset = master_aperset[fiber],
                        lower_limit = 5,
                        upper_limit = 5,
                        )
            message = 'Fiber {}: 1D spectra of {} orders extracted'.format(
                       fiber, len(spec1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            # pack to a structured array
            spec = []
            for aper, item in sorted(spec1d.items()):
                flux = item['flux_sum']
                n = flux.size
                # pack to table
                item = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux,                           # flux
                        np.zeros(n, dtype=np.float32),  # error
                        np.zeros(n, dtype=np.float32),  # background
                        np.zeros(n),                    # mask
                        )
                spec.append(item)
            spec = np.array(spec, dtype=spectype)

            # wavelength calibration
            spec, card_lst = reference_spec_wavelength(spec,
                                [ref_calib_lst[fiber]], [1.0])

            prefix = 'HIERARCH GAMSE WLCALIB '
            for key, value in card_lst:
                head.append((prefix + key, value))

            # pack and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        ])
            fname = '{}_{}_{}.fits'.format(fileid, fiber, oned_suffix)
            filename = os.path.join(odspath, fname)
            hdu_lst.writeto(filename, overwrite=True)

            message = '1D spectra written to "{}"'.format(filename)
            logger.info(logger_prefix + message)
            print(screen_prefix + message)
