import os
import re
import shutil
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import median_filter
import scipy.interpolate as intp

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat  import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                                mosaic_spec)
from ...echelle.background import find_background, get_interorder_background
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_self_wavelength,
                                select_calib_auto, select_calib_manu,
                                )
from .common import (get_bias, get_mask, correct_overscan,
                    TraceFigure, BackgroundFigure,
                     select_calib_from_database)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg)

def reduce_singlefiber(config, logtable):
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

    # initialize general card list
    general_card_lst = {}
    ############################# parse bias ###################################

    bias, bias_card_lst = get_bias(config, logtable)

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for single fiber
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}

    for logitem in logtable:
        objname = logitem['object'].lower().strip()
        # above only valid for single fiber

        if re.match('^flat[\s\S]*', objname):
            # the object name of the channel matches "flat ???"

            # find a proper name for this flat
            if objname=='flat':
                # no special names given, use "flat_A_15"
                flatname = '{:g}'.format(logitem['exptime'])
            else:
                # flatname is given. replace space with "_"
                # remove "flat" before the objectname. e.g.,
                # "Flat Red" becomes "Red" 
                char = objname[4:].strip()
                flatname = char.replace(' ','_')

            # add flatname to flat_groups
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem)

    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {}
    flat_mask_lst = {}
    flat_norm_lst = {}
    flat_sens_lst = {}
    flat_spec_lst = {}
    flat_info_lst = {}
    aperset_lst   = {}

    # first combine the flats
    for flatname, logitem_lst in flat_groups.items():
        nflat = len(logitem_lst)       # number of flat fieldings

        # single-fiber
        flat_filename    = os.path.join(midpath,
                            'flat_{}.fits'.format(flatname))
        aperset_filename = os.path.join(midpath,
                            'trace_flat_{}.trc'.format(flatname))
        aperset_regname  = os.path.join(midpath,
                            'trace_flat_{}.reg'.format(flatname))
        trace_figname = os.path.join(figpath,
                        'trace_flat_{}.{}'.format(flatname, fig_format))

        # get flat_data and mask_array
        if mode=='debug' and os.path.exists(flat_filename) \
            and os.path.exists(aperset_filename):
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
        else:
            # if the above conditions are not satisfied, comine each flat
            data_lst = []
            head_lst = []
            exptime_lst = []

            print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
            fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^19s} {:^8} {:^6}'
            head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                        'obsdate', 'N(sat)', 'Q95')

            for iframe, logitem in enumerate(logitem_lst):
                # read each individual flat frame
                fname = '{}.fits'.format(logitem['fileid'])
                filename = os.path.join(rawpath, fname)
                data, head = fits.getdata(filename, header=True)
                exptime_lst.append(head[exptime_key])
                mask = get_mask(data, head)
                sat_mask = (mask&4>0)
                bad_mask = (mask&2>0)
                if iframe == 0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += sat_mask

                # correct overscan for flat
                data, card_lst = correct_overscan(data, head, readout_mode)
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

            # get mean exposure time and write it to header
            head = fits.Header()
            exptime = np.array(exptime_lst).mean()
            head[exptime_key] = exptime

            # find saturation mask
            sat_mask = allmask > nflat/2.
            flat_mask = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

            # get exposure time normalized flats
            flat_norm = flat_data/exptime

            # create the trace figure
            tracefig = TraceFigure()

            section = config['reduce.trace']
            aperset = find_apertures(flat_data, flat_mask,
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
            title = 'Trace for {}'.format(flat_filename)
            tracefig.suptitle(title, fontsize=15)
            tracefig.savefig(trace_figname)

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
            figname = 'slit_flat_{}.{}'.format(flatname, fig_format)
            fig_slit = os.path.join(figpath, figname)

            # prepare the name for slit file
            fname = 'slit_flat_{}.dat'.format(flatname)
            slit_file = os.path.join(midpath, fname)

            section = config['reduce.flat']

            flat_sens, flat_spec = get_fiber_flat(
                        data            = flat_data,
                        mask            = flat_mask,
                        apertureset     = aperset,
                        slit_step       = section.getint('slit_step'),
                        nflat           = nflat,
                        q_threshold     = section.getfloat('q_threshold'),
                        smooth_A_func   = smooth_aperpar_A,
                        smooth_k_func   = smooth_aperpar_k,
                        smooth_c_func   = smooth_aperpar_c,
                        smooth_bkg_func = smooth_aperpar_bkg,
                        fig_aperpar     = fig_aperpar,
                        fig_overlap     = None,
                        fig_slit        = fig_slit,
                        slit_file       = slit_file,
                        )

            # pack results and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data, head),
                        fits.ImageHDU(flat_mask),
                        fits.ImageHDU(flat_norm),
                        fits.ImageHDU(flat_sens),
                        fits.BinTableHDU(flat_spec),
                        ])
            hdu_lst.writeto(flat_filename, overwrite=True)

            # now flt_data and mask_array are prepared

        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = flat_mask
        flat_norm_lst[flatname] = flat_norm
        flat_sens_lst[flatname] = flat_sens
        flat_spec_lst[flatname] = flat_spec
        flat_info_lst[flatname] = {'exptime': exptime}
        aperset_lst[flatname]   = aperset

        # continue to the next colored flat

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midpath, 'flat.fits')
    trac_file = os.path.join(midpath, 'trace.trc')
    treg_file = os.path.join(midpath, 'trace.reg')

    if len(flat_groups) == 1:
        # there's only ONE "color" of flat
        flatname = list(flat_groups)[0]

        # copy the flat fits
        fname = 'flat_{}.fits'.format(flatname)
        oriname = os.path.join(midpath, fname)
        shutil.copyfile(oriname, flat_file)

        '''
        # copy the trc file
        oriname = 'trace_flat_{}.trc'.format(flatname)
        shutil.copyfile(os.path.join(midpath, oriname), trac_file)

        # copy the reg file
        oriname = 'trace_flat_{}.reg'.format(flatname)
        shutil.copyfile(os.path.join(midpath, oriname), treg_file)
        '''

        flat_sens = flat_sens_lst[flatname]

        # no need to aperset mosaic
        master_aperset = list(aperset_lst.values())[0]
    else:
        # mosaic apertures
        section = config['reduce.flat']
        # determine the mosaic order
        name_lst = sorted(flat_info_lst,
                    key = lambda x: flat_info_lst.get(x)['exptime'])

        master_aperset = mosaic_flat_auto(
                aperture_set_lst = aperset_lst,
                max_count        = section.getfloat('mosaic_maxcount'),
                name_lst         = name_lst,
                )
        # mosaic original flat images
        flat_data = mosaic_images(flat_data_lst, master_aperset)
        # mosaic flat mask images
        flat_mask = mosaic_images(flat_mask_lst, master_aperset)
        # mosaic exptime-normalized flat images
        flat_norm = mosaic_images(flat_norm_lst, master_aperset)
        # mosaic sensitivity map
        flat_sens = mosaic_images(flat_sens_lst, master_aperset)
        # mosaic 1d spectra of flats
        flat_spec = mosaic_spec(flat_spec_lst, master_aperset)

        zeromask = (flat_sens == 0.0)
        flat_sens[zeromask] = 1.0

        # pack and save to fits file
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(flat_data),
                    fits.ImageHDU(flat_mask),
                    fits.ImageHDU(flat_norm),
                    fits.ImageHDU(flat_sens),
                    fits.BinTableHDU(flat_spec),
                    ])
        hdu_lst.writeto(flat_file, overwrite=True)

        master_aperset.save_txt(trac_file)
        master_aperset.save_reg(treg_file)

    ############################## Extract ThAr ################################

    # get the data shape
    ny, nx = flat_sens.shape

    # define dtype of 1-d spectra
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
        data, card_lst = correct_overscan(data, head, readout_mode)
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
        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )
        head = master_aperset.to_fitsheader(head)
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
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=wlcalib_spectype)
   
        figname = 'wlcalib_{}.{}'.format(fileid, fig_format)
        wlcalib_fig = os.path.join(figpath, figname)

        section = config['reduce.wlcalib']

        title = '{}.fits'.format(fileid)

        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                database_path = section.get('database_path')
                database_path = os.path.expanduser(database_path)

                message = ('Searching for archive wavelength calibration'
                           'file in "{}"'.format(database_path))
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

                ref_spec, ref_calib = select_calib_from_database(
                            database_path, head[statime_key])
    
                if ref_spec is None or ref_calib is None:

                    message = 'Archive wavelength calibration file not found'
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

                    # determine the name of the output figure during lamp shift
                    # finding.
                    if mode == 'debug':
                        figname1 = 'lamp_ccf_{:+2d}_{:+03d}.png'
                        figname2 = 'lamp_ccf_scatter.png'
                        fig_ccf     = os.path.join(figpath, figname1)
                        fig_scatter = os.path.join(figpath, figname2)
                    else:
                        fig_ccf     = None
                        fig_scatter = None

                    result = find_caliblamp_offset(ref_spec, spec,
                                aperture_k  = aperture_k,
                                pixel_k     = pixel_k,
                                fig_ccf     = fig_ccf,
                                fig_scatter = fig_scatter,
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

    '''
    ############################## Extract Flat ################################
    flat_norm = flat_norm/flat_map
    section = config['reduce.extract']
    spectra1d = extract_aperset(flat_norm, mask_data,
                apertureset = master_aperset,
                lower_limit = section.getfloat('lower_limit'),
                upper_limit = section.getfloat('upper_limit'),
                )
    # pack spectrum
    spec = []
    for aper, item in sorted(spectra1d.items()):
        flux_sum = item['flux_sum']
        spec.append((aper, 0, flux_sum.size,
            np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
    spec = np.array(spec, dtype=spectype)
    
    # wavelength calibration
    weight_lst = get_time_weight(ref_datetime_lst, head[statime_key])
    head = fits.Header()
    spec, head = wl_reference_singlefiber(spec, head, ref_calib_lst, weight_lst)

    # pack and save wavelength referenced spectra
    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(header=head),
                fits.BinTableHDU(spec),
                ])
    filename = os.path.join(odspath, 'flat'+oned_suffix+'.fits')
    hdu_lst.writeto(filename, overwrite=True)
    '''

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
    extr_filter = config['reduce.extract'].get('extract',
                        'lambda row: row["imgtype"]=="sci"')
    extr_filter = eval(extr_filter)
    extr_items = list(filter(extr_filter, logtable))

    #################### Extract Science Spectrum ##############################
    for logitem in extr_items:

        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objname = logitem['object']
        exptime = logitem['exptime']

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
        data, card_lst = correct_overscan(data, head, readout_mode)
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
        background = get_interorder_background(data, mask, master_aperset)
        for y in np.arange(ny):
            m = mask[y,:]==0
            f = intp.InterpolatedUnivariateSpline(
                    allx[m], background[y,:][m], k=3)
            background[y,:][~m] = f(allx[~m])
        background = median_filter(background, size=(9,1), mode='nearest')
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

        ## correct background
        #section = config['reduce.background']
        #fig_sec = os.path.join(figpath,
        #            'bkg_{}_sec.{}'.format(fileid, fig_format))
        #stray = find_background(data, mask,
        #        apertureset_lst = master_aperset,
        #        ncols           = section.getint('ncols'),
        #        distance        = section.getfloat('distance'),
        #        yorder          = section.getint('yorder'),
        #        fig_section     = fig_sec,
        #        )
        #data = data - stray
        ####
        #outfilename = os.path.join(midpath, '%s_bkg.fits'%fileid)
        #fits.writeto(outfilename, data)
        ## plot stray light
        #fig_stray = os.path.join(figpath,
        #            'bkg_{}_stray.{}'.format(fileid, fig_format))
        #plot_background_aspect1(data+stray, stray, fig_stray)

        data = data - background
        message = 'Background corrected. Max = {:.2f}; Mean = {:.2f}'.format(
                    background.max(), background.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectrum
        section = config['reduce.extract']
        lower_limit = section.getfloat('lower_limit')
        upper_limit = section.getfloat('upper_limit')

        # extract 1d spectra of the object
        spectra1d = extract_aperset(data, mask,
                        apertureset = master_aperset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        message = '1D spectra of {} orders extracted'.format(len(spectra1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectra for straylight/background light
        background1d = extract_aperset(background, mask,
                        apertureset = master_aperset,
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
                    np.zeros(n, dtype=np.int16),    # mask
                    back_flux,                      # background
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
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
