import os
import re
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                            mosaic_spec)
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, select_calib_from_database,
                                get_time_weight, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_self_wavelength)
from ...echelle.background import find_background, simple_debackground
from ...utils.obslog import parse_num_seq
from ..common import plot_background_aspect1
from .common import (print_wrapper, get_mask, get_bias,
                    correct_overscan, TraceFigure)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg)

def reduce_singlefiber(config, logtable):
    """Data reduction for single-fiber configuration.

    Args:
        config (:class:`configparser.ConfigParser`): The configuration of
            reduction.
        logtable (:class:`astropy.table.Table`): The observing log.

    """

    # extract keywords from config file
    section = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')
    section = config['reduce']
    midproc     = section.get('midproc')
    onedspec    = section.get('onedspec')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):   os.mkdir(report)
    if not os.path.exists(onedspec): os.mkdir(onedspec)
    if not os.path.exists(midproc):  os.mkdir(midproc)

    ################################ parse bias ################################
    bias, bias_card_lst = get_bias(config, logtable)

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for both single fiber and multi-fibers
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}

    for logitem in logtable:
        objname = logitem['object'].lower().strip()

        if re.match('^flat[\s\S]*', objname):
            # the object name of the channel matches "flat ???"
            
            # find a proper name (flatname) for this flat
            if objname=='flat':
                # no special names given, use exptime
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
    flat_dsum_lst = {}
    flat_sens_lst = {}
    flat_spec_lst = {}
    flat_info_lst = {}
    aperset_lst   = {}

    # first combine the flats
    for flatname, item_lst in sorted(flat_groups.items()):
        # number of flat fieldings
        nflat = len(item_lst)

        # single-fiber
        flat_filename = os.path.join(midproc,
                        'flat_{}.fits'.format(flatname))
        aperset_filename = os.path.join(midproc,
                        'trace_flat_{}.trc'.format(flatname))
        aperset_regname = os.path.join(midproc,
                        'trace_flat_{}.reg'.format(flatname))
        trace_figname = os.path.join(report,
                        'trace_flat_{}.{}'.format(flatname, fig_format))

        # get flat_data and mask_array for each flat group
        if mode=='debug' and os.path.exists(flat_filename) \
            and os.path.exists(aperset_filename):
            # read flat data and mask array
            hdu_lst = fits.open(flat_filename)
            flat_data = hdu_lst[0].data
            flat_mask = hdu_lst[1].data
            flat_norm = hdu_lst[2].data
            flat_dsum = hdu_lst[3].data
            flat_sens = hdu_lst[4].data
            flat_spec = hdu_lst[5].data
            exptime   = hdu_lst[0].header[exptime_key]
            hdu_lst.close()
            aperset = load_aperture_set(aperset_filename)
        else:
            # if the above conditions are not satisfied, comine each flat
            data_lst = []
            head_lst = []
            exptime_lst = []

            print('* Combine {} Flat Images: {}'.format(
                    nflat, flat_filename))

            for i_item, logitem in enumerate(item_lst):
                # read each individual flat frame
                filename = os.path.join(rawdata, logitem['fileid']+'.fits')
                data, head = fits.getdata(filename, header=True)
                exptime_lst.append(head[exptime_key])
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data)

                # generate the mask for all images
                sat_mask = (mask&4>0)
                bad_mask = (mask&2>0)
                if i_item == 0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += sat_mask

                # correct overscan for flat
                data, card_lst, overmean, overstd = correct_overscan(
                                                    data, mask, direction)
                # head['BLANK'] is only valid for integer arrays.
                if 'BLANK' in head:
                    del head['BLANK']
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
                message_lst = [
                        '  - FileID: {}'.format(logitem['fileid']),
                        logitem['object'],
                        'exptime = {:<5g}'.format(logitem['exptime']),
                        'Nsat = {:<6d}'.format(logitem['nsat']),
                        'Q95 = {:<5d}'.format(logitem['q95']),
                        ]
                print('    '.join(message_lst))

                data_lst.append(data)

            if nflat == 1:
                flat_data = data_lst[0]
                flat_dsum = data_lst[0]
            else:
                data_lst = np.array(data_lst)
                flat_data = combine_images(data_lst,
                                mode       = 'mean',
                                upper_clip = 10,
                                maxiter    = 5,
                                maskmode   = (None, 'max')[nflat>3],
                                )
                flat_dsum = flat_data*nflat

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

            # if debackground before detecting the orders, then we lose the 
            # ability to detect the weak blue orders.
            #xnodes = np.arange(0, flat_data.shape[1], 200)
            #flat_debkg = simple_debackground(flat_data, mask_array, xnodes,
            # smooth=5)
            #aperset = find_apertures(flat_debkg, mask_array,
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

            # do flat fielding
            # prepare the output midproc figures in debug mode
            if mode=='debug':
                figname = 'flat_aperpar_{}_%03d.{}'.format(
                            flatname, fig_format)
                fig_aperpar = os.path.join(report, figname)
            else:
                fig_aperpar = None

            # prepare the name for slit figure
            figname = 'slit_flat_{}.{}'.format(
                        flatname, fig_format)
            fig_slit = os.path.join(report, figname)

            # prepare the name for slit file
            fname = 'slit_flat_{}.dat'.format(flatname)
            slit_file = os.path.join(midproc, fname)

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
                        fits.ImageHDU(flat_dsum),
                        fits.ImageHDU(flat_sens),
                        fits.BinTableHDU(flat_spec),
                        ])
            hdu_lst.writeto(flat_filename, overwrite=True)

            # now flt_data and mask_array are prepared


        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = flat_mask
        flat_norm_lst[flatname] = flat_norm
        flat_dsum_lst[flatname] = flat_dsum
        flat_sens_lst[flatname] = flat_sens
        flat_spec_lst[flatname] = flat_spec
        flat_info_lst[flatname] = {'exptime': exptime}
        aperset_lst[flatname]   = aperset

        # continue to the next colored flat

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    if len(flat_groups) == 1:
        # there's only ONE "color" of flat
        flatname = list(flat_groups)[0]

        # copy the flat fits
        fname = 'flat_{}.fits'.format(flatname)
        oriname = os.path.join(midproc, fname)
        shutil.copyfile(oriname, flat_file)

        '''
        # copy the trc file
        oriname = 'trace_flat_{}.trc'.format(flatname)
        shutil.copyfile(os.path.join(midproc, oriname), trac_file)

        # copy the reg file
        oriname = 'trace_flat_{}.reg'.format(flatname)
        shutil.copyfile(os.path.join(midproc, oriname), treg_file)
        '''

        flat_sens = flat_sens_lst[flatname]
    
        # no need to mosaic aperset
        master_aperset = list(aperset_lst.values())[0]
    else:
        # mosaic apertures
        section = config['reduce.flat']
        # determine the mosaic order
        name_lst = sorted(flat_info_lst,
                    key=lambda x: flat_info_lst.get(x)['exptime'])

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
        # mosaic summed flat images
        flat_dsum = mosaic_images(flat_dsum_lst, master_aperset)
        # mosaic sensitivity map
        flat_sens = mosaic_images(flat_sens_lst, master_aperset)
        # mosaic 1d spectra of flats
        flat_spec = mosaic_spec(flat_spec_lst, master_aperset)

        # pack and save to fits file
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(flat_data),
                    fits.ImageHDU(flat_mask),
                    fits.ImageHDU(flat_norm),
                    fits.ImageHDU(flat_dsum),
                    fits.ImageHDU(flat_sens),
                    fits.BinTableHDU(flat_spec),
                    ])
        hdu_lst.writeto(flat_file, overwrite=True)

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
            ('mask',       (np.int32, nx)),
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
        imgtype = logitem['imgtype']
        objname = logitem['object']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        message = ('FileID: {} ({}) OBJECT: {{{}}} - wavelength '
                   'identification'.format(fileid, imgtype, objname))
        logger.info(message)
        print(message)

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)

        head.append(('HIERARCH GAMSE CCD GAIN', 1.0))
        # correct overscan for ThAr
        data, card_lst, overmean, overstd = correct_overscan(
                                            data, mask, direction)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']
        for key, value in card_lst:
            head.append((key, value))

        message = 'Overscan corrected. Mean = {:.2f}'.format(overmean)
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

        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )
        head = master_aperset.to_fitsheader(head)
    
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size
            # search for flat flux
            m = flat_spec['aperture']==aper
            flat_flux = flat_spec[m][0]['flux']

            # pack to table
            item = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux
                    np.zeros(n),                    # mask
                    )
            spec.append(item)
        spec = np.array(spec, dtype=wlcalib_spectype)
    
        figname = 'wlcalib_{}.{}'.format(fileid, fig_format)
        wlcalib_fig = os.path.join(report, figname)

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
                        database_path, statime_key, head[statime_key])
    
                if ref_spec is None or ref_calib is None:

                    message = ('Did not find any archive wavelength'
                               'calibration file')
                    logger.info(logger_prefix + message)
                    print(screen_prefix + message)

                    # if failed, pop up a calibration window and
                    # identify the wavelengths manually
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
                    # determine the direction
                    message = 'Found archive wavelength calibration file'
                    logger.info(logger_prefix + message)
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
                        fig_ccf     = os.path.join(report, figname1)
                        fig_scatter = os.path.join(report, figname2)
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

            # then use this ThAr as the reference
            ref_calib = calib
            ref_spec  = spec
        else:
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
        
        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(spec, calib)

        for key, value in card_lst:
            key = 'HIERARCH GAMSE WLCALIB '+key
            head.append((key, value))
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    fits.BinTableHDU(identlist),
                    ])
        # save in midproc as a wlcalib reference file
        fname = 'wlcalib.{}.fits'.format(fileid)
        filename = os.path.join(midproc, fname)
        hdu_lst.writeto(filename, overwrite=True)

        # save a second time in onedspec
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(onedspec, fname)
        hdu_lst.writeto(filename, overwrite=True)

        # pack to calib_lst
        calib_lst[logitem['frameid']] = calib

    # print fitting summary
    fmt_string = (' [{:3d}] {}'
                    ' - ({:4g} sec)'
                    ' - {:4d}/{:4d} r.m.s. = {:7.5f}')
    for frameid, calib in sorted(calib_lst.items()):
        print(fmt_string.format(frameid, calib['fileid'], logitem['exptime'],
            calib['nuse'], calib['ntot'], calib['std']))
    
    # print promotion and read input frameid list
    while(True):
        string = input('Select References: ')
        ref_frameid_lst  = []
        ref_calib_lst    = []
        ref_datetime_lst = []
        succ = True
        for s in string.split(','):
            s = s.strip()
            if len(s)>0 and s.isdigit() and int(s) in calib_lst:
                frameid = int(s)
                calib   = calib_lst[frameid]
                ref_frameid_lst.append(frameid)
                ref_calib_lst.append(calib)
                ref_datetime_lst.append(calib['date-obs'])
            else:
                print('Warning: "{}" is an invalid calib frame'.format(s))
                succ = False
                break
        if succ:
            break
        else:
            continue

    #################### Extract Science Spectrum ##############################

    for logitem in logtable:
        
        # logitem alias
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objname = logitem['object']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        if imgtype != 'sci':
            continue

        filename = os.path.join(rawdata, fileid+'.fits')

        message = 'FileID: {} ({}) OBJECT: {{{}}} - start reduction'.format(
                    fileid, imgtype, objname)
        logger.info(message)
        print(message)

        # read raw data
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)

        head.append(('HIERARCH GAMSE CCD GAIN', 1.0))

        # correct overscan
        data, card_lst, overmean, overstd = correct_overscan(
                                            data, mask, direction)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']
        for key, value in card_lst:
            head.append((key, value))
        message = 'Overscan corrected. Mean = {:.2f}'.format(overmean)
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

        # background correction
        section = config['reduce.background']
        ncols    = section.getint('ncols')
        distance = section.getfloat('distance')
        yorder   = section.getint('yorder')
        subtract = section.getboolean('subtract')
        excluded_frameids = section.get('excluded_frameids')
        excluded_frameids = parse_num_seq(excluded_frameids)

        if (subtract and frameid not in excluded_frameids) or \
           (not subtract and frameid in excluded_frameids):

            figname = 'bkg_{}_sec.{}'.format(fileid, fig_format)
            fig_sec = os.path.join(report, figname)

            stray = find_background(data, mask,
                            aperturesets = master_aperset,
                            ncols        = ncols,
                            distance     = distance,
                            yorder       = yorder,
                            fig_section  = fig_sec,
                    )
            data = data - stray

            # put information into header
            prefix = 'HIERARCH GAMSE BACKGROUND '
            head.append((prefix + 'CORRECTED', True))
            head.append((prefix + 'XMETHOD',   'cubic spline'))
            head.append((prefix + 'YMETHOD',   'polynomial'))
            head.append((prefix + 'NCOLUMN',   ncols))
            head.append((prefix + 'DISTANCE',  distance))
            head.append((prefix + 'YORDER',    yorder))

            # plot stray light
            figname = 'bkg_{}_stray.{}'.format(fileid, fig_format)
            fig_stray = os.path.join(report, figname)
            plot_background_aspect1(data+stray, stray, fig_stray)

            message = 'background corrected. max value = {}'.format(
                        stray.max())
        else:
            stray = None
            # put information into header
            prefix = 'HIERARCH GAMSE BACKGROUND '
            head.append((prefix + 'CORRECTED', False))
            message = 'background not corrected'

        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectrum
        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )

        message = '1D spectra of {} orders extracted'.format(len(spectra1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # pack spectrum
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size
            # seach for flat flux
            m = flat_spec['aperture']==aper
            flat_flux = flat_spec[m][0]['flux']
            item = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux_sum
                    flat_flux,                      # 1d spectra of flat
                    stray1d[aper]['flux_sum'],      # 1d spectra of background
                    )
            spec.append(item)
        spec = np.array(spec, dtype=spectype)

        # wavelength calibration
        weight_lst = get_time_weight(ref_datetime_lst, head[statime_key])

        message = 'Wavelength calibration: weights={}'.format(
                    ','.join(['{:8.4f}'.format(w) for w in weight_lst]))
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
        filename = os.path.join(onedspec, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
