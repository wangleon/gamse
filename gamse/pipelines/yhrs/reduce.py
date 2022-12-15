import os
import shutil
import datetime
import dateutil.parser
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import gaussian_filter, median_filter
import scipy.interpolate as intp
from scipy.signal import savgol_filter

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat import (mosaic_flat_auto, mosaic_images, mosaic_spec,
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
from ..common import load_obslog, load_config
from .common import (print_wrapper, correct_overscan, get_mask,
        TraceFigure, AlignFigure, SpatialProfileFigure,
        select_calib_from_database)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg, get_flat)

def reduce_rawdata():
    """Reduce the Lijiang2.4m HRS data.
    """

    # read obslog and config
    config = load_config('YHRS\S*\.cfg$')
    logtable = load_obslog('log\.\S*\.txt$', fmt='astropy')

    # extract keywords from config file
    section = config['data']
    rawpath     = section.get('rawpath')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')

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

    ### correct bias #####
    section = config['reduce.bias']
    bias_file = section.get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        hdu_lst = fits.open(bias_file)
        bias = hdu_lst[-1].data
        head = hdu_lst[0].header
        hdu_lst.close()

        # message
        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        bias_data_lst = []
        bias_card_lst = []

        filterfunc = lambda item: item['object'] is not np.ma.masked and \
                                  item['object'].lower()=='bias' and \
                                  item['speed'].lower()=='slow'

        bias_items = list(filter(filterfunc, logtable))
        # get the number of bias images
        n_bias = len(bias_items)

        if n_bias == 0:
            # there is no bias frames
            bias = None
        else:
            fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^21s} {:5s}'
            head_str = fmt_str.format('frameid', 'FileID', 'Object',
                        'exptime', 'obsdate', 'speed')

            for iframe, logitem in enumerate(bias_items):
                frameid  = logitem['frameid']
                fileid   = logitem['fileid']
                _objname = logitem['object']
                objectname = '' if _objname is np.ma.masked else _objname
                exptime  = logitem['exptime']
                obsdate  = logitem['obsdate']
                speed    = logitem['speed']

                # now filter the bias frames
                fname = '{}.fits'.format(fileid)
                filename = os.path.join(rawpath, fname)
                data, head = fits.getdata(filename, header=True)
                data = correct_overscan(data, head)

                # pack the data and fileid list
                bias_data_lst.append(data)

                # append the file information
                prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(iframe+1)
                card = (prefix+' FILEID', fileid)
                bias_card_lst.append(card)

                # print info
                if iframe == 0:
                    print('* Combine Bias Images: "{}"'.format(bias_file))
                    print(head_str)
                message = fmt_str.format('[{:d}]'.format(frameid),
                            fileid, objectname, exptime, obsdate, speed)
                print(message)

            prefix = 'HIERARCH GAMSE BIAS '
            bias_card_lst.append((prefix + 'NFILE', n_bias))

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            combine_mode = 'mean'
            cosmic_clip  = section.getfloat('cosmic_clip')
            maxiter      = section.getint('maxiter')
            maskmode    = (None, 'max')[n_bias>=3]

            bias_combine = combine_images(bias_data_lst,
                    mode        = combine_mode,
                    upper_clip  = cosmic_clip,
                    maxiter     = maxiter,
                    maskmode    = maskmode,
                    ncores      = ncores,
                    )

            bias_card_lst.append((prefix+'COMBINE_MODE', combine_mode))
            bias_card_lst.append((prefix+'COSMIC_CLIP',  cosmic_clip))
            bias_card_lst.append((prefix+'MAXITER',      maxiter))
            bias_card_lst.append((prefix+'MASK_MODE',    str(maskmode)))

            # create the hdu list to be saved
            hdu_lst = fits.HDUList()
            # create new FITS Header for bias
            head = fits.Header()
            for card in bias_card_lst:
                head.append(card)
            head['HIERARCH GAMSE FILECONTENT 0'] = 'BIAS COMBINED'
            hdu_lst.append(fits.PrimaryHDU(data=bias_combine, header=head))

            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                ny, nx = bias_combine.shape
                newcard_lst = []
                if smooth_method in ['gauss', 'gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')
                    bias_smooth = gaussian_filter(bias_combine,
                                        sigma = smooth_sigma,
                                        mode  = smooth_mode)

                    # write information to FITS header
                    newcard_lst.append((prefix+'SMOOTH CORRECTED',  True))
                    newcard_lst.append((prefix+'SMOOTH METHOD', 'GAUSSIAN'))
                    newcard_lst.append((prefix+'SMOOTH SIGMA',  smooth_sigma))
                    newcard_lst.append((prefix+'SMOOTH MODE',   smooth_mode))
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                # pack the cards to bias_card_lst and also hdu_lst
                for card in newcard_lst:
                    hdu_lst[0].header.append(card)
                    bias_card_lst.append(card)
                hdu_lst.append(fits.ImageHDU(data=bias_smooth))
                # update the file contents in primary HDU
                card = ('HIERARCH GAMSE FILECONTENT 1', 'BIAS SMOOTHED')
                hdu_lst[0].header.append(card)

                # bias is the result array to return
                bias = bias_smooth

            else:
                # bias not smoothed
                card = (prefix+'SMOOTH CORRECTED', False)
                bias_card_lst.append(card)
                hdu_lst[0].header.append(card)

                # bias is the result array to return
                bias = bias_combine

            # save to FITS
            hdu_lst.writeto(bias_file, overwrite=True)

            message = 'Bias image written to "{}"'.format(bias_file)
            logger.info(message)
            print(message)

    ########################################################
    ndisp = 4096
    types = [
            ('aperture',    np.int16),
            ('order',       np.int16),
            ('wavelength',  (np.float64, ndisp)),
            ('flux',        (np.float32, ndisp)),
            ('error',       (np.float32, ndisp)),
            ('background',  (np.float32, ndisp)),
            ('mask',        (np.int16,   ndisp)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    ######## Combine the flats and trace the orders #########
    flat_data_lst = {}
    flat_mask_lst = {}
    flat_norm_lst = {}
    flat_sens_lst = {}
    flat_spec_lst = {}
    flat_info_lst = {}
    aperset_lst   = {}

    ############ find flat groups ##################
    # initialize flat_groups for single fiber
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}

    for logitem in logtable:
        speed   = logitem['speed'].lower().strip()
        if logitem['object'] is np.ma.masked:
            continue
        objname = logitem['object'].lower().strip()

        if objname.lower()=='flat' and speed=='slow':
            flatname = '{:g}sec'.format(logitem['exptime'])

            # add flatname to flat_groups
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem)

    p1, p2, pstep = -10, 10, 0.1
    profile_x = np.arange(p1, p2+1e-4, pstep)
    disp_x_lst = np.arange(48, ndisp, 500)

    all_profile_lst = {}
    for flatname, logitem_lst in flat_groups.items():
        nflat = len(logitem_lst)    # number of flat files

        flat_filename    = os.path.join(midpath,
                            'flat_{}.fits'.format(flatname))
        aperset_filename = os.path.join(midpath,
                            'trace_flat_{}.trc'.format(flatname))
        aperset_regname  = os.path.join(midpath,
                            'trace_flat_{}.reg'.format(flatname))
        trace_figname = os.path.join(figpath,
                            'trace_flat_{}.{}'.format(flatname, fig_format))
        profile_filename = os.path.join(midpath,
                            'profile_flat_{}.fits'.format(flatname))

        # get flat_data and mask_array
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
            exptime = hdu_lst[0].header[exptime_key]
            hdu_lst.close()
            aperset = load_aperture_set(aperset_filename)
            disp_x_lst, profile_x, profile_lst = read_crossprofile(
                    profile_filename)
        else:
            # if the above conditions are not satisfied, comine each flat
            data_lst = []
            head_lst = []
            exptime_lst = []
            obsdate_lst = []

            print('* Combine {} Flat iamges: {}'.format(nflat, flat_filename))
            fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^21s} {:5s} {:8} {:6}'
            head_str = fmt_str.format('frameid', 'FileID', 'Object',
                        'exptime', 'obsdate', 'speed', 'N(sat)', 'Q95')

            for iframe, logitem in enumerate(logitem_lst):
                fileid = logitem['fileid']
                # read each individual flat frame
                fname = '{}.fits'.format(fileid)
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
                data = correct_overscan(data, head)

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
                            logitem['speed'],
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

            tracefig = TraceFigure(datashape=flat_data.shape)
            alignfig = AlignFigure()    # create the align figure

            section = config['reduce.trace']
            aperset = find_apertures(flat_data, flat_mask,
                        scan_step  = section.getint('scan_step'),
                        minimum    = section.getfloat('minimum'),
                        separation = section.get('separation'),
                        align_deg  = section.getint('align_deg'),
                        conv_core  = 0,
                        fill       = False,
                        filling    = section.getfloat('filling'),
                        recenter   = 'threshold',
                        degree     = section.getint('degree'),
                        display    = section.getboolean('display'),
                        fig_trace  = tracefig,
                        fig_align  = alignfig,
                        )

            flat_fname = os.path.basename(flat_filename)

            # save the trace figure
            tracefig.adjust_positions()
            title = 'Order tracing for {}'.format(flat_fname)
            tracefig.suptitle(title, fontsize=15)
            tracefig.savefig(trace_figname)

            # save the alignment figure
            alignfig.adjust_axes()
            title = 'Order Alignment for {}'.format(flat_fname)
            alignfig.suptitle(title, fontsize=12)
            align_figname = os.path.join(figpath,
                    'align_{}.{}'.format(flatname, fig_format))
            alignfig.savefig(align_figname)
            alignfig.close()

            aperset.save_txt(aperset_filename)
            aperset.save_reg(aperset_regname)

            section = config['reduce.flat']


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
            figname = os.path.join(figpath,
                    'spatial_profile_{}.png'.format(flatname))
            title = 'Spatial Profile of Flat using {}'.format(flat_filename)
            fig_spatial.suptitle(title)
            fig_spatial.savefig(figname)
            fig_spatial.close()

            # pack 1-d spectra of flat
            flat_spec = []
            for aper, flatspec in sorted(flatspec_lst.items()):
                n = flatspec.size

                # get the indices of not NaN values in flatspec
                idx_notnan = np.nonzero(~np.isnan(flatspec))[0]
                if idx_notnan.size > n/2:
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
                    flatspec[i1:i2] = newspec

                row = (aper, 0,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flatspec,                       # flux
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
                        fits.PrimaryHDU(flat_data, head),
                        fits.ImageHDU(flat_mask),
                        fits.ImageHDU(flat_norm),
                        fits.ImageHDU(flat_sens),
                        fits.BinTableHDU(flat_spec),
                        ])
            hdu_lst.writeto(flat_filename, overwrite=True)

        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = flat_mask
        flat_norm_lst[flatname] = flat_norm
        flat_sens_lst[flatname] = flat_sens
        flat_spec_lst[flatname] = flat_spec
        flat_info_lst[flatname] = {'exptime': exptime}
        aperset_lst[flatname]   = aperset
        all_profile_lst[flatname] = profile_lst

    ############## Mosaic Flats ###############
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

    ############## average cross-order profiles  ###############
    profile = np.array([all_profile_lst[flatname] for flatname in flat_groups])
    profile = profile.mean(axis=0)

    #################### Extract ThAr #######################

    # get the data shape
    ny, nx = flat_sens.shape

    calib_lst = {}

    # filter ThAr frames
    filter_thar = lambda item: item['object'] is not np.ma.masked and \
                                item['object'].lower()=='thar' and \
                                item['speed'].lower()=='slow'

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
        data = correct_overscan(data, head)

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
        lower_limit = 10
        upper_limit = 10
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
            row = (aper, 0,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux
                    np.zeros(n, dtype=np.float32),  # error
                    np.zeros(n, dtype=np.float32),  # background
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=spectype)


        # temporarily added
        hdu_lst = fits.HDUList([
            fits.PrimaryHDU(header=head),
            fits.BinTableHDU(spec),
            ])
        hdu_lst.writeto('wlcalib_{}_ods.fits'.format(fileid), overwrite=True)
        exit()

        figname = 'wlcalib_{}.{}'.format(fileid, fig_format)
        wlcalib_fig = os.path.join(figpath, figname)

        section = config['reduce.wlcalib']

        title = 'Wavelength Indentification for {}.fits'.format(fileid)

        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                index_file = os.path.join(os.path.dirname(__file__),
                                '../../data/calib/wlcalib_yhrs.dat')

                message = ('Searching for archive wavelength calibration'
                           'file in "{}"'.format(
                               os.path.basename(index_file)))
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

                ref_spec, ref_calib = select_calib_from_database(
                            index_file, head[statime_key])
            else:
                message = 'No database searching. Identify lines manually'
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

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

    exit()
    ################# trace ##################
    # fiter flat frames
    filterfunc = lambda item: item['object'] is not np.ma.masked and \
                        item['object'].lower()=='flat' and \
                        item['speed'].lower()=='slow'
    logitem_lst = list(filter(filterfunc, logtable))

    nflat = len(logitem_lst)

    flat_filename    = os.path.join(midpath, 'flat.fits')
    aperset_filename = os.path.join(midpath, 'trace.trc')
    aperset_regname  = os.path.join(midpath, 'trace.reg')
    trace_figname    = os.path.join(figpath, 'trace.{}'.format(fig_format))
    profile_filename = os.path.join(midpath, 'profile.fits')

    if mode=='debug' and os.path.exists(flat_filename) \
        and os.path.exists(aperset_filename) \
        and os.path.exists(profile_filename):
        pass
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
            fileid = logitem['fileid']
            # read each individual flat frame
            fname = '{}.fits'.format(fileid)
            filename = os.path.join(rawpath, fname)
            data, head = fits.getdata(filename, header=True)
            exptime_lst.append(head[exptime_key])
            obsdate_lst.append(dateutil.parser.parse(head[statime_key]))

            # correct overscan for flat
            data = correct_overscan(data, head)

            # correct bias for flat, if has bias
            if bias is None:
                message = 'No bias. skipped bias correction'
            else:
                data = data - bias
                message = 'Bias corrected'
            logger.info(message)

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

        flat_head = fits.Header()

        # calculat the mean exposure time and write it to the new header
        exptime = np.mean(exptime_lst)
        flat_head[exptime_key] = exptime

        # calculate the mean start time and write it to the new header
        delta_t_lst = [(t-obsdate_lst[0]).total_seconds() for t in obsdate_lst]
        mean_delta_t = datetime.timedelta(seconds=np.mean(delta_t_lst))
        mean_obsdate = obsdate_lst[0] + mean_delta_t
        flat_head[statime_key] = mean_obsdate.isoformat()

    flat_mask = np.zeros_like(flat_data, dtype=np.int16)

    tracefig = TraceFigure(datashape=flat_data.shape)
    alignfig = AlignFigure()    # create the align figure

    section = config['reduce.trace']
    aperset = find_apertures(flat_data, flat_mask,
                scan_step  = section.getint('scan_step'),
                minimum    = section.getfloat('minimum'),
                separation = section.get('separation'),
                align_deg  = section.getint('align_deg'),
                conv_core  = 0,
                fill       = False,
                filling    = section.getfloat('filling'),
                recenter   = 'threshold',
                degree     = section.getint('degree'),
                display    = section.getboolean('display'),
                fig_trace  = tracefig,
                fig_align  = alignfig,
                )

    # save the trace figure
    tracefig.adjust_positions()
    title = 'Order tracing using Flat'
    tracefig.suptitle(title, fontsize=15)
    trace_figname = os.path.join(figpath, 'trace.{}'.format(fig_format))
    tracefig.savefig(trace_figname)
    tracefig.close()

    # save the alignment figure
    alignfig.adjust_axes()
    title = 'Order Alignment for {}'.format(fname)
    alignfig.suptitle(title, fontsize=12)
    align_figname = os.path.join(figpath, 'align.{}'.format(fig_format))
    alignfig.savefig(align_figname)
    alignfig.close()

    aperset_filename = os.path.join(midpath, 'trace.trc')
    aperset_regname  = os.path.join(midpath, 'trace.reg')

    aperset.save_txt(aperset_filename)
    aperset.save_reg(aperset_regname)


    section = config['reduce.flat']

    p1, p2, pstep = -10, 10, 0.1
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




    # pack results and save to fits
    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(flat_data, flat_head),
                fits.ImageHDU(flat_mask),
                ])
    hdu_lst.writeto(flat_filename, overwrite=True)
