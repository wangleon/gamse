import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ...echelle.trace import load_aperture_set
from ...echelle.imageproc import combine_images
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_self_wavelength,
                                select_calib_auto, select_calib_manu,
                                )
from .common import correct_overscan, select_calib_from_database
from .trace import find_apertures
from .flat import get_flat

def reduce_rawdata(config, logtable):

    # extract keywords from config file
    section      = config['data']
    rawpath      = section.get('rawpath')

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

    ##################### parse bias ##################################
    section = config['reduce.bias']
    bias_file = section.get('bias_file')
    if mode=='debug' and os.path.exists(bias_file):
        hdu_lst = fits.open(bias_file)
        bias = hdu_lst[-1].data
        head = hdu_lst[0].header
        hdu_lst.close()

        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        bias_item_lst = [logitem for logitem in logtable
                            if logitem['obstype']=='BIAS']
        n_bias = len(bias_item_lst)
        if n_bias == 0:
            # no bias
            bias = None
        else:
            fmt_str = '  - {:>7s} {:^11} {:^10s} {:^7} {:^23s} {:^6}'
            head_str = fmt_str.format('frameid', 'fileid', 'obstype', 'exptime',
                        'obsdate', 'q95')

            bias_data_lst = []
            bias_card_lst = []

            for ilogitem, logitem in enumerate(bias_item_lst):

                fname = '{}.fits'.format(logitem['fileid'])
                filename = os.path.join(rawpath, fname)
                data, head = fits.getdata(filename, header=True)
                data, mask = correct_overscan(data, head)
                bias_data_lst.append(data)

                # append the file information to header
                prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ilogitem+1)
                card = (prefix+' FILEID', logitem['fileid'])
                bias_card_lst.append(card)

                if ilogitem == 0:
                    print('* Combine Bias Image: "{}"'.format(bias_file))
                    print(head_str)
                message = fmt_str.format(
                            '[{:d}]'.format(logitem['frameid']),
                            logitem['fileid'], logitem['obstype'],
                            logitem['exptime'], logitem['obsdate'],
                            logitem['q95'],
                        )
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
            ny, nx = bias_combine.shape
            bias_mean = bias_combine.mean(axis=0)
            bias = np.repeat([bias_mean], ny, axis=0)

            # plot bias mean in a figure
            fig = plt.figure(dpi=150)
            ax = fig.gca()
            ax.plot(bias_mean, lw=0.5)
            figname = os.path.join(figpath, 'bias.png')
            fig.savefig(figname)
            plt.close(fig)

            bias_card_lst.append((prefix+'COMBINE_MODE', combine_mode))
            bias_card_lst.append((prefix+'COSMIC_CLIP',  cosmic_clip))
            bias_card_lst.append((prefix+'MAXITER',      maxiter))
            bias_card_lst.append((prefix+'MASK_MODE',    str(maskmode)))

            # create new FITS Header for bias
            head = fits.Header()
            for card in bias_card_lst:
                head.append(card)
            head['HIERARCH GAMSE FILECONTENT 0'] = 'BIAS COMBINED'
            head['HIERARCH GAMSE FILECONTENT 1'] = 'BIAS YMEAN'
            # create the hdu list to be saved
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(data=bias_combine, header=head),
                        fits.ImageHDU(data=bias),
                ])
            hdu_lst.writeto(bias_file, overwrite=True)

    ######################### parse flat ##########################
    section = config['reduce.flat']
    flat_file = section.get('flat_file')
    if mode=='debug' and os.path.exists(flat_file):
        hdu_lst = fits.open(flat_file)
        hdu_lst.close()
    else:
        flat_item_lst = [logitem for logitem in logtable
                            if logitem['obstype']=='FLAT']
        n_flat = len(flat_item_lst)
        if n_flat == 0:
            # no flat
            flat = None
        else:
            fmt_str = '  - {:>7s} {:^11} {:^10s} {:^7} {:^23s} {:^6}'
            head_str = fmt_str.format('frameid', 'fileid', 'obstype', 'exptime',
                        'obsdate', 'q95')

            flat_data_lst = []
            flat_card_lst = []

            for ilogitem, logitem in enumerate(flat_item_lst):
                fname = '{}.fits'.format(logitem['fileid'])
                filename = os.path.join(rawpath, fname)
                data, head = fits.getdata(filename, header=True)
                data, mask = correct_overscan(data, head)
                data = data - bias
                flat_data_lst.append(data)

                if ilogitem == 0:
                    print('* Combine Flat Image: "{}"'.format(flat_file))
                    print(head_str)
                message = fmt_str.format(
                            '[{:d}]'.format(logitem['frameid']),
                            logitem['fileid'], logitem['obstype'],
                            logitem['exptime'], logitem['obsdate'],
                            logitem['q95'],
                        )
                print(message)

            # combine flat images
            flat_data_lst = np.array(flat_data_lst)

            combine_mode = 'mean'
            cosmic_clip  = section.getfloat('cosmic_clip')
            maxiter      = section.getint('maxiter')
            maskmode     = (None, 'max')[n_flat>=3]

            flat_data = combine_images(flat_data_lst,
                    mode        = combine_mode,
                    upper_clip  = cosmic_clip,
                    maxiter     = maxiter,
                    maskmode    = maskmode,
                    ncores      = ncores,
                    )

    #################### trace orders ##########################
    section = config['reduce.trace']

    trac_file = os.path.join(midpath, 'trace.txt')
    tracA_file = os.path.join(midpath, 'trace_A.txt')
    tracB_file = os.path.join(midpath, 'trace_B.txt')

    if mode=='debug' \
        and os.path.exists(trac_file) \
        and os.path.exists(tracA_file) \
        and os.path.exists(tracB_file):
        aperset   = load_aperture_set(trac_file)
        aperset_A = load_aperture_set(tracA_file)
        aperset_B = load_aperture_set(tracB_file)
    else:
        aperset, aperset_A, aperset_B = find_apertures(flat_data,
                        scan_step = section.getint('scan_step'),
                        align_deg = section.getint('align_deg'),
                        degree    = section.getint('degree'),
                        mode      = mode,
                        figpath   = figpath,
                        )

        aperset.save_txt(trac_file)
        aperset_A.save_txt(tracA_file)
        aperset_B.save_txt(tracB_file)


    ####################### Parse Flat Fielding ##########################
    flat_file = os.path.join(midpath, 'flat.fits')

    # get sensitivity map and 1d spectra of flat
    flat_sens, flatspec_lst = get_flat(flat_data, aperset)
    ny, nx = flat_sens.shape

    # pack the final 1-d spectra of flat
    flatspectable = [(aper, flatspec) for aper, flatspec
                        in sorted(flatspec_lst.items())]

    # define the datatype of flat 1d spectra
    flatspectype = np.dtype(
                    {'names':   ['aperture', 'flux'],
                     'formats': [np.int32, (np.float32, ny)],
                     })
    flatspectable = np.array(flatspectable, dtype=flatspectype)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(flat_data),
                fits.ImageHDU(flat_sens),
                fits.BinTableHDU(flatspectable),
            ])
    hdu_lst.writeto(flat_file, overwrite=True)

    ##################### define dtype of 1-d spectra ####################

    # get the data shape
    ny, nx = flat_sens.shape

    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('wavelength', (np.float64, nx)),
            ('flux',       (np.float32, nx)),
            ('error',      (np.float32, nx)),
            ('background', (np.float32, nx)),
            ('mask',       (np.int32,   nx)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    ############################ Extract ThAr ###########################

    calib_lst = {}

    # filter ThAr frames
    filter_thar = lambda item: item['obstype'].lower() == 'COMPARISON'

    thar_items = list(filter(filter_thar, logtable))

    for ithar, logitem in enumerate(thar_items):
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        obstype = logitem['obstype']
        objname = logitem['object']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fmt_str = 'FileID: {} ({}) OBJECT: {} - wavelength identification'
        message = fmt_str.format(fileid, obstype, objname)
        logger.info(message)
        print(message)

        fname = '{}.fits'.format(fileid)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)

        data, mask = correct_overscan(data)
        message = 'Overscan corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct bias for ThAr
        data = data - bias
        message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct flat field for ThAr
        satmask = (mask==4)
        data1 = data/flat_sens
        data[~satmask] = data1[~satmask]
        # now non-saturated pixels are flat field corrected.
        # saturated pixels are remained
        message = 'Flat Filed corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract ThAr spectra
        spectra1d = extract_aperset(data, mask,
                        apertureset = aperset,
                        lower_limit = -15,
                        upper_limit = 15,
                        )
        message = '1D spectra extracted for {:d} orders'.format(len(spectra1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
        
        # pack to a structured array
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size

            # pack to table
            row = (aper, 0,                         # aperture and order number
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

        title = '{}.fits'.format(fileid)

        def wlfit_filter(item):
            if item['pixel']>4200:
                return False
            else:
                return True

        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                index_file = os.path.join(os.path.dirname(__file__),
                                '../../data/calib/wlcalib_espadons.dat')

                message = ('Searching for archive wavelength calibration'
                           'file in "{}"'.format(index_file))
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
                        fit_filter       = wlfit_filter,
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
                fit_filter       = wlfit_filter,
                )

        # add more infos in calib
        calib['fileid']   = fileid
        calib['date-obs'] = logitem['obsdate']
        calib['exptime']  = logitem['exptime']
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
