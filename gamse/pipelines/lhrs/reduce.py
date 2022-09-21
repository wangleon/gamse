import re
import os
import datetime
import dateutil.parser
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import median_filter
import scipy.interpolate as intp

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import load_aperture_set
from ...echelle.background import get_interorder_background
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib,
        reference_self_wavelength, reference_spec_wavelength,
        get_calib_weight_lst, find_caliblamp_offset,
        select_calib_from_database, select_calib_auto,
        FWHMMapFigure, ResolutionFigure,
        )
from ..common import load_obslog, load_config
from .common import (print_wrapper, correct_overscan, get_mask,
        BackgroundFigure, TraceFigure, AlignFigure)
from ...echelle.trace import find_apertures
#from .trace import find_apertures

def reduce_rawdata():
    """Reduc the LAMOST-HRS data.
    """

    # read obslog and config
    config = load_config('LHRS\S*\.cfg$')
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

    ############## correct bias #####
    mode = config['reduce'].get('mode')
    bias_file = config['reduce.bias'].get('bias_file')
    
    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        bias = hdu_lst[-1].data
        head = hdu_lst[0].header
        hdu_lst.close()

        reobj = re.compile('GAMSE BIAS[\s\S]*')
        # filter header cards that match the above pattern
        bias_card_lst = [(card.keyword, card.value) for card in head.cards
                            if reobj.match(card.keyword)]

        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
    else:


        bias_data_lst = []
        bias_card_lst = []

        filterfunc = lambda item: item['object'] is not np.ma.masked and \
                                  item['object'].lower()=='bias'
        bias_items = list(filter(filterfunc, logtable))
        # get the number of bias images
        n_bias = len(bias_items)

        if n_bias == 0:
            # there is no bias frames
            bias = None
        else:
            fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^19s}'
            head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                         'obsdate')

            for iframe, logitem in enumerate(bias_items):
                frameid  = logitem['frameid']
                fileid   = logitem['fileid']
                _objname = logitem['object']
                objectname = '' if _objname is np.ma.masked else _objname
                exptime  = logitem['exptime']
                obsdate  = logitem['obsdate']

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
                            fileid, objectname, exptime, obsdate)
                print(message)

            prefix = 'HIERARCH GAMSE BIAS '
            bias_card_lst.append((prefix + 'NFILE', n_bias))

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            combine_mode = 'mean'
            section = config['reduce.bias']
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

            hdu_lst.writeto(bias_file, overwrite=True)

            bias = bias_combine

    ################# trace ##################
    '''
    ####### find files used to trace ##########
    fmt_str = ('   - {:>5s} {:^15s} {:<20s} {:>7} {:>8} {:>8}')
    head_str = fmt_str.format('FID', 'fileid', 'object', 'exptime', 'nsat',
                                'q95')
    print(head_str)
    for logitem in logtable:
        objectname = logitem['object']
        _objectname = '' if objectname is np.ma.masked else objectname
        message = fmt_str.format('[{:d}]'.format(logitem['frameid']),
                    logitem['fileid'],
                    _objectname,
                    logitem['exptime'],
                    logitem['nsat'], logitem['q95'])
        print(message)
    prompt = 'Select file for tracing order positions: '
    while(True):
        input_string = input(prompt.format(''))
        try:
            frameid = int(input_string)
            if frameid in logtable['frameid']:
                traceid = frameid
                break
            else:
                continue
        except:
            continue
    mask = logtable['frameid']==traceid
    fileid = logtable[mask][0]['fileid']
    fname = '{}.fits'.format(fileid)
    filename = os.path.join(rawpath, fname)
    flat_data, head = fits.getdata(filename, header=True)
    flat_data = correct_overscan(flat_data, head)
    core = np.hanning(31)
    core /= core.sum()
    for col in np.arange(flat_data.shape[1]):
        flat_data[:,col] = np.convolve(flat_data[:,col], core, mode='same')
    flat_mask = np.zeros_like(flat_data, dtype=np.int16)
    '''

    # fiter flat frames
    filterfunc = lambda item: item['object'] is not np.ma.masked and \
                        item['object'].lower()=='flat'
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
        sat_mask = allmask > 0
        flat_mask = np.int16(sat_mask)*4 + np.int16(bad_mask)*2


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
    title = 'Order tracing using {}'.format(fname)
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

    # pack results and save to fits
    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(flat_data, flat_head),
                fits.ImageHDU(flat_mask),
                ])
    hdu_lst.writeto(flat_filename, overwrite=True)

    ###############################
    # get the data shape
    ny, nx = flat_data.shape

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


    ################ Extract ThAr #################
    calib_lst = {}

    # filter the ThAr frames
    filter_thar = lambda item: item['object'] is not np.ma.masked and \
                        item['object'].lower()=='thar'
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

        fmt_str = 'FileID: {} ({}) OBJECT: {} - wavelength identification'
        message = fmt_str.format(fileid, imgtype, objname)
        logger.info(message)
        print(message)

        fname = '{}.fits'.format(fileid)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data = correct_overscan(data, head)

        message = 'Overscan corrected.'
        print(screen_prefix + message)

        # correct bias for flat, if has bias
        if bias is None:
            message = 'No bias. skipped bias correction'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract ThAr spectra
        section = config['reduce.extract']
        lower_limit = section.getfloat('lower_limit')
        upper_limit = section.getfloat('upper_limit')

        spectra1d = extract_aperset(data, mask,
                    apertureset = aperset,
                    lower_limit = lower_limit,
                    upper_limit = upper_limit,
                    )
        message = '1D spectra extracted for {:d} orders'.format(len(spectra1d))
        print(screen_prefix + message)

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

        title = 'Wavelength Identification for {}'.format(fname)

        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                index_file = os.path.join(os.path.dirname(__file__),
                                '../../data/calib/wlcalib_lhrs.dat')

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
                    exit()
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
                exit()

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
                fwhm_range=(6, 14), fwhmrv_range=(8, 12))
        title = 'FWHM Map for {}'.format(fileid)
        fwhm_fig.suptitle(title, fontsize=13)
        figname = 'fwhm_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        fwhm_fig.savefig(figfilename)
        fwhm_fig.close()

        # plot resolution map
        reso_fig = ResolutionFigure(spec, identlist, aperset,
                resolution_range=(2.5e4, 4e4))
        title = 'Resolution Map for {}'.format(fileid)
        reso_fig.suptitle(title, fontsize=13)
        figname = 'resolution_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        reso_fig.savefig(figfilename)
        reso_fig.close()

        # pack to calib_lst
        calib_lst[frameid] = calib

        '''
        if ithar == 0:
            # this is the first ThAr frame in this observing run
            if False:
                pass
            else:
                message = 'No database searching.'
                print(screen_prefix + message)

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
            pass

        # reference the ThAr spectra
        spec, card_lst, identlist = reference_self_wavelength(spec, calib)

        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    fits.BinTableHDU(identlist),
                    ])
        fname = 'wlcalib.{}.fits'.format(fileid)
        filename = os.path.join(midpath, fname)
        hdu_lst.writeto(filename, overwrite=True)
        '''

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

    ########### Extract Flat Spectrum ###########
    extract_flat = True
    if extract_flat:

        # prepar message prefix
        logger_prefix = 'Flat - '
        screen_prefix = '    - '

        message = 'Flat fielding'
        logger.info(message)
        print(message)

        # get background light for flat field
        ny, nx = flat_data.shape
        allx = np.arange(nx)
        background = get_interorder_background(flat_data, flat_mask, aperset)
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
        fig_bkg = BackgroundFigure(flat_data, background,
                    title   = 'Background Correction for {}'.format('flat'),
                    figname = figfilename,
                    )
        fig_bkg.close()

        # remove background light
        flat_data = flat_data - background
        message = 'Background corrected. Max = {:.2f}; Mean = {:.2f}'.format(
                    background.max(), background.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        section = config['reduce.extract']
        lower_limit = section.getfloat('lower_limit')
        upper_limit = section.getfloat('upper_limit')

        # extract 1d spectra of the object
        spectra1d = extract_aperset(flat_data, flat_mask,
                        apertureset = aperset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        norder = len(spectra1d)
        message = '1D spectra of {} orders extracted'.format(norder)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # extract 1d spectra for straylight/background light
        background1d = extract_aperset(background, flat_mask,
                        apertureset = aperset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        message = '1D straylight of {} orders extracted'.format(
                    len(background1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        prefix = 'HIERARCH GAMSE EXTRACTION '
        flat_head.append((prefix + 'LOWER LIMIT', lower_limit))
        flat_head.append((prefix + 'UPPER LIMIT', upper_limit))

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
                        obsdate = flat_head[statime_key],
                        exptime = flat_head[exptime_key],
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
            flat_head.append((prefix + key, value))

        # pack and save wavelength referenced spectra
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=flat_head),
                    fits.BinTableHDU(spec),
                    ])

        # write 1d spectra to fits file
        fname = '{}_{}.fits'.format('flat', oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

    ########### Extract Science Spectrum ##########
    extr_filter = lambda row: row['imgtype']=='sci'
    extr_items = list(filter(extr_filter, logtable))

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

        fmt_str = 'FileID: {} ({}) OBJECT: {} - wavelength identification'
        message = fmt_str.format(fileid, imgtype, objname)
        logger.info(message)
        print(message)

        fname = '{}.fits'.format(fileid)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data = correct_overscan(data, head)

        message = 'Overscan corrected.'
        print(screen_prefix + message)

        # correct bias for flat, if has bias
        if bias is None:
            message = 'No bias. skipped bias correction'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)


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
            pass
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

        # write 1d spectra to fits file
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
