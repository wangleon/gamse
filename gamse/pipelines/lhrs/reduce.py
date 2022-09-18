import re
import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ...echelle.trace import load_aperture_set
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, reference_self_wavelength,
                                reference_spec_wavelength)
from ..common import load_obslog, load_config
from .common import (print_wrapper, correct_overscan, get_mask,
                    TraceFigure, AlignFigure)
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
    statime_key = 'DATE-OBS'
    exptime_key = 'EXPOSURE'

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
                fname = '{}.fit'.format(fileid)
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
    fname = '{}.fit'.format(fileid)
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

        print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
        fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^23s} {:^8} {:^6}'
        head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                    'obsdate', 'N(sat)', 'Q95')

        for iframe, logitem in enumerate(logitem_lst):
            # read each individual flat frame
            fname = '{}.fit'.format(logitem['fileid'])
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
                fits.PrimaryHDU(flat_data, head),
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

    filter_thar = lambda item: item['object'] is not np.ma.masked and \
                        item['object'].lower()=='thar'
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

        fname = '{}.fit'.format(fileid)
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
            message = 'Bias corrected'
        print(screen_prefix + message)

        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
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

        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(spec),
                    #fits.BinTableHDU(identlist),
                    ])
        fname = 'wlcalib_{}.fit'.format(fileid)
        filename = os.path.join(midpath, fname)
        hdu_lst.writeto(filename, overwrite=True)

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
        fname = 'wlcalib.{}.fit'.format(fileid)
        filename = os.path.join(midpath, fname)
        hdu_lst.writeto(filename, overwrite=True)
        '''

