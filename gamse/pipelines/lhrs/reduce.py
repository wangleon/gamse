import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, reference_self_wavelength,
                                reference_spec_wavelength)
from ..common import load_obslog, load_config
from .common import print_wrapper, correct_overscan, get_mask, TraceFigure

def reduce_rawdata():
    """Reduc the LAMOST-HRS data.
    """

    # read obslog and config
    config = load_config('LHRS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

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

    ############# trace ################
    fmt_str = ('   - {:>5s} {:^15s} {:<20s} {:>7} {:>8} {:>8}')
    head_str = fmt_str.format('FID', 'fileid', 'object', 'exptime', 'nsat',
                                'q95')
    print(head_str)
    for logitem in logtable:
        message = fmt_str.format('[{:d}]'.format(logitem['frameid']),
                    logitem['fileid'], logitem['object'], logitem['exptime'],
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
    section = config['reduce.trace']

    tracefig = TraceFigure(datashape=flat_data.shape)

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

    title = 'Order tracing using {}'.format(fname)
    tracefig.suptitle(title)
    tracefig.adjust_positions()
    figname = 'trace_{}.png'.format(traceid)
    tracefig.savefig(figname)
    tracefig.close()

    aperset.save_txt('trace_{}.trc'.format(traceid))
    aperset.save_reg('trace_{}.reg'.format(traceid))

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
            ('mask',        (np.int32, nx)),
            ]
    names, formats = list(zip(*types))
    wlcalib_spectype = np.dtype({'names': names, 'formats': formats})

    filter_thar = lambda item: item['fileid'][0:4]=='thar'
    thar_items = list(filter(filter_thar, logtable))

    for ithar, logitem in enumerate(thar_items):
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fmt_str = 'FileID: {} ({}) OBJECT: - wavelength identification'
        message = fmt_str.format(fileid, imgtype)
        print(message)

        fname = '{}.fit'.format(fileid)
        filename = os.path.join(rawpath, fname)
        data, head = fits.getdata(filename, header=True)
        data = correct_overscan(data, head)
        mask = np.zeros_like(data, dtype=np.int16)

        message = 'Overscan corrected.'
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
                    np.zeros(n, dtype=np.int16),    # mask
                    )
            spec.append(row)
        spec = np.array(spec, dtype=wlcalib_spectype)

        figname = 'wlcalib_{}.{}'.format(fileid, fig_format)
        wlcalib_fig = os.path.join(figpath, figname)
        section = config['reduce.wlcalib']
        title = 'Wavelength Identification for {}'.format(fname)

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

