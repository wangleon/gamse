import os
import datetime
import dateutil.parser
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import gaussian_filter

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ..common import load_obslog, load_config
from .common import (print_wrapper, correct_overscan,
        TraceFigure, AlignFigure)
from ...echelle.trace import find_apertures

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
        pass
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
            fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^19s} {:5s}'
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
