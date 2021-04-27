import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ...echelle.trace import load_aperture_set
from ...echelle.imageproc import combine_images
from .common import correct_overscan
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

    flat_sens, flat_spec = get_flat(flat_data, aperset)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(flat_data),
                fits.ImageHDU(flat_sens),
                fits.BinTableHDU(flat_spec),
            ])
    hdu_lst.writeto(flat_file, overwrite=True)

    ############################ Extract ThAr ###########################
