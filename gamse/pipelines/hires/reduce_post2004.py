import os
import re
import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.interpolate as intp
from scipy.signal import savgol_filter
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.background import simple_debackground
from ...echelle.extract import extract_aperset
from ...echelle.flat import get_slit_flat
from ...utils.obslog import parse_num_seq
from ...utils.onedarray import iterative_savgol_filter
from .common import (print_wrapper, parse_3ccd_images, mosaic_3_images,
                    BiasFigure, TraceFigure)

def reduce_post2004(config, logtable):
    """
    """

    # extract keywords from config file
    section = config['data']
    rawpath     = section.get('rawpath')

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

    nccd = 3

    ########################## load file selection #############################
    sel_lst = {}
    filesel_filename = 'file_selection.txt'
    if os.path.exists(filesel_filename):
        sel_file = open(filesel_filename)
        for row in sel_file:
            row = row.strip()
            if len(row)==0 or row[0] in '#':
                continue
            g = row.split(':')
            key, value = g[0].strip(), g[1].strip()
            if len(value)>0:
                sel_lst[key] = value
        sel_file.close()

    ################################ parse bias ################################

    #bias_lst, bias_card_lst = get_bias_post2004(config, logtable)

    section = config['reduce.bias']
    bias_file = section['bias_file']

    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        # pack bias image
        bias_lst = [hdu_lst[iccd+1+nccd].data for iccd in range(nccd)]
        hdu_lst.close()

        reobj = re.compile('GAMSE BIAS[\s\S]*')
        # filter header cards that match the above pattern
        bias_card_lst = [(card.keyword, card.value) for card in head.cards
                            if reobj.match(card.keyword)]

        message = 'Load bias data from file: "{}"'.format(bias_file)
        logger.info(message)
        print(message)

    else:

        bias_data_lst = [[] for iccd in range(nccd)]
        bias_card_lst = []
        bias_items = list(filter(lambda item: item['object'].lower()=='bias',
                                 logtable))
        n_bias = len(bias_items)

        fmt_str = '  - {:>7s} {:^17} {:^20s} {:^7}'
        head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime')

        print('* Combine Bias Images: {}'.format(bias_file))
        print(head_str)

        for logitem in bias_items:
    
            fname = '{}.fits'.format(logitem['fileid'])
            filename = os.path.join(rawpath, fname)
            hdu_lst = fits.open(filename)
            data_lst, mask_lst = parse_3ccd_images(hdu_lst)
            hdu_lst.close()
    
            for iccd in range(nccd):
                bias_data_lst[iccd].append(data_lst[iccd])

            # print info
            message = fmt_str.format(
                        '[{:d}]'.format(logitem['frameid']),
                        logitem['fileid'], logitem['object'],
                        logitem['exptime']
                        )
            print(message)

        prefix = 'HIERARCH GAMSE BIAS '
        bias_card_lst.append((prefix + 'NFILE', n_bias))
    
        combine_mode = 'mean'
        section = config['reduce.bias']
        cosmic_clip  = section.getfloat('cosmic_clip')
        maxiter      = section.getint('maxiter')
        maskmode    = (None, 'max')[n_bias>=3]
    
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
        head['HIERARCH GAMSE FILECONTENT 0'] = 'NONE'

        # pack the primary HDU
        hdu_lst.append(fits.PrimaryHDU(header=head))

        # prepare the list for image data of each CCD
        bias_lst = []

        # scan for each ccd
        for iccd in range(nccd):
            ### 3 CCDs loop begins here ###
            bias_data_lst[iccd] = np.array(bias_data_lst[iccd])
    
            sub_bias = combine_images(bias_data_lst[iccd],
                mode        = combine_mode,
                upper_clip  = cosmic_clip,
                maxiter     = maxiter,
                maskmode    = maskmode,
                ncores      = ncores,
                )

            # pack bias of each CCD into sub_bias_lst
            bias_lst.append(sub_bias)

            message = ('\033[{}mCombined bias for CCD {}: '
                        'Mean = {:6.2f}\033[0m'.format(
                        (34, 32, 31)[iccd], iccd+1, sub_bias.mean()))
            print(message)
        
            key = 'HIERARCH GAMSE FILECONTENT {}'.format(iccd+1)
            hdu_lst[0].header[key] = 'BIAS COMBINED FOR CCD{}'.format(iccd+1)
            hdu_lst.append(fits.ImageHDU(data=sub_bias))

        # initialize bias figure
        bias_fig = BiasFigure(data_lst=bias_lst)

        new_bias_lst = []

        # calculate and plot ymean of bias data
        for iccd in range(nccd):
            data = bias_lst[iccd]
            ny, nx = data.shape
            ally = np.arange(ny)
            ymean = data.mean(axis=1)
            ax_ycut = bias_fig.ax_ycut_lst[iccd]
            ax_ycut.plot(ymean, ally, color='C3', lw=0.5, alpha=0.6)

            # use iterative savgol filter to smooth ymean
            ysmo, _, _, _ = iterative_savgol_filter(ymean, winlen=301, order=3,
                                maxiter=10, upper_clip=3, lower_clip=3)
    
            ax_ycut.plot(ysmo, ally, color='r', lw=0.5, alpha=1)
    
            new_bias = np.tile(ysmo, (nx,1)).T
            new_bias_lst.append(new_bias)

        # save and close the figure
        figfilename = os.path.join(figpath, 'bias.png')
        bias_fig.savefig(figfilename)
        plt.close(bias_fig)

        for iccd in range(nccd):
            hdu_lst.append(fits.ImageHDU(data=new_bias_lst[iccd]))
            # update the file content in primary HDU
            key = 'HIERARCH GAMSE FILECONTENT {}'.format(1+nccd+iccd)
            card = (key, 'BIAS USED FOR CCD{}'.format(iccd+1))
            hdu_lst[0].header.append(card)
        bias_lst = new_bias_lst

        # save to FITS
        hdu_lst.writeto(bias_file, overwrite=True)

        message = 'Bias image written to "{}"'.format(bias_file)
        logger.info(message)
        print(message)

    ########################## find flat groups #########################
    flat_file = config['reduce.flat'].get('flat_file')

    flatdata_lst = []
    # a list of 3 combined flat images. [Image1, Image2, Image3]
    # bias has been corrected already. but not rotated yet.
    flatmask_lst = []
    # a list of 3 flat masks

    if mode=='debug' and os.path.exists(flat_file):
        # read flat data from existing file
        hdu_lst = fits.open(flat_file)
        for iccd in range(nccd):
            flatdata_lst.append(hdu_lst[iccd*2+1].data)
            flatmask_lst.append(hdu_lst[iccd*2+2].data)
        flatdata = hdu_lst[nccd*2+1].data.T
        flatmask = hdu_lst[nccd*2+2].data.T
        hdu_lst.close()
        message = 'Loaded flat data from file: {}'.format(flat_file)
        print(message)

        # alias of flat data and mask
        flatdata1 = flatdata_lst[0].T
        flatmask1 = flatmask_lst[0].T
        flatdata2 = flatdata_lst[1].T
        flatmask2 = flatmask_lst[1].T
        flatdata3 = flatdata_lst[2].T
        flatmask3 = flatmask_lst[2].T

    else:
        fmt_str = ('  - {:>5s} {:^17} {:<20s} {:^1s}I2 {:>7}'
                    ' \033[34m{:>8}\033[0m'
                    ' \033[32m{:>8}\033[0m'
                    ' \033[31m{:>8}\033[0m'
                    ' \033[34m{:>8}\033[0m'
                    ' \033[32m{:>8}\033[0m'
                    ' \033[31m{:>8}\033[0m')
        head_str = fmt_str.format('FID', 'fileid', 'object', '', 'exptime',
                    'nsat_1','nsat_2', 'nsat_3', 'q95_1', 'q95_2', 'q95_3')
        print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)
        # print the flat list
        print(head_str)
        for logitem in logtable:
            if len(logitem['object'])>=8 and logitem['object'][0:8]=='flatlamp':
                message = fmt_str.format(
                        '[{:d}]'.format(logitem['frameid']),
                        logitem['fileid'], logitem['object'], logitem['i2'],
                        logitem['exptime'],
                        logitem['nsat_1'], logitem['nsat_2'], logitem['nsat_3'],
                        logitem['q95_1'], logitem['q95_2'], logitem['q95_3'],
                        )
                print(print_wrapper(message, logitem))

        flat_group_lst = {}
        for iccd in range(nccd):

            key = 'flat CCD%d'%(iccd+1)
            sel_string = sel_lst[key] if key in sel_lst else ''
            prompt = '\033[{1}mSelect flats for CCD {0} [{2}]: \033[0m'.format(
                      iccd+1, (34, 32, 31)[iccd], sel_string)

            # read selected files from terminal
            while(True):
                input_string = input(prompt)
                if len(input_string.strip())==0:
                    # nothing input
                    if key in sel_lst:
                        # nothing input but already in selection list
                        flat_group_lst[iccd] = parse_num_seq(sel_lst[key])
                        break
                    else:
                        # repeat prompt
                        continue
                else:
                    # something input
                    frameid_lst = parse_num_seq(input_string)
                    # pack
                    flat_group_lst[iccd] = frameid_lst
                    # put input string into selection list
                    sel_lst[key] = input_string.strip()
                    break

        # now combine flat images

        flat_hdu_lst = [fits.PrimaryHDU()]
        # flat_hdu_lst is the final HDU list to be saved as fits

        for iccd in range(nccd):
            frameid_lst = flat_group_lst[iccd]

            # now combine flats for this CCD
            flat_data_lst = []
            # flat_data_lst is a list of flat images to be combined.
            # flat_data_lst = [Image1, Image2, Image3, Image4, ... ...]

            #scan the logtable
            # log loop inside the CCD loop because flats for different CCDs are
            # in different files
            for logitem in logtable:
                if logitem['frameid'] in frameid_lst:
                    filename = os.path.join(rawpath, logitem['fileid']+'.fits')
                    hdu_lst = fits.open(filename)
                    data_lst, mask_lst = parse_3ccd_images(hdu_lst)
                    hdu_lst.close()

                    # correct bias and pack into flat_data_lst
                    if bias_lst is None:
                        flat_data_lst.append(data_lst[iccd])
                        message = 'No bias. skipped bias correction'
                    else:
                        flat_data_lst.append(data_lst[iccd]-bias_lst[iccd])
                        message = 'Bias corrected'
                    logger.info(message)

                    # print info
                    # initialize flat mask
                    if len(flat_data_lst) == 1:
                        flatmask = mask_lst[iccd]
                    flatmask = flatmask | mask_lst[iccd]

            n_flat = len(flat_data_lst)

            if n_flat == 0:
                continue
            elif n_flat == 1:
                flatdata = flat_data_lst[0]
            else:
                flat_data_lst = np.array(flat_data_lst)
                flatdata = combine_images(flat_data_lst,
                            mode       = 'mean',
                            upper_clip = 10,
                            maxiter    = 5,
                            maskmode   = (None, 'max')[n_flat>=3],
                            ncores     = ncores,
                            )
                #print('\033[{1}mCombined flat data for CCD {0}: \033[0m'.format(
                #    iccd+1, (34, 32, 31)[iccd]))
            flatdata_lst.append(flatdata)
            flatmask_lst.append(flatmask)

            # pack the combined flat data into flat_hdu_lst
            head = fits.Header()
            head['HIERARCH GAMSE FLAT CCD{} NFILE'.format(iccd+1)] = n_flat
            flat_hdu_lst.append(fits.ImageHDU(flatdata, head))
            flat_hdu_lst.append(fits.ImageHDU(flatmask))
        # CCD loop ends here

        # alias of flat data and mask
        flatdata1 = flatdata_lst[0].T
        flatmask1 = flatmask_lst[0].T
        flatdata2 = flatdata_lst[1].T
        flatmask2 = flatmask_lst[1].T
        flatdata3 = flatdata_lst[2].T
        flatmask3 = flatmask_lst[2].T

        # mosaic flat data
        flatdata, flatmask = mosaic_3_images(
                                data_lst = (flatdata1, flatdata2, flatdata3),
                                mask_lst = (flatmask1, flatmask2, flatmask3),
                                )

        flat_hdu_lst.append(fits.ImageHDU(flatdata.T))
        flat_hdu_lst.append(fits.ImageHDU(flatmask.T))
        # write flat data to file
        flat_hdu_lst = fits.HDUList(flat_hdu_lst)
        flat_hdu_lst.writeto(flat_file, overwrite=True)
        print('Flat data written to {}'.format(flat_file))

    ######################### find & trace orders ##########################

    # simple debackground for all 3 CCDs
    xnodes = np.arange(0, flatdata1.shape[1], 200)
    flatdbkg1 = simple_debackground(flatdata1, flatmask1, xnodes, smooth=20,
                    deg=3, maxiter=10)

    xnodes = np.arange(0, flatdata2.shape[1], 200)
    flatdbkg2 = simple_debackground(flatdata2, flatmask2, xnodes, smooth=20,
                    deg=3, maxiter=10)

    xnodes = np.arange(0, flatdata3.shape[1], 200)
    flatdbkg3 = simple_debackground(flatdata3, flatmask3, xnodes, smooth=20,
                    deg=3, maxiter=10)

    allimage, allmask = mosaic_3_images(
                        data_lst = (flatdbkg1, flatdbkg2, flatdbkg3),
                        mask_lst = (flatmask1, flatmask2, flatmask3),
                        )

    tracefig = TraceFigure()

    section = config['reduce.trace']
    aperset = find_apertures(allimage, allmask,
                scan_step  = section.getint('scan_step'),
                minimum    = section.getfloat('minimum'),
                separation = section.get('separation'),
                align_deg  = section.getint('align_deg'),
                filling    = section.getfloat('filling'),
                degree     = section.getint('degree'),
                display    = section.getboolean('display'),
                fig        = tracefig,
                )
    # decorate trace fig and save to file
    tracefig.adjust_positions()
    tracefig.suptitle('Trace for all 3 CCDs', fontsize=15)
    figfile = os.path.join(figpath, 'trace.png')
    tracefig.savefig(figfile)

    trcfile = os.path.join(midpath, 'trace.trc')
    aperset.save_txt(trcfile)

    regfile = os.path.join(midpath, 'trace.reg')
    aperset.save_reg(regfile, transpose=True)

    # save mosaiced flat image
    trace_hdu_lst = fits.HDUList(
                        [fits.PrimaryHDU(allimage.T),
                         fits.ImageHDU(allmask.T),
                        ])
    trace_hdu_lst.writeto(config['reduce.trace'].get('file'), overwrite=True)

    ######################### Extract flat spectrum ############################

    spectra1d = extract_aperset(flatdata, flatmask,
                    apertureset = aperset,
                    lower_limit = 6,
                    upper_limit = 6,
                    )

    flatmap = get_slit_flat(flatdata, flatmask,
                apertureset = aperset,
                spectra1d   = spectra1d,
                lower_limit = 6,
                upper_limit = 6,
                deg         = 7,
                q_threshold = 20**2,
                figfile     = 'spec_%02d.png',
                )
    fits.writeto('flat_resp.fits', flatmap, overwrite=True)
