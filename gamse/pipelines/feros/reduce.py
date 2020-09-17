import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from .common import (get_bias, get_mask, correct_overscan,
                    fix_badpixels,
                    TraceFigure, BackgroundFigure,
                    )

def reduce_feros(config, logtable):
    """Reduce the single fiber data of FEROS.

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

    ################ parse bias ########################
    bias, bias_card_lst = get_bias(config, logtable)

    ############### find flat groups ##################

    # initialize flat_groups for single fiber
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}

    for logitem in logtable:
        if logitem['object']=='FLAT' and logitem['binning']=='(1, 1)':

            # find a proper name for this flat
            flatname = '{:g}'.format(logitem['exptime'])

            # add flatname to flat_groups
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem)

    ################# Combine the flats and trace the orders ###################


    # first combine the flats
    for flatname, logitem_lst in flat_groups.items():
        nflat = len(logitem_lst)       # number of flat fieldings

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
            pass
        else:
            # if the above conditions are not satisfied, comine each flat
            data_lst = []
            head_lst = []
            exptime_lst = []

            print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
            fmt_str = '  - {:>7s} {:^23} {:^8s} {:^7} {:^8} {:^6}'
            head_str = fmt_str.format('frameid', 'FileID', 'Object', 'exptime',
                        'N(sat)', 'Q95')

            for iframe, logitem in enumerate(logitem_lst):
                # read each individual flat frame
                fname = 'FEROS.{}.fits'.format(logitem['fileid'])
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
                data, card_lst = correct_overscan(data, head)
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
                            logitem['exptime'],
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
            fig = plt.figure(dpi=300)
            ax = fig.gca()
            ax.plot(flat_data[2166, 0:400],lw=0.5, color='C0')
            # fix badpixels in flat
            flat_data = fix_badpixels(flat_data, bad_mask)
            ax.plot(flat_data[2166, 0:400],lw=0.5, color='C1')
            plt.show()

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
                        transpose  = True,
                        scan_step  = section.getint('scan_step'),
                        minimum    = section.getfloat('minimum'),
                        separation = section.get('separation'),
                        align_deg  = section.getint('align_deg'),
                        filling    = section.getfloat('filling'),
                        degree     = section.getint('degree'),
                        conv_core  = 20,
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

            #section = config['reduce.flat']




            # pack results and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data, head),
                        fits.ImageHDU(flat_mask),
                        fits.ImageHDU(flat_norm),
                        #fits.ImageHDU(flat_sens),
                        #fits.BinTableHDU(flat_spec),
                        ])
            hdu_lst.writeto(flat_filename, overwrite=True)

            # now flt_data and mask_array are prepared
