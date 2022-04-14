import os
import re
import math
import shutil
import struct
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time
import scipy.interpolate as intp
from scipy.ndimage.filters import median_filter

from ...echelle.imageproc import combine_images, savitzky_golay_2d
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                             mosaic_spec,
                             save_crossprofile, read_crossprofile,
                             )
from ...echelle.extract import (extract_aperset, extract_aperset_optimal,
                            extract_aperset_optimal_multifiber)
from ...echelle.wlcalib import (wlcalib, recalib,
                                get_calib_weight_lst, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_pixel_wavelength,
                                reference_self_wavelength,
                                select_calib_auto, select_calib_manu,
                                combine_fiber_cards,
                                combine_fiber_spec,
                                combine_fiber_identlist,
                                )
from ...echelle.background import (find_background, simple_debackground,
                                   get_interorder_background, get_xdisp_profile,
                                   BackgroundLight,
                                   find_best_background,
                                   select_background_from_database,
                                   )
from ...utils.obslog import parse_num_seq
from .common import (print_wrapper, get_mask, get_bias, correct_overscan,
                     select_calib_from_database,
                     TraceFigure, BackgroundFigure, BrightnessProfileFigure,
                     SpatialProfileFigure,
                     )
from .flat import get_flat

def get_fiberobj_lst(string, delimiter='|'):
    """Split the object names for multiple fibers.

    Args:
        string (str): Input object string.
        delimiter (str): Delimiter of different fibers.
    Returns:
        list: a list consist of (ifiber, objname), where **ifiber** is an
            integer, and **objname** is a string.
    """
    object_lst = [s.strip() for s in string.split(delimiter)]
    fiberobj_lst = list(filter(lambda v: len(v[1])>0,
                                enumerate(object_lst)))
    return fiberobj_lst

def get_fiberobj_string(fiberobj_lst, nfiber):
    result_lst = []
    for ifiber in range(nfiber):
        fiber = chr(ifiber+65)
        fibercode = '[{}]'.format(fiber)

        found = False
        for fiberobj in fiberobj_lst:
            if fiberobj[0]==ifiber:
                string = '{:s}  {:s}'.format(fibercode, fiberobj[1])
                result_lst.append(string)
                found = True
                break
        if not found:
            string = '{:s}  ---- '.format(fibercode)
            result_lst.append(string)
    return ' '.join(result_lst)


def reduce_doublefiber(config, logtable):
    """Data reduction for multiple-fiber configuration.
    
    Args:
        config (:class:`configparser.ConfigParser`): The configuration of
            reduction.
        logtable (:class:`astropy.table.Table`): The observing log.
    """

    # extract keywords from config file
    section = config['data']
    rawpath     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')
    # if mulit-fiber, get fiber offset list from config file
    fiber_offsets = [float(v) for v in section.get('fiberoffset').split(',')]
    section = config['reduce']
    midpath = section.get('midpath', None)
    if midpath is None:
        midpath = section.get('midproc')    # old style
    odspath     = section.get('odspath', None)
    if odspath is None:
        odspath = section.get('onedspec')   # old style
    figpath     = section.get('figpath', None)
    if figpath is None:
        figpath = section.get('report')     # old style
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

    n_fiber = 2

    ################################ parse bias ################################
    result = get_bias(config, logtable)
    bias, bias_card_lst, n_bias, bias_overstd, ron_bias = result

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for multi-fibers
    flat_groups = {chr(ifiber+65): {} for ifiber in range(n_fiber)}
    # flat_groups = {'A':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}
    #                'B':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}}

    for logitem in logtable:
        fiberobj_lst = get_fiberobj_lst(logitem['object'])

        if len(fiberobj_lst) != 1:
            continue

        ifiber, objname = fiberobj_lst[0]
        fiber = chr(ifiber+65)
        objname = objname.lower()

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
            if flatname not in flat_groups[fiber]:
                flat_groups[fiber][flatname] = []
            flat_groups[fiber][flatname].append(logitem)

    '''
    # print the flat_groups
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        print(fiber)
        for flatname, item_lst in flat_groups[fiber].items():
            print(flatname)
            for item in item_lst:
                print(fiber, flatname, item['fileid'], item['exptime'])
    '''

    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_mask_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_norm_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_dsum_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_sens_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_corr_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_spec_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_oned_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_raw_lst  = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_info_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_data_bkg = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_bkg_lst  = {fiber: {} for fiber in sorted(flat_groups.keys())}
    aperset_lst   = {fiber: {} for fiber in sorted(flat_groups.keys())}

    ######### define the datatype of flat 1d spectra ########
    if bias is None:
        ndisp = 2048
    else:
        ncros, ndisp = bias.shape

    flatspectype = np.dtype(
                {'names':   ['aperture', 'flux'],
                 'formats': [np.int32, (np.float32, ndisp)],
                 })


    # first combine the flats

    p1, p2, pstep = -10, 10, 0.1
    profile_x = np.arange(p1, p2+1e-4, pstep)
    disp_x_lst = np.arange(32, ndisp, 400)

    all_profile_lst = {}
    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        all_profile_lst[fiber] = {}
        for flatname, item_lst in sorted(fiber_flat_lst.items()):
            # number of flat fieldings
            nflat = len(item_lst)

            flat_filename = os.path.join(midpath,
                    'flat_{}_{}.fits'.format(fiber, flatname))
            aperset_filename = os.path.join(midpath,
                    'trace_flat_{}_{}.trc'.format(fiber, flatname))
            aperset_regname = os.path.join(midpath,
                    'trace_flat_{}_{}.reg'.format(fiber, flatname))
            trace_figname = os.path.join(figpath,
                    'trace_flat_{}_{}.{}'.format(fiber, flatname, fig_format))
            profile_filename = os.path.join(midpath,
                    'profile_flat_{}_{}.fits'.format(fiber, flatname))

            # prepare image arrays and tables either in debug or normal mode
            if mode=='debug' and os.path.exists(flat_filename) \
                and os.path.exists(aperset_filename):
                # in debug mode and all required files exist
                # read flat data and mask array
                hdu_lst = fits.open(flat_filename)
                flat_data = hdu_lst[0].data
                flat_mask = hdu_lst[1].data
                flat_norm = hdu_lst[2].data
                flat_dsum = hdu_lst[3].data
                flat_sens = hdu_lst[4].data
                flat_corr = hdu_lst[5].data
                flat_spec = hdu_lst[6].data
                exptime   = hdu_lst[0].header[exptime_key]
                hdu_lst.close()
                aperset = load_aperture_set(aperset_filename)
                _, _, profile_lst = read_crossprofile(profile_filename)
                all_profile_lst[fiber][flatname] = profile_lst
            else:
                # in normal mode, or any required file does not exist

                # combine each flat
                data_lst = []
                head_lst = []
                exptime_lst = []

                print('* Combine {} Flat Images: {}'.format(
                        nflat, flat_filename))

                fmt_str = '    - {:>5s} {:18s} {:20s} {:7} {:23s} {:6} {:5}'
                head_str= fmt_str.format('ID', 'fileid', 'object', 'exptime',
                            'obsdate', 'nsat', 'q95')
                print(head_str)

                for i_item, logitem in enumerate(item_lst):
                    # read each individual flat frame
                    fname = '{}.fits'.format(logitem['fileid'])
                    filename = os.path.join(rawpath, fname)
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
                    fiberobj_lst = get_fiberobj_lst(logitem['object'], '|')
                    fiberobj_str = get_fiberobj_string(fiberobj_lst, n_fiber)
                    string = fmt_str.format(
                            '[{:d}]'.format(logitem['frameid']),
                            logitem['fileid'], fiberobj_str,
                            '{:<7g}'.format(logitem['exptime']),
                            str(logitem['obsdate']),
                            '{:<6d}'.format(logitem['nsat']),
                            '{:<5d}'.format(logitem['q95']),
                            )
                    print(print_wrapper(string, logitem))

                    data_lst.append(data)

                if nflat == 1:
                    flat_data = data_lst[0]
                    flat_dsum = data_lst[0]
                else:
                    data_lst = np.array(data_lst)
                    flat_data = combine_images(data_lst,
                                    mode        = 'mean',
                                    upper_clip  = 10,
                                    maxiter     = 5,
                                    maskmode    = (None, 'max')[nflat>3],
                                    ncores      = ncores,
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
                #flat_dbkg = simple_debackground(flat_data, flat_mask, xnodes,
                # smooth=5)
                #aperset = find_apertures(flat_dbkg, flat_mask,
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
                color = {'A':'green','B':'yellow'}[fiber]
                aperset.save_reg(aperset_regname, fiber=fiber, color=color)

                # do flat fielding
                # prepare the output mid-process figures in debug mode
                if mode=='debug':
                    figname = 'flat_aperpar_{}_{}_%03d.{}'.format(
                                fiber, flatname, fig_format)
                    fig_aperpar = os.path.join(figpath, figname)
                else:
                    fig_aperpar = None

                # prepare the name for slit figure
                figname = 'slit_flat_{}_{}.{}'.format(
                            fiber, flatname, fig_format)
                fig_slit = os.path.join(figpath, figname)

                # prepare the name for slit file
                fname = 'slit_flat_{}_{}.dat'.format(fiber, flatname)
                slit_file = os.path.join(midpath, fname)

                section = config['reduce.flat']

                '''
                flat_sens, flat_spec = get_flat(
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
                '''
                fig_spatial = SpatialProfileFigure()
                flat_sens, flatspec_lst, profile_lst = get_flat(
                        data        = flat_data,
                        mask        = flat_mask,
                        apertureset = aperset,
                        nflat       = nflat,
                        q_threshold = section.getfloat('q_threshold'),
                        mode        = 'debug',
                        fig_spatial = fig_spatial,
                        flatname    = '{}_{}'.format(fiber, flatname),
                        profile_x   = profile_x,
                        disp_x_lst  = disp_x_lst,
                        )
                #newaperset_regname = os.path.join(midpath,
                #    'trace_flat_{}_{}_new.reg'.format(fiber, flatname))
                #newaperset.save_reg(newaperset_regname,
                #            fiber=fiber, color=color)

                figname = os.path.join(figpath,
                            'spatial_profile_flat_{}_{}.png'.format(
                            fiber, flatname))
                title = 'Spatial Profile of flat_{}_{}'.format(
                            fiber, flatname)
                fig_spatial.suptitle(title)
                fig_spatial.savefig(figname)
                fig_spatial.close()

                # pack 1-d spectra of flat
                flat_spec = [(aper, flatspec) for aper, flatspec
                                in sorted(flatspec_lst.items())]
                flat_spec = np.array(flat_spec, dtype=flatspectype)

                # save cross-profiles
                all_profile_lst[fiber][flatname] = profile_lst
                save_crossprofile(profile_filename, disp_x_lst,
                                    p1, p2, pstep, profile_lst)

                flat_corr = flat_data/flat_sens

                
                head.append(('HIERARCH GAMSE FILECONTENT 0', 'FLAT COMBINED'))
                head.append(('HIERARCH GAMSE FILECONTENT 1', 'FLAT MASK'))
                head.append(('HIERARCH GAMSE FILECONTENT 2', 'FLAT NORM'))
                head.append(('HIERARCH GAMSE FILECONTENT 3', 'FLAT DSUM'))
                head.append(('HIERARCH GAMSE FILECONTENT 4', 'FLAT SENSITIVITY'))
                head.append(('HIERARCH GAMSE FILECONTENT 5', 'FLAT CORRECTED'))
                head.append(('HIERARCH GAMSE FILECONTENT 6', 'FLAT ONEDSPEC'))
                # pack results and save to fits
                hdu_lst = fits.HDUList([
                            fits.PrimaryHDU(flat_data, head),
                            fits.ImageHDU(flat_mask),
                            fits.ImageHDU(flat_norm),
                            fits.ImageHDU(flat_dsum),
                            fits.ImageHDU(flat_sens),
                            fits.ImageHDU(flat_corr),
                            fits.BinTableHDU(flat_spec),
                            ])
                hdu_lst.writeto(flat_filename, overwrite=True)

            #### mode switch ends here

            # prepare flat_oned, a table containing self-corrected 1d spectra
            # of flat
            section = config['reduce.extract']
            flat_1d = extract_aperset(flat_corr, flat_mask,
                        apertureset = aperset,
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )
            table = [(aper, row['flux_sum'])
                        for aper, row in sorted(flat_1d.items())]
            flat_oned = np.array(table, dtype=flatspectype)

            # prepare flat_raw1d, a table containing non-corrected 1d spectra
            # of flat
            section = config['reduce.extract']
            flat_raw1d = extract_aperset(flat_data, flat_mask,
                        apertureset = aperset,
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )
            table = [(aper, row['flux_sum'])
                        for aper, row in sorted(flat_raw1d.items())]
            flat_raw1d = np.array(table, dtype=flatspectype)


            '''
            # correct background for flat
            fig_sec = os.path.join(figpath,
                    'bkg_flat_{}_{}_sec.{}'.format(fiber, flatname, fig_format))
            section = config['reduce.background']
            stray = find_background(data, mask,
                    aperturesets = aperset,
                    ncols        = section.getint('ncols'),
                    distance     = section.getfloat('distance'),
                    yorder       = section.getint('yorder'),
                    fig_section  = fig_sec,
                    )
            flat_dbkg = flat_data - stray
            # plot stray light of flat
            fig_stray = os.path.join(figpath,
                        'bkg_flat_{}_{}_stray.{}'.format(
                        fiber, flatname, fig_format))
            plot_background_aspect1(flat_data, stray, fig_stray)
            # extract 1d spectrum of flat
            section = config['reduce.extract']
            spectra1d = extract_aperset(flat_dbkg, mask,
                            apertureset = aperset,
                            lower_limit = section.getfloat('lower_limit'),
                            upper_limit = section.getfloat('upper_limit'),
                        )
            '''

            # append the flat data and mask
            flat_data_lst[fiber][flatname] = flat_data
            flat_mask_lst[fiber][flatname] = flat_mask
            flat_norm_lst[fiber][flatname] = flat_norm
            flat_dsum_lst[fiber][flatname] = flat_dsum
            flat_sens_lst[fiber][flatname] = flat_sens
            flat_corr_lst[fiber][flatname] = flat_corr
            flat_spec_lst[fiber][flatname] = flat_spec
            flat_oned_lst[fiber][flatname] = flat_oned
            flat_raw_lst[fiber][flatname]  = flat_raw1d
            flat_info_lst[fiber][flatname] = {'exptime': exptime}
            flat_bkg_lst[fiber][flatname]  = flat_data
            aperset_lst[fiber][flatname]   = aperset

            # continue to the next colored flat
        # continue to the next fiber

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midpath, 'flat.fits')
    trac_file = os.path.join(midpath, 'trace.trc')
    treg_file = os.path.join(midpath, 'trace.reg')

    # master aperset is a dict of {fiber: aperset}.
    master_aperset = {}

    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        #fibercode = struct.pack('>i', -257981040+ifiber).decode()
        fibercode = '[{}]'.format(fiber)
        fiber_flat_lst = flat_groups[fiber]

        # determine the mosaiced flat filename
        flat_fiber_file = os.path.join(midpath,
                            'flat_{}.fits'.format(fiber))
        trac_fiber_file = os.path.join(midpath,
                            'trace_{}.trc'.format(fiber))
        treg_fiber_file = os.path.join(midpath,
                            'trace_{}.reg'.format(fiber))

        if len(fiber_flat_lst) == 1:
            # there's only ONE "color" of flat
            flatname = list(fiber_flat_lst)[0]

            '''
            # copy the trc file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.trc'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.trc'.format(flatname)
            shutil.copyfile(os.path.join(midpath, oriname), trac_fiber_file)
            # copy the reg file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.reg'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.reg'.format(flatname)
            shutil.copyfile(os.path.join(midpath, oriname), treg_fiber_file)
            '''

            flat_data = flat_data_lst[fiber][flatname]
            flat_mask = flat_mask_lst[fiber][flatname]
            flat_norm = flat_norm_lst[fiber][flatname]
            flat_dsum = flat_dsum_lst[fiber][flatname]
            flat_sens = flat_sens_lst[fiber][flatname]
            flat_corr = flat_corr_lst[fiber][flatname]
            flat_spec = flat_spec_lst[fiber][flatname]
            flat_oned = flat_oned_lst[fiber][flatname]
            flat_raw  = flat_raw_lst[fiber][flatname]
    
            # no need to mosaic aperset
            master_aperset[fiber] = list(aperset_lst[fiber].values())[0]
        else:
            # mosaic apertures
            section = config['reduce.flat']
            # determine the mosaic order
            name_lst = sorted(flat_info_lst[fiber],
                        key=lambda x: flat_info_lst[fiber].get(x)['exptime'])

            master_aperset[fiber] = mosaic_flat_auto(
                    aperture_set_lst = aperset_lst[fiber],
                    max_count        = section.getfloat('mosaic_maxcount'),
                    name_lst         = name_lst,
                    )
            # mosaic original flat images
            flat_data = mosaic_images(flat_data_lst[fiber],
                                        master_aperset[fiber])
            # mosaic flat mask images
            flat_mask = mosaic_images(flat_mask_lst[fiber],
                                        master_aperset[fiber])
            # mosaic exptime-normalized flat images
            flat_norm = mosaic_images(flat_norm_lst[fiber],
                                        master_aperset[fiber])
            # mosaic summed flat images
            flat_dsum = mosaic_images(flat_dsum_lst[fiber],
                                        master_aperset[fiber])
            # mosaic sensitivity map
            flat_sens = mosaic_images(flat_sens_lst[fiber],
                                        master_aperset[fiber])
            # mosaic corrected flat image
            flat_corr = mosaic_images(flat_corr_lst[fiber],
                                        master_aperset[fiber])

            # mosaic 1d spectra of flats
            flat_spec = mosaic_spec(flat_spec_lst[fiber],
                                        master_aperset[fiber])
            flat_oned = mosaic_spec(flat_oned_lst[fiber],
                                        master_aperset[fiber])
            flat_raw = mosaic_spec(flat_raw_lst[fiber],
                                        master_aperset[fiber])

        # change contents of several lists
        flat_data_lst[fiber] = flat_data
        flat_mask_lst[fiber] = flat_mask
        flat_norm_lst[fiber] = flat_norm
        flat_dsum_lst[fiber] = flat_dsum
        flat_sens_lst[fiber] = flat_sens
        flat_corr_lst[fiber] = flat_corr
        flat_spec_lst[fiber] = flat_spec
        flat_oned_lst[fiber] = flat_oned
        flat_raw_lst[fiber] = flat_raw

        # pack and save to fits file
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(flat_data),
                    fits.ImageHDU(flat_mask),
                    fits.ImageHDU(flat_norm),
                    fits.ImageHDU(flat_dsum),
                    fits.ImageHDU(flat_sens),
                    fits.ImageHDU(flat_corr),
                    fits.BinTableHDU(flat_spec),
                    ])
        hdu_lst.writeto(flat_fiber_file, overwrite=True)


        # align different fibers
        if ifiber == 0:
            ref_aperset = master_aperset[fiber]
        else:
            # find the postion offset (yshift) relative to the first fiber ("A")
            # the postion offsets are identified by users in the config file.
            # the first one (index=0) is shift of fiber B. second one is C...
            yshift = fiber_offsets[ifiber-1]
            offset = master_aperset[fiber].find_aper_offset(
                        ref_aperset, yshift=yshift)

            # print and logging
            message = 'Fiber {}, aperture offset = {}'.format(fiber, offset)
            print(message)
            logger.info(message)

            # correct the aperture offset
            master_aperset[fiber].shift_aperture(-offset)

            # also correct the aperture number in flatspec
            flat_spec_lst[fiber]['aperture'] -= offset

            flat_oned_lst[fiber]['aperture'] -= offset
            flat_raw_lst[fiber]['aperture'] -= offset

    # save the mosaic, offset-corrected aperset to txt files
    for fiber, aperset in sorted(master_aperset.items()):
        # save as .trc file
        fname = 'trace_{}.trc'.format(fiber)
        outfilename = os.path.join(midpath, fname)
        aperset.save_txt(outfilename)
        message = '{} Apertures for fiber {} saved to "{}"'.format(
                    len(aperset), fiber, outfilename)
        logger.info(message)
        print(message)

        # save as .reg file
        fname = 'trace_{}.reg'.format(fiber)
        outfilename = os.path.join(midpath, fname)
        color = {'A': 'green', 'B': 'yellow'}[fiber]
        aperset.save_reg(outfilename, fiber=fiber, color=color)

    # find all the aperture list for all fibers
    allmax_aper = -99
    allmin_aper = 999
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        allmax_aper = max(allmax_aper, max(master_aperset[fiber]))
        allmin_aper = min(allmin_aper, min(master_aperset[fiber]))

    # pack all aperloc into a single list
    all_aperloc_lst = []
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        aperset = master_aperset[fiber]
        for aper, aperloc in aperset.items():
            x, y = aperloc.get_position()
            center = aperloc.get_center()
            all_aperloc_lst.append([fiber, aper, aperloc, center])

    # mosaic flat map
    sorted_aperloc_lst = sorted(all_aperloc_lst, key=lambda x:x[3])
    ny, nx = flat_data.shape
    master_flatdata = np.empty((ny, nx))
    master_flatmask = np.empty((ny, nx))
    master_flatnorm = np.empty((ny, nx))
    master_flatdsum = np.empty((ny, nx))
    master_flatsens = np.empty((ny, nx))
    master_flatcorr = np.empty((ny, nx))
    yy, xx = np.mgrid[:ny, :nx]
    prev_line = np.zeros(nx)
    for i in np.arange(len(sorted_aperloc_lst)-1):
        fiber, aper, aperloc, center = sorted_aperloc_lst[i]
        x, y = aperloc.get_position()
        next_fiber, _, next_aperloc, _ = sorted_aperloc_lst[i+1]
        next_x, next_y = next_aperloc.get_position()
        next_line = np.int32(np.round((y + next_y)/2.))
        #print(fiber, aper, center, prev_line, next_line)
        mask = (yy >= prev_line)*(yy < next_line)
        master_flatdata[mask] = flat_data_lst[fiber][mask]
        master_flatmask[mask] = flat_mask_lst[fiber][mask]
        master_flatnorm[mask] = flat_norm_lst[fiber][mask]
        master_flatdsum[mask] = flat_dsum_lst[fiber][mask]
        master_flatsens[mask] = flat_sens_lst[fiber][mask]
        master_flatcorr[mask] = flat_corr_lst[fiber][mask]
        prev_line = next_line
    # parse the last order
    mask = yy >= prev_line
    master_flatdata[mask] = flat_data_lst[next_fiber][mask]
    master_flatmask[mask] = flat_mask_lst[next_fiber][mask]
    master_flatnorm[mask] = flat_norm_lst[next_fiber][mask]
    master_flatdsum[mask] = flat_dsum_lst[next_fiber][mask]
    master_flatsens[mask] = flat_sens_lst[next_fiber][mask]
    master_flatcorr[mask] = flat_corr_lst[next_fiber][mask]

    # pack and save to fits file
    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(master_flatdata),
                fits.ImageHDU(master_flatmask),
                fits.ImageHDU(master_flatnorm),
                fits.ImageHDU(master_flatdsum),
                fits.ImageHDU(master_flatsens),
                fits.ImageHDU(master_flatcorr),
                ])
    hdu_lst.writeto(flat_file, overwrite=True)

    ############## averaeg cross-order profiles  ###############
    profile = {}
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        _profile = np.array([all_profile_lst[fiber][flatname]
                    for flatname in flat_groups[fiber]])
        profile[fiber] = _profile.mean(axis=0)

    ############################## Extract ThAr ################################

    # get the data shape
    ny, nx = flat_sens.shape

    # define dtype of 1-d spectra for wlcalib files
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
    # calib_lst is a hierarchical dict of calibration results
    # calib_lst = {
    #       'A': {'frameid1': calib_dict1, 'frameid2': calib_dict2, ...}
    #       'B': {'frameid1': calib_dict3, 'frameid2': calib_dict4, ...}
    #       ... ...
    #       }

    def filter_thar(logitem):
        fiberobj_lst = get_fiberobj_lst(logitem['object'])
        newlst = list(filter(lambda v: v[1].lower()=='thar', fiberobj_lst))
        return len(newlst) == len(fiberobj_lst)

    # start and end point in pixel and order for the 2d ThAr fit
    pixel_range = (101, 2006)
    order_range = (61, 149)

    # range in which the 2d thar fit is performed 
    def wlfit_filter(item):
        if pixel_range[0] <= item['pixel'] <= pixel_range[1] and \
           order_range[0] <= item['order'] <= order_range[1]:
            return True
        else:
            return False

    thar_items = list(filter(filter_thar, logtable))

    for ithar, logitem in enumerate(thar_items):
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        obj     = logitem['object']
        exptime = logitem['exptime']
        obsdate = logitem['obsdate']
        if isinstance(obsdate, Time):
            obsdate = obsdate.isot

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        fiberobj_lst = get_fiberobj_lst(logitem['object'], '|')
        fiberobj_str = get_fiberobj_string(fiberobj_lst, n_fiber)

        # now all objects in fiberobj_lst must be ThAr

        message = ('FileID: {} ({}) OBJECT: {}'
                    ' - wavelength identification').format(
                    fileid, imgtype, fiberobj_str)
        logger.info(message)
        print(message)

        # read raw data
        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
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

        # initialize data for all fibers
        all_spec      = {}
        all_cards     = {}
        all_identlist = {}

        for ifiber, objname in fiberobj_lst:
            fiber = chr(ifiber+65)
            objname = objname.lower()

            section = config['reduce.extract']
            lower_limit = section.getfloat('lower_limit')
            upper_limit = section.getfloat('upper_limit')
            apertureset = master_aperset[fiber]

            spectra1d = extract_aperset(data, mask,
                        apertureset = apertureset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                        )
            message = 'Fiber {}: 1D spectra of {} orders extracted'.format(
                       fiber, len(spectra1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)
            head = master_aperset[fiber].to_fitsheader(head, fiber=fiber)

            # pack to a structured array
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                n = flux_sum.size

                # pack to table
                item = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux_sum,                       # flux
                        np.zeros(n),                    # mask
                        )
                spec.append(item)
            spec = np.array(spec, dtype=wlcalib_spectype)

            figname = 'wlcalib_{}_{}.{}'.format(fileid, fiber, fig_format)
            wlcalib_fig = os.path.join(figpath, figname)

            section = config['reduce.wlcalib']

            title = '{}.fits - Fiber {}'.format(fileid, fiber)

            if ithar == 0:
                # this is the first ThAr frame in this observing run
                if section.getboolean('search_database'):
                    # find previouse calibration results
                    index_file = os.path.join(os.path.dirname(__file__),
                                    '../../data/calib/wlcalib_foces.dat')

                    message = ('Searching for archive wavelength calibration '
                               'file in "{}"'.format(
                                   os.path.basename(index_file)))
                    logger.info(logger_prefix + message)
                    print(screen_prefix + message)
                    
                    ref_spec, ref_calib = select_calib_from_database(
                            index_file, obsdate)

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
                            fit_filter  = wlfit_filter,
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
                        elif direction[1]==ref_direction[1]:
                            aperture_k = 1
                        else:
                            aperture_k = -1

                        if direction[2] == '?':
                            pixel_k = None
                        elif direction[2]==ref_direction[2]:
                            pixel_k = 1
                        else:
                            pixel_k = -1

                        result = find_caliblamp_offset(ref_spec, spec,
                                    aperture_k  = aperture_k,
                                    pixel_k     = pixel_k,
                                    pixel_range = (-30, 30),
                                    mode        = mode,
                                    )
                        aperture_koffset = (result[0], result[1])
                        pixel_koffset    = (result[2], result[3])

                        #fig = plt.figure()
                        #ax = fig.gca()
                        #m1 = spec['aperture']==10
                        #ax.plot(spec[m1][0]['flux'])
                        #m2 = ref_spec['aperture']==9
                        #ax.plot(ref_spec[m2][0]['flux'])
                        #plt.show()

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
                        fit_filter    = wlfit_filter,
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
                    fit_filter       = wlfit_filter,
                    )

            # add more infos in calib
            calib['fileid']   = fileid
            calib['date-obs'] = head[statime_key]
            calib['exptime']  = head[exptime_key]

            # reference the ThAr spectra
            spec, card_lst, identlist = reference_self_wavelength(spec, calib)

            # add the fit_filter keywords to card_lst
            card_lst.append(('PIXEL_RANGE', str(pixel_range)))
            card_lst.append(('ORDER_RANGE', str(order_range)))

            # append all spec, card list and ident lists
            all_spec[fiber]      = spec
            all_cards[fiber]     = card_lst
            all_identlist[fiber] = identlist

            # save calib results and the oned spec for this fiber
            head_fiber = head.copy()
            prefix = 'HIERARCH GAMSE WLCALIB '
            for key,value in card_lst:
                head_fiber.append((prefix+key, value))

            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head_fiber),
                        fits.BinTableHDU(spec),
                        fits.BinTableHDU(identlist),
                        ])
            fname = 'wlcalib_{}_{}.fits'.format(fileid, fiber)
            filename = os.path.join(midpath, fname)
            hdu_lst.writeto(filename, overwrite=True)

            # pack to calib_lst
            if fiber not in calib_lst:
                calib_lst[fiber] = {}
            calib_lst[fiber][frameid] = calib

        # fiber loop ends here
        # combine different fibers
        # combine cards for FITS header
        newcards = combine_fiber_cards(all_cards)
        # combine spectra
        newspec = combine_fiber_spec(all_spec)
        # combine ident line list
        newidentlist = combine_fiber_identlist(all_identlist)

        # append cards to fits header
        prefix = 'HIERARCH GAMSE WLCALIB '
        for key, value in newcards:
            head.append((prefix+key, value))

        # pack and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(newspec),
                    fits.BinTableHDU(newidentlist),
                    ])
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

    # print fitting summary
    fmt_string = (
        ' [{:3d}] {} - fiber {:1s} ({:4g} sec) - {:4d}/{:4d} RMS = {:7.5f}')
    section = config['reduce.wlcalib']
    auto_selection = section.getboolean('auto_selection')

    if auto_selection:
        rms_threshold    = section.getfloat('rms_threshold', 0.005)
        group_contiguous = section.getboolean('group_contiguous', True)
        time_diff        = section.getfloat('time_diff', 120)

        ref_calib_lst = {}

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if fiber in calib_lst and len(calib_lst[fiber])>0:
                ref_calib_lst[fiber] = select_calib_auto(calib_lst[fiber],
                                            rms_threshold    = rms_threshold,
                                            group_contiguous = group_contiguous,
                                            time_diff        = time_diff,
                                        )
            else:
                # because n_fiber = 2, ifiber = 0, 1.
                # the other fiber is 1-ifiber
                other_fiber = chr((1-ifiber)+65)

                ref_calib_lst[fiber] = select_calib_auto(
                                        calib_lst[other_fiber],
                                        rms_threshold    = rms_threshold,
                                        group_contiguous = group_contiguous,
                                        time_diff        = time_diff,
                                    )

            if len(ref_calib_lst[fiber])==0:
                # if no proper calib found for this fiber.
                # then change another fiber
                other_fiber = chr((1-ifiber)+65)

                ref_calib_lst[fiber] = select_calib_auto(
                                        calib_lst[other_fiber],
                                        rms_threshold    = rms_threshold,
                                        group_contiguous = group_contiguous,
                                        time_diff        = time_diff,
                                    )

            if len(ref_calib_lst[fiber])==0:
                # if still cannnot find a calib for this fiber
                pass

        # print ThAr summary and selected calib
        mix_calib_lst = {}
        for fiber, fiber_calib_lst in sorted(calib_lst.items()):
            for frameid, calib in sorted(fiber_calib_lst.items()):
                mix_calib_lst[(frameid, fiber)] = calib

        ref_fileid_lst = {}
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            ref_fileid_lst[fiber] = [calib['fileid']
                                        for calib in ref_calib_lst[fiber]]

        for key, calib in sorted(mix_calib_lst.items()):
            frameid, fiber = key
            string = fmt_string.format(frameid, calib['fileid'], fiber,
                            calib['exptime'], calib['nuse'], calib['ntot'],
                            calib['std'])
            sel_fibers = []
            for ifiber in range(n_fiber):
                fiber = chr(ifiber+65)
                if calib['fileid'] in ref_fileid_lst[fiber]:
                    sel_fibers.append(fiber)
            if len(sel_fibers)>0:
                string = '\033[91m{} [selected for fiber {}]\033[0m'.format(
                            string, ','.join(sel_fibers))
            print(string)

    else:
        # print the fitting summary
        mix_calib_lst = {}
        for fiber, fiber_calib_lst in sorted(calib_lst.items()):
            for frameid, calib in sorted(fiber_calib_lst.items()):
                mix_calib_lst[(frameid, fiber)] = calib
        for key, calib in sorted(mix_calib_lst.items()):
            frameid, fiber = key
            string = fmt_string.format(frameid, calib['fileid'], fiber,
                            calib['exptime'], calib['nuse'], calib['ntot'],
                            calib['std'])
            print(string)

        ref_calib_lst = {}
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            promotion = 'Select References for fiber {}: '.format(fiber)

            if fiber in calib_lst and len(calib_lst[fiber])>0:
                ref_calib_lst[fiber] = select_calib_manu(calib_lst[fiber],
                                        promotion = promotion,
                                        )
            else:
                other_fiber = chr((1-ifiber)+65)
                ref_calib_lst[fiber] = select_calib_manu(calib_lst[other_fiber],
                                        promotion = promotion,
                                        )

    # define dtype of 1-d spectra for all fibers
    types = [
            ('aperture',     np.int16),
            ('order',        np.int16),
            ('points',       np.int16),
            ('wavelength',   (np.float64, nx)),
            ('flux_sum',     (np.float32, nx)),
            ('flux_sum_err', (np.float32, nx)),
            ('flux_sum_mask',(np.int16,   nx)),
            ('flux_opt',     (np.float32, nx)),
            ('flux_opt_err', (np.float32, nx)),
            ('flux_opt_mask',(np.int16,   nx)),
            ('flux_raw',     (np.float32, nx)),
            ('blaze',        (np.float32, nx)),
            ('flat_sum',     (np.float32, nx)),
            ('flat_raw',     (np.float32, nx)),
            ('flat_bkg',     (np.float32, nx)),
            ('background',   (np.float32, nx)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    extracted_fileid_lst = []

    ################### Extract stray light of flats ###################
    flat_bkg  = {}
    back_flat_1d = {}
    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        for flatname, item_lst in sorted(fiber_flat_lst.items()):
            data = flat_bkg_lst[fiber][flatname]
            background = get_interorder_background(data,
                                apertureset = master_aperset[fiber])
            background = median_filter(background, size=(9,1), mode='nearest')
            background = savitzky_golay_2d(background, window_length=(21, 101),
                            order=3, mode='nearest')
            flat_data_bkg[fiber][flatname] = background

        flat_bkg[fiber] = mosaic_images(flat_data_bkg[fiber],
                                        master_aperset[fiber])
        # extract 1d spectra for stray light of the mosaic flat
        section = config['reduce.extract']
        data = flat_bkg[fiber]
        mask = np.zeros_like(data, dtype=np.bool)
        lower_limit = section.getfloat('lower_limit')
        upper_limit = section.getfloat('upper_limit')
        apertureset = master_aperset[fiber]

        back_flat_1d[fiber] = extract_aperset(data, mask,
                                apertureset = apertureset,
                                lower_limit = lower_limit,
                                upper_limit = upper_limit,
                                )

        message = 'Fiber {}: 1D straylight of {} orders extracted'.format(
            fiber, len(back_flat_1d[fiber]))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

    #################### Extract Spectra with Single Objects ###################

    # first round, find the images with only single objects. extract the
    # spectra, and save the background lights
    saved_bkg_lst = []

    for logitem in logtable:
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objects = logitem['object']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        # filter out the single objects but bias and dark. because they are also
        # appear to be "single" objects
        if objects.strip().lower() in ['bias', 'dark']:
            continue

        # split the object names and make obj_lst
        fiberobj_lst = get_fiberobj_lst(objects, '|')
        fiberobj_str = get_fiberobj_string(fiberobj_lst, n_fiber)

        # filter out images with multi-fibers
        if len(fiberobj_lst) != 1:
            continue
        ifiber, objname = fiberobj_lst[0]
        fiber = chr(ifiber+65)
        #fibercode_cmd = struct.pack('>i', -257981040+ifiber).decode()
        fibercode = '[{}]'.format(fiber)

        # filter out Flat and ThAr
        if objname.lower()[0:4] in ['flat', 'thar']:
            continue

        message = 'FileID: {} ({}) OBJECT: {}'.format(
                    fileid, imgtype, fiberobj_str)
        logger.info(message)
        print(message)

        # read raw data
        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
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

        # 2d image to extract the raw flux
        raw_data = data.copy()

        # creating the variance map to track the errors
        variance_map = np.maximum(data, 0) + overstd**2 + ron_bias**2

        # correct flat
        data = data/master_flatsens
        variance_map = variance_map/master_flatsens**2
        message = 'Flat field corrected'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # including the error of the sens.map corretion
        master_flatdsum_0 = 1.0*master_flatdsum  # bad hack for copying
        # making sure that not affected pixels (pixels with a value of 1) 
        # have no impact on the total error
        master_flatdsum_0[np.where(master_flatsens == 1.0)] = 1e99 
        variance_map = variance_map + data**2/master_flatdsum_0

        # get background lights
        background = get_interorder_background(data,
                        apertureset = master_aperset[fiber])
        background = median_filter(background, size=(9,1), mode='nearest')
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

        # get order brightness profile
        result = get_xdisp_profile(data, master_aperset[fiber])
        aper_num_lst, aper_pos_lst, aper_brt_lst = result

        # calibrate the wavelength of background
        # get weights for calib list
        weight_lst = get_calib_weight_lst(ref_calib_lst[fiber],
                        obsdate = head[statime_key],
                        exptime = head[exptime_key],
                        )
        message_lst = ['Fiber {}: Wavelength calibration:'.format(fibercode)]
        for i, calib in enumerate(ref_calib_lst[fiber]):
            string = ' '*len(screen_prefix)
            string = string + '{} ({:4g} sec) {} weight = {:5.3f}'.format(
                        calib['fileid'], calib['exptime'], calib['date-obs'],
                        weight_lst[i])
            message_lst.append(string)
        message = os.linesep.join(message_lst)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        ny, nx = data.shape
        pixel_lst = np.repeat(nx//2, aper_num_lst.size)
        # reference the wavelengths of background image
        results = reference_pixel_wavelength(pixel_lst, aper_num_lst,
                    ref_calib_lst[fiber], weight_lst)
        aper_ord_lst, aper_wav_lst = results

        if objname.lower() in ['comb', 'fp']:
            objtype = objname.lower()
        else:
            objtype = 'star'

        # pack to background list
        bkg_info = {
                    'fileid': fileid,
                    'fiber': fiber,
                    'object': objname,
                    'objtype': objtype,
                    'exptime': exptime,
                    'date-obs': head[statime_key],
                    }
        bkg_obj = BackgroundLight(
                    info         = bkg_info,
                    header       = head,
                    data         = background,
                    aper_num_lst = aper_num_lst,
                    aper_ord_lst = aper_ord_lst,
                    aper_pos_lst = aper_pos_lst,
                    aper_brt_lst = aper_brt_lst,
                    aper_wav_lst = aper_wav_lst,
                    )
        # save to fits
        outfilename = os.path.join(midpath, 'bkg_{}.fits'.format(fileid))
        bkg_obj.savefits(outfilename)
        # pack to saved_bkg_lst
        saved_bkg_lst.append(bkg_obj)

        # extract 1d spectrum
        section = config['reduce.extract']

        # exract spectra
        all_spec  = {}   # use to pack final 1d spectrum
        lower_limits = {'A':section.getfloat('lower_limit'), 'B':4}
        upper_limits = {'A':section.getfloat('upper_limit'), 'B':4}

        lower_limit = lower_limits[fiber]
        upper_limit = upper_limits[fiber]
        apertureset = master_aperset[fiber]
        
        # extract 1d spectra of the object
        spectra1d = extract_aperset(data, mask,
                        apertureset = apertureset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        message = 'Fiber {}: 1D spectra of {} orders extracted'.format(
                    fibercode, len(spectra1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
        
        # extract 1d error of the object
        error1d = extract_aperset(variance_map, mask,
                        apertureset = apertureset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                        variance    = True,
                    )
        message = 'Fiber {}: 1D error sum of {} orders extracted'.format(
                    fibercode, len(error1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
        
        # extract 1d raw flux summed up
        specraw1d = extract_aperset(raw_data, mask,
                        apertureset = apertureset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        message = 'Fiber {}: 1D raw spectra of {} orders extracted'.format(
                    fibercode, len(specraw1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
            
        # extract 1d spectra for straylight/background light
        background1d = extract_aperset(background, mask,
                        apertureset = apertureset,
                        lower_limit = lower_limit,
                        upper_limit = upper_limit,
                    )
        message = 'Fiber {}: 1D straylight of {} orders extracted'.format(
                    fibercode, len(background1d))
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        prefix = 'HIERARCH GAMSE EXTRACTION FIBER {} '.format(fiber)
        head.append((prefix + 'LOWER LIMIT', lower_limit))
        head.append((prefix + 'UPPER LIMIT', upper_limit))


        # optimal method
        result = extract_aperset_optimal(data, mask,
                    background  = background,
                    apertureset = master_aperset[fiber],
                    gain        = 1.02,
                    ron         = 2.72,
                    profilex    = profile_x,
                    disp_x_lst  = disp_x_lst,
                    main_disp   = 'x',
                    upper_clipping = 5,
                    recenter    = True,
                    mode        = mode,
                    profile_lst = profile[fiber],
                    )
        flux_opt_lst = result[0]
        flux_err_lst = result[1]
        back_opt_lst = result[2]
        #flux_sum_lst = result[3]
        #back_sum_lst = result[4]

        # pack spectrum
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            n = flux_sum.size
            # search for flat flux
            m = flat_spec_lst[fiber]['aperture']==aper
            flat_flux = flat_spec_lst[fiber][m][0]['flux']
            # read error/varriance and calc. error
            flux_err  = np.sqrt(error1d[aper]['flux_sum'])
            # read raw flux
            flux_raw  = specraw1d[aper]['flux_sum']
            # 1d flat flux sum and raw
            flat_sum  = flat_oned_lst[fiber][m][0]['flux']
            flat_raw  = flat_raw_lst[fiber][m][0]['flux']
            # background 1d flat
            back_flat = back_flat_1d[fiber][aper]['flux_sum']
            # background 1d flux
            back_flux = background1d[aper]['flux_sum']

            item = (aper, 0, n,
                    np.zeros(n, dtype=np.float64),  # wavelength
                    flux_sum,                       # flux_sum
                    flux_err,                       # flux_sum_err
                    np.zeros(n, dtype=np.int16),    # flux_sum_mask
                    flux_opt_lst[aper],             # flux_opt
                    flux_err_lst[aper],             # flux_opt_err
                    np.zeros(n, dtype=np.int16),    # flux_opt_mask
                    flux_raw,                       # flux_raw
                    flat_flux,                      # flat
                    flat_sum,
                    flat_raw,
                    back_flat,                      # background flat
                    back_flux,                      # background
                    )
            spec.append(item)
        spec = np.array(spec, dtype=spectype)

        # wavelength calibration
        # weight_lst has already been determined when doing the background
        spec, card_lst = reference_spec_wavelength(spec,
                            ref_calib_lst[fiber], weight_lst)

        # add the fit_filter keywords to card_lst
        card_lst.append(('PIXEL_RANGE', str(pixel_range)))
        card_lst.append(('ORDER_RANGE', str(order_range)))

        all_spec[fiber] = spec
        #all_cards[fiber] = card_lst
        prefix = 'HIERARCH GAMSE WLCALIB FIBER {} '.format(fiber)
        for key, value in card_lst:
            head.append((prefix + key, value))
        #newcards = combine_fiber_cards(all_cards)
        newspec = combine_fiber_spec(all_spec)
        #for key, value in newcards:
        #    key = 'HIERARCH GAMSE WLCALIB '+key
        #    head.append((key, value))
        # pack and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(newspec),
                    ])
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        extracted_fileid_lst.append(fileid)

    ###################### Extract Other Spectra ###############################

    for logitem in logtable:
        # logitem alias
        frameid = logitem['frameid']
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']
        objects = logitem['object']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        # filter out already extracted files
        if fileid in extracted_fileid_lst:
            continue

        fiberobj_lst = get_fiberobj_lst(objects, '|')
        fiberobj_str = get_fiberobj_string(fiberobj_lst, n_fiber)

        # filter out non-sci but not Comb/Comb files
        if imgtype != 'sci':
            if len(fiberobj_lst) != 2:
                continue
            elif len(list(
                filter(lambda v:v[1].lower()=='comb', fiberobj_lst)
                )) != 2:
                # filter out when objects != Comb/Comb
                continue
            else:
                pass


        message = 'FileID: {} ({}) OBJECT: {}'.format(
                    fileid, imgtype, fiberobj_str)
        logger.info(message)
        print(message)

        # read raw data
        filename = os.path.join(rawpath, '{}.fits'.format(fileid))
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

        # 2d image to extract the raw flux
        raw_data = data.copy()

        # creating the variance map to track the errors
        variance_map = np.maximum(data, 0) + overstd**2 + ron_bias**2

        # correct flat
        data = data/master_flatsens
        variance_map = variance_map/master_flatsens**2
        message = 'Flat field corrected'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # including the error of the sens.map corretion
        master_flatdsum_0 = 1.0*master_flatdsum  # bad hack for copying
        # making sure that not affected pixels (pixels with a value of 1) 
        # have no impact on the total error
        master_flatdsum_0[np.where(master_flatsens == 1.0)] = 1e99 
        variance_map = variance_map + data**2/master_flatdsum_0

        # background correction

        '''
        if len(fiberobj_lst)==1:
            section = config['reduce.background']
            ncols    = section.getint('ncols')
            distance = section.getfloat('distance')
            yorder   = section.getint('yorder')
            subtract = section.getboolean('subtract')
            excluded_frameids = section.get('excluded_frameids')
            excluded_frameids = parse_num_seq(excluded_frameids)
            
            if (subtract and frameid not in excluded_frameids) or \
               (not subtract and frameid in excluded_frameids):
            
                # find apertureset list for this item
                apersets = {}
                for (ifiber, objt) in fiberobj_lst:
                    fiber = chr(ifiber+65)
                    apersets[fiber] = master_aperset[fiber]
            
                figname = 'bkg_{}_sec.{}'.format(fileid, fig_format)
                fig_sec = os.path.join(figpath, figname)
            
                stray = find_background(data, mask,
                                aperturesets = apersets,
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
                bkgfig = BackgroundFigure()
                bkgfig.plot_background(data+stray, stray)
                bkgfig.suptitle('Background Correction for {}'.format(fileid))
                figname = 'bkg_{}_stray.{}'.format(fileid, fig_format)
                fig_stray = os.path.join(figpath, figname)
                bkgfig.savefig(fig_stray)
            
                message = 'FileID: {} - background corrected. max value = {}'.format(
                        fileid, stray.max())
            else:
                stray = None
                # put information into header
                prefix = 'HIERARCH GAMSE BACKGROUND '
                head.append((prefix + 'CORRECTED', False))
                message = 'FileID: {} - background not corrected.'.format(fileid)
            
            logger.info(message)
            print(message)
        '''

        background = np.zeros_like(data, dtype=data.dtype)

        fiber_obs_bkg_lst = {}
        fiber_sel_bkg_lst = {}
        fiber_scale_lst = {}
        for (ifiber, objname) in fiberobj_lst:
            fiber = chr(ifiber+65)
            #fibercode_cmd = struct.pack('>i', -257981040+ifiber).decode()
            fibercode = '[{}]'.format(fiber)

            result = get_xdisp_profile(data, master_aperset[fiber])
            aper_num_lst, aper_pos_lst, aper_brt_lst = result

            weight_lst = get_calib_weight_lst(ref_calib_lst[fiber],
                            obsdate = head[statime_key],
                            exptime = head[exptime_key],
                            )
            ny, nx = data.shape
            pixel_lst = np.repeat(nx//2, aper_num_lst.size)
            results = reference_pixel_wavelength(pixel_lst, aper_num_lst,
                        ref_calib_lst[fiber], weight_lst)
            aper_ord_lst, aper_wav_lst = results

            obs_bkg_obj = BackgroundLight(
                            aper_num_lst = aper_num_lst,
                            aper_pos_lst = aper_pos_lst,
                            aper_brt_lst = aper_brt_lst,
                            aper_ord_lst = aper_ord_lst,
                            aper_wav_lst = aper_wav_lst,
                            )

            # find objtype to search for the same kind of background light
            if objname.lower() in ['comb', 'fp']:
                objtype = objname.lower()
            else:
                objtype = 'star'

            find_background = False
            selected_bkg = find_best_background(saved_bkg_lst, obs_bkg_obj,
                                fiber, objname, head[statime_key], objtype)
            if selected_bkg is None:
                # not found in today's data
                database_path = config['reduce.background'].get('database_path')
                database_path = os.path.expanduser(database_path)
                selected_bkg = select_background_from_database(database_path,
                                shape     = data.shape,
                                fiber     = fiber,
                                direction = config['data'].get('direction'),
                                objtype   = objtype,
                                obj       = objname,
                                )
                if selected_bkg is None:
                    # not found either in database
                    message = 'Error: No background found in the database'
                    logger.info(logger_prefix + message)
                    print(screen_prefix + message)
                else:
                    # background found in database
                    find_background = True
            else:
                # background found in the same dataset
                find_background = True

            if find_background:
                scale = obs_bkg_obj.find_brightness_scale(selected_bkg)

                # pack to result list
                fiber_obs_bkg_lst[fiber] = obs_bkg_obj
                fiber_sel_bkg_lst[fiber] = selected_bkg
                fiber_scale_lst[fiber] = scale

                message = ('Use background of {} for Fiber {}. '
                           'scale = {:6.3f}'.format(
                            selected_bkg.info['fileid'], fibercode, scale))
                logger.info(logger_prefix + message)
                print(screen_prefix + message)

            background = background + selected_bkg.data*scale

        # plot brightness profile
        figname = 'bkgbrt_{}.png'.format(fileid)
        figfilename = os.path.join(figpath, figname)
        fig_bp = BrightnessProfileFigure(
                    fiber_obs_bkg_lst,
                    fiber_sel_bkg_lst,
                    fiber_scale_lst,
                    title = 'Brightness Profile of {}'.format(fileid),
                    filename = figfilename,
                )
        fig_bp.close()

        # plot stray light
        figname = 'bkg2d_{}.{}'.format(fileid, fig_format)
        figfilename = os.path.join(figpath, figname)
        fig_bkg = BackgroundFigure(data, background,
                title = 'Background Correction for {}'.format(fileid),
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
        all_spec  = {}   # use to pack final 1d spectrum
        #all_cards = {}
        lower_limits = {'A':section.getfloat('lower_limit'), 'B':4}
        upper_limits = {'A':section.getfloat('upper_limit'), 'B':4}
        for ifiber, obj in fiberobj_lst:
            fiber = chr(ifiber+65)
            #fibercode_cmd = struct.pack('>i', -257981040+ifiber).decode()
            fibercode = '[{}]'.format(fiber)

            #all_cards[fiber] = []

            lower_limit = lower_limits[fiber]
            upper_limit = upper_limits[fiber]
            apertureset = master_aperset[fiber]
            
            # extract 1d spectra of the object
            spectra1d = extract_aperset(data, mask,
                            apertureset = apertureset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                        )
            message = 'Fiber {}: 1D spectra of {} orders extracted'.format(
                        fibercode, len(spectra1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)
            
            # extract 1d error of the object
            error1d = extract_aperset(variance_map, mask,
                            apertureset = apertureset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                            variance    = True,
                        )
            message = 'Fiber {}: 1D error sum of {} orders extracted'.format(
                        fibercode, len(error1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)
            
            # extract 1d raw flux summed up
            specraw1d = extract_aperset(raw_data, mask,
                            apertureset = apertureset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                        )
            message = 'Fiber {}: 1D raw spectra of {} orders extracted'.format(
                        fibercode, len(specraw1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)
            
            # extract 1d spectra for stray light
            background1d = extract_aperset(background, mask,
                            apertureset = apertureset,
                            lower_limit = lower_limit,
                            upper_limit = upper_limit,
                        )
            message = 'Fiber {}: 1D straylight of {} orders extracted'.format(
                        fibercode, len(background1d))
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            prefix = 'HIERARCH GAMSE EXTRACTION FIBER {} '.format(fiber)
            head.append((prefix + 'LOWER LIMIT', lower_limit))
            head.append((prefix + 'UPPER LIMIT', upper_limit))


            # optimal method
            result = extract_aperset_optimal_multifiber(data, mask,
                        background      = background,
                        apertureset_lst = master_aperset,
                        extract_fiber   = fiber,
                        gain            = 1.02,
                        ron             = 2.72,
                        profilex        = profile_x,
                        disp_x_lst      = disp_x_lst,
                        main_disp       = 'x',
                        all_profile_lst = profile,
                        )
            flux_opt_lst = result[0]
            flux_err_lst = result[1]
            back_opt_lst = result[2]
            #flux_sum_lst = result[3]
            #back_sum_lst = result[4]

            # pack spectrum
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                n = flux_sum.size
                # search for flat flux
                m = flat_spec_lst[fiber]['aperture']==aper
                flat_flux = flat_spec_lst[fiber][m][0]['flux']
                #read error/varriance and calc. error
                flux_err  = np.sqrt(error1d[aper]['flux_sum'])
                #read raw flux
                flux_raw  = specraw1d[aper]['flux_sum']
                # 1d flat flux sum and raw
                flat_sum = flat_oned_lst[fiber][m][0]['flux']
                flat_raw = flat_raw_lst[fiber][m][0]['flux']
                # background 1d flat
                back_flat = back_flat_1d[fiber][aper]['flux_sum']
                # background 1d flux
                back_flux = background1d[aper]['flux_sum']

                item = (aper, 0, n,
                        np.zeros(n, dtype=np.float64),  # wavelength
                        flux_sum,                       # flux_sum
                        flux_err,                       # flux_sum_err
                        np.zeros(n, dtype=np.int16),    # flux_sum_mask
                        flux_opt_lst[aper],             # flux_opt
                        flux_err_lst[aper],             # flux_opt_err
                        np.zeros(n, dtype=np.int16),    # flux_opt_mask
                        flux_raw,                       # flux_raw
                        flat_flux,                      # flat
                        flat_sum,
                        flat_raw,
                        back_flat,                      # background flat
                        back_flux,                      # background
                        )
                spec.append(item)
            spec = np.array(spec, dtype=spectype)

            # wavelength calibration
            weight_lst = get_calib_weight_lst(ref_calib_lst[fiber],
                            obsdate = head[statime_key],
                            exptime = head[exptime_key],
                            )
            message_lst = ['Fiber {}: Wavelength calibration:'.format(fibercode)]
            for i, calib in enumerate(ref_calib_lst[fiber]):
                string = ' '*len(screen_prefix)
                string = string + '{} ({:4g} sec) {} weight = {:5.3f}'.format(
                            calib['fileid'], calib['exptime'],
                            calib['date-obs'], weight_lst[i])
                message_lst.append(string)
            message = os.linesep.join(message_lst)
            logger.info(logger_prefix + message)
            print(screen_prefix + message)

            spec, card_lst = reference_spec_wavelength(spec,
                                ref_calib_lst[fiber], weight_lst)

            # add the fit_filter keywords to card_lst
            card_lst.append(('PIXEL_RANGE', str(pixel_range)))
            card_lst.append(('ORDER_RANGE', str(order_range)))

            all_spec[fiber] = spec
            #all_cards[fiber] = card_lst
            prefix = 'HIERARCH GAMSE WLCALIB FIBER {} '.format(fiber)
            for key, value in card_lst:
                head.append((prefix + key, value))
                
        #newcards = combine_fiber_cards(all_cards)
        newspec = combine_fiber_spec(all_spec)
        #for key, value in newcards:
        #    key = 'HIERARCH GAMSE WLCALIB '+key
        #    head.append((key, value))
        # pack and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(newspec),
                    ])
        fname = '{}_{}.fits'.format(fileid, oned_suffix)
        filename = os.path.join(odspath, fname)
        hdu_lst.writeto(filename, overwrite=True)

        message = '1D spectra written to "{}"'.format(filename)
        logger.info(logger_prefix + message)
        print(screen_prefix + message)
