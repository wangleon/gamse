import os
import re
import shutil
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images, array_to_table, fix_pixels
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat  import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                                mosaic_spec)
from ...echelle.background import find_background
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, get_calib_from_header,
                                get_time_weight, find_caliblamp_offset,
                                reference_spec_wavelength,
                                reference_self_wavelength)
from ..common import plot_background_aspect1, FormattedInfo
from .common import (get_bias, get_mask, correct_overscan, TraceFigure,
                     select_calib_from_database)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg)

def reduce_doublefiber(config, logtable):
    """Reduce the multi-fiber data of Xinglong 2.16m HRS.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of observing log.

    """

    # extract keywords from config file
    section      = config['data']
    rawdata      = section.get('rawdata')
    statime_key  = section.get('statime_key')
    exptime_key  = section.get('exptime_key')
    direction    = section.get('direction')
    readout_mode = section.get('readout_mode')
    # if mulit-fiber, get fiber offset list from config file
    fiber_offsets = [float(v) for v in section.get('fiberoffset').split(',')]

    section     = config['reduce']
    midproc     = section.get('midproc')
    onedspec    = section.get('onedspec')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):   os.mkdir(report)
    if not os.path.exists(onedspec): os.mkdir(onedspec)
    if not os.path.exists(midproc):  os.mkdir(midproc)

    # define a fiber splitting function
    def get_fiberobj_lst(string):
        object_lst = [s.strip() for s in string.split(';')]
        fiberobj_lst = list(filter(lambda v: len(v[1])>0,
                                    enumerate(object_lst)))
        return fiberobj_lst

    n_fiber = 2

    ############################# parse bias ###################################

    bias, bias_card_lst= get_bias(config, logtable)

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for multi-fibers
    flat_groups = {chr(ifiber+65): {} for ifiber in range(n_fiber)}
    # flat_groups = {'A':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}
    #                'B':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}}

    for logitem in logtable:
        fiberobj_lst = [v.strip() for v in logitem['object'].split(';')]

        if n_fiber > len(fiberobj_lst):
            continue

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            objname = fiberobj_lst[ifiber].lower().strip()
            mobj = re.match('^flat[\s\S]*', objname)
            if mobj is not None:
                # the object name of the channel matches "flat ???"
            
                # check the lengthes of names for other channels
                # if this list has no elements (only one fiber) or has no
                # names, this frame is a single-channel flat
                other_lst = [name for i, name in enumerate(fiberobj_lst)
                                    if i != ifiber and len(name)>0]
                if len(other_lst)>0:
                    # this frame is not a single chanel flat. Skip
                    continue

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
    flat_sens_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_spec_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_info_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    aperset_lst   = {fiber: {} for fiber in sorted(flat_groups.keys())}

    # first combine the flats
    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        for flatname, item_lst in sorted(fiber_flat_lst.items()):
            nflat = len(item_lst)       # number of flat fieldings

            flat_filename = os.path.join(midproc,
                    'flat_{}_{}.fits'.format(fiber, flatname))
            aperset_filename = os.path.join(midproc,
                    'trace_flat_{}_{}.trc'.format(fiber, flatname))
            aperset_regname = os.path.join(midproc,
                    'trace_flat_{}_{}.reg'.format(fiber, flatname))
            trace_figname = os.path.join(report,
                    'trace_flat_{}_{}.{}'.format(fiber, flatname, fig_format))

            # get flat_data and mask_array for each flat group
            if mode=='debug' and os.path.exists(flat_filename) \
                and os.path.exists(aperset_filename):
                # read flat data and mask array
                hdu_lst = fits.open(flat_filename)
                flat_data = hdu_lst[0].data
                flat_mask = hdu_lst[1].data
                flat_norm = hdu_lst[2].data
                flat_sens = hdu_lst[3].data
                flat_spec = hdu_lst[4].data
                exptime    = hdu_lst[0].header[exptime_key]
                hdu_lst.close()
                aperset = load_aperture_set(aperset_filename)
            else:
                # if the above conditions are not satisfied, comine each flat
                data_lst = []
                head_lst = []
                exptime_lst = []

                print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))

                for i_item, logitem in enumerate(item_lst):
                    # read each individual flat frame
                    filename = os.path.join(rawdata, logitem['fileid']+'.fits')
                    data, head = fits.getdata(filename, header=True)
                    exptime_lst.append(head[exptime_key])
                    mask = get_mask(data, head)

                    # generate the mask for all images
                    sat_mask = (mask&4>0)
                    bad_mask = (mask&2>0)
                    if i_item == 0:
                        allmask = np.zeros_like(mask, dtype=np.int16)
                    allmask += sat_mask

                    # correct overscan for flat
                    data, card_lst = correct_overscan(data, head, readout_mode)
                    for key, valaue in card_lst:
                        head.append((key, value))

                    # correct bias for flat, if has bias
                    if bias is not None:
                        data = data - bias
                        message = 'Bias corrected'
                    else:
                        message = 'No bias. skipped bias correction'
                    logger.info(message)

                    # print info
                    message = ('FileId: {} {}'
                                '    exptime={:5g} sec'
                                '    Nsat={:6d}'
                                '    Q95={:5d}').format(
                                item['fileid'], item['object'], item['exptime'],
                                item['nsat'], item['q95'])
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

                # create the trace figure
                tracefig = TraceFigure()

                # if debackground before detecting the orders, then we lose the 
                # ability to detect the weak blue orders.
                #xnodes = np.arange(0, flat_data.shape[1], 200)
                #flat_debkg = simple_debackground(flat_data, mask_array, xnodes,
                # smooth=5)
                #aperset = find_apertures(flat_debkg, mask_array,
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
                aperset.save_reg(aperset_regname, fiber=fiber,
                                color={'A':'green','B':'yellow'}[fiber])

                # do the flat fielding
                # prepare the output midproc figures in debug mode
                if mode=='debug':
                    figname = 'flat_aperpar_{}_{}_%03d.{}'.format(
                                fiber, flatname, fig_format)
                    fig_aperpar = os.path.join(report, figname)
                else:
                    fig_aperpar = None
                            
                # prepare the name for slit figure
                figname = 'slit_flat_{}_{}.{}'.format(fiber, flatname, fig_format)
                fig_slit = os.path.join(report, figname)

                # prepare the name for slit file
                fname = 'slit_flat_{}_{}.dat'.format(fiber, flatname)
                slit_file = os.path.join(midproc, fname)

                section = config['reduce.flat']

                flat_sens, flat_spec = get_fiber_flat(
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

                # pack results and save to fits
                hdu_lst = fits.HDUList([
                            fits.PrimaryHDU(flat_data, head),
                            fits.ImageHDU(flat_mask),
                            fits.ImageHDU(flat_norm),
                            fits.ImageHDU(flat_sens),
                            fits.BinTableHDU(flat_spec),
                            ])
                hdu_lst.writeto(flat_filename, overwrite=True)

                # now flt_data and mask_array are prepared

            # append the flat data and mask
            flat_data_lst[fiber][flatname] = flat_data
            flat_mask_lst[fiber][flatname] = flat_mask
            flat_norm_lst[fiber][flatname] = flat_norm
            flat_sens_lst[fiber][flatname] = flat_sens
            flat_spec_lst[fiber][flatname] = flat_spec
            flat_info_lst[fiber][flatname] = {'exptime': exptime}
            aperset_lst[fiber][flatname]   = aperset

            # continue to the next colored flat
        # continue to the next fiber

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    master_aperset = {}

    flat_fiber_lst = []

    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        fiber_flat_lst = flat_groups[fiber]

        # determine the mosaiced flat filename
        flat_fiber_file = os.path.join(midproc,
                            'flat_{}.fits'.format(fiber))
        trac_fiber_file = os.path.join(midproc,
                            'trace_{}.trc'.format(fiber))
        treg_fiber_file = os.path.join(midproc,
                            'trace_{}.reg'.format(fiber))

        if len(fiber_flat_lst) == 1:
            # there's only ONE "color" of flat
            flatname = list(fiber_flat_lst)[0]

            # copy the flat fits
            oriname = 'flat_{}_{}.fits'.format(fiber, flatname)
            shutil.copyfile(os.path.join(midproc, oriname), flat_fiber_file)

            '''
            # copy the trc file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.trc'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.trc'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), trac_fiber_file)

            # copy the reg file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.reg'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.reg'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), treg_fiber_file)
            '''

            flat_map = flatmap_lst[fiber][flatname]
    
            # no need to mosaic aperset
            master_aperset[fiber] = list(aperset_lst[fiber].values())[0]
        else:
            # mosaic apertures
            section = config['reduce.flat']
            # determine the mosaic order
            name_lst = sorted(flat_info_lst[fiber],
                        key=lambda x: flat_info_lst[fiber].get(x)['exptime'])

            # if there is no flat data in this fiber. continue
            if len(aperset_lst[fiber])==0:
                continue

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
            # mosaic sensitivity map
            flat_sens = mosaic_images(flat_sens_lst[fiber],
                                        master_aperset[fiber])
            # mosaic 1d spectra of flats
            flat_spec = mosaic_spec(flat_spec_lst[fiber], master_aperset)

            # change contents of several lists
            flat_data_lst[fiber] = flat_data
            flat_mask_lst[fiber] = flat_mask
            flat_norm_lst[fiber] = flat_norm
            flat_sens_lst[fiber] = flat_sens
            flat_spec_lst[fiber] = flat_spec

            flat_fiber_lst.append(fiber)
    
            # pack and save to fits file
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data),
                        fits.ImageHDU(flat_mask),
                        fits.ImageHDU(flat_norm),
                        fits.ImageHDU(flat_sens),
                        fits.BinTableHDU(flat_spec),
                        ])
            hdu_lst.writeto(flat_fiber_file, overwrite=True)

    # fill blank fibers
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        if fiber not in master_aperset:
            master_aperset[fiber] = master_aperset['A'].copy()
            offset = fiber_offsets[ifiber-1]
            master_aperset[fiber].add_offset(offset)

    # align different fibers

    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)

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
            message = 'fiber {}, aperture offset = {}'.format(fiber, offset)
            print(message)
            logger.info(message)

            # correct the aperture offset
            master_aperset[fiber].shift_aperture(-offset)

    # find all the aperture list for all fibers
    allmax_aper = -99
    allmin_aper = 999
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        allmax_aper = max(allmax_aper, max(master_aperset[fiber]))
        allmin_aper = min(allmin_aper, min(master_aperset[fiber]))

    #fig = plt.figure(dpi=150)
    #ax = fig.gca()
    #test_data = {'A': np.ones((2048, 2048))+1,
    #             'B': np.ones((2048, 2048))+2}

    # pack all aperloc into a single list
    all_aperloc_lst = []
    for fiber in flat_fiber_lst:
        aperset = master_aperset[fiber]
        for aper, aperloc in aperset.items():
            x, y = aperloc.get_position()
            center = aperloc.get_center()
            all_aperloc_lst.append([fiber, aper, aperloc, center])
            #ax.plot(x, y, color='gy'[ifiber], lw=1)

    # mosaic flat map
    sorted_aperloc_lst = sorted(all_aperloc_lst, key=lambda x:x[3])
    h, w = flat_map.shape
    master_flatdata = np.ones_like(flat_data)
    master_flatmask = np.ones_like(mask_data)
    master_flatmap  = np.ones_like(flat_map)
    master_flatnorm = np.ones_like(flat_norm)
    yy, xx = np.mgrid[:h, :w]
    prev_line = np.zeros(w)
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
        master_flatsens[mask] = flat_sens_lst[fiber][mask]
        prev_line = next_line
    # parse the last order
    mask = yy >= prev_line
    master_flatdata[mask] = flat_data_lst[next_fiber][mask]
    master_flatmask[mask] = flat_mask_lst[next_fiber][mask]
    master_flatnorm[mask] = flat_norm_lst[next_fiber][mask]
    master_flatsens[mask] = flat_sens_lst[next_fiber][mask]

    #ax.imshow(master_flatmap, alpha=0.6)
    #plt.show()
    #print(h, w)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(master_flatdata),
                fits.ImageHDU(master_flatmask),
                fits.ImageHDU(master_flatnorm),
                fits.ImageHDU(master_flatsens),
                ])
    hdu_lst.writeto(flat_file, overwrite=True)

    ############################## Extract ThAr ################################

    # get the data shape
    ny, nx = flat_sens.shape

    # define dtype of 1-d spectra for all fibers
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, nx)),
            ('flux',       (np.float32, nx)),
            ('flat',       (np.float32, nx)),
            ('background', (np.float32, nx)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})

    calib_lst = {}
    # calib_lst is a hierarchical dict of calibration results
    # calib_lst = {
    #       'frameid1': {'A': calib_dict1, 'B': calib_dict2, ...},
    #       'frameid2': {'A': calib_dict1, 'B': calib_dict2, ...},
    #       ... ...
    #       }
    count_thar = 0
    for logitem in logtable:

        frameid = logitem['frameid']
        imgtype = logitem['imgtype']
        fileid  = logitem['fileid']
        exptime = logitem['exptime']

        # prepare message prefix
        logger_prefix = 'FileID: {} - '.format(fileid)
        screen_prefix = '    - '

        if imgtype != 'cal':
            continue

        fiberobj_lst = [v.strip().lower()
                        for v in logitem['object'].split(';')]

        # check if there's any other objects
        has_others = False
        for fiberobj in fiberobj_lst:
            if len(fiberobj)>0 and fiberobj != 'thar':
                has_others = True
        if has_others:
            continue

        # now all objects in fiberobj_lst must be thar

        count_thar += 1
        print('Wavelength Calibration for {}'.format(fileid))

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, card_lst = correct_overscan(data, head, readout_mode)
        for key, value in card_lst:
            head.append((key, value))

        # correct bias for ThAr, if has bias
        if bias is None:
            message = 'No Bias'
        else:
            data = data - bias
            message = 'Bias corrected. Mean = {:.2f}'.format(bias.mean())
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        head.append(('HIERARCH GAMSE BACKGROUND CORRECTED', False))

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if fiberobj_lst[ifiber] != 'thar':
                continue

            section = config['reduce.extract']
            spectra1d = extract_aperset(data, mask,
                        apertureset = master_aperset[fiber],
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )

            # pack to a structured array
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                spec.append((aper, 0, flux_sum.size,
                        np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
            spec = np.array(spec, dtype=spectype)

            wlcalib_fig = os.path.join(report,
                    'wlcalib_{}_{}.{}'.format(fileid, fiber, fig_format))

            section = config['reduce.wlcalib']

            title = '{}.fits - Fiber {}'.format(fileid, fiber)

            if count_thar == 1:
                # this is the first ThAr frame in this observing run
                if section.getboolean('search_database'):
                    # find previouse calibration results
                    database_path = section.get('database_path')
                    database_path = os.path.expanduser(database_path)

                    message = ('Searching for archive wavelength calibration'
                               'file in "{}"'.format(database_path))
                    logger.info(logger_prefix + message)

                    ref_spec, ref_calib = select_calib_from_database(
                            database_path, head[statime_key])

                    if ref_spec is None or ref_calib is None:

                        message = ('Did not find any archive wavelength'
                                   'calibration file')
                        logger.info(logger_prefix + message)

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
                            )
                    else:
                        # if success, run recalib
                        # determine the direction
                        message = 'Found archive wavelength calibration file'
                        logger.info(message)

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

                        # determine the name of the output figure during lamp
                        # shift finding.
                        if mode == 'debug':
                            figname1 = 'lamp_ccf_{:+2d}_{:+03d}.png'
                            figname2 = 'lamp_ccf_scatter.png'
                            fig_ccf     = os.path.join(report, figname1)
                            fig_scatter = os.path.join(report, figname2)
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

                        message = 'Aperture offset = {}; Pixel offset = {}'.format(
                                    aperture_koffset, pixel_koffset)
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
                    )

            # add more infos in calib
            calib['fileid']   = fileid
            calib['date-obs'] = head[statime_key]
            calib['exptime']  = head[exptime_key]

            # reference the ThAr spectra
            spec, card_lst, identlist = reference_self_wavelength(spec, calib)

            # save calib results into fits header
            for key, value in card_lst:
                key = 'HIERARCH GAMSE WLCALIB '+key
                head.append((key, value))

            # save onedspec into FITS
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        fits.BinTableHDU(identlist),
                        ])
            filename = os.path.join(midproc,
                                    'wlcalib.{}.{}.fits'.format(fileid, fiber))
            hdu_lst.writeto(filename, overwrite=True)

            # pack to calib_lst
            if frameid not in calib_lst:
                calib_lst[frameid] = {}
            calib_lst[frameid][fiber] = calib
            
        # fiber loop ends here

    # print fitting summary
    fmt_string = (' [{:3d}] {}'
                    ' - fiber {:1s} ({:4g} sec)'
                    ' - {:4d}/{:4d} r.m.s. = {:7.5f}')
    for frameid, calib_fiber_lst in sorted(calib_lst.items()):
        for fiber, calib in sorted(calib_fiber_lst.items()):
            print(fmt_string.format(frameid, calib['fileid'], fiber,
                calib['exptime'], calib['nuse'], calib['ntot'], calib['std']))

    # print promotion and read input frameid list
    ref_frameid_lst  = {}
    ref_calib_lst    = {}
    ref_datetime_lst = {}
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        while(True):
            string = input('Select References for fiber {}: '.format(fiber))
            ref_frameid_lst[fiber]  = []
            ref_calib_lst[fiber]    = []
            ref_datetime_lst[fiber] = []
            succ = True
            for s in string.split(','):
                s = s.strip()
                if len(s)>0 and s.isdigit() and int(s) in calib_lst:
                    frameid = int(s)
                    calib   = calib_lst[frameid]
                    ref_frameid_lst[fiber].append(frameid)
                    if fiber in calib:
                        usefiber = fiber
                    else:
                        usefiber = list(calib.keys())[0]
                        print(('Warning: no ThAr for fiber {}. '
                                'Use fiber {} instead').format(fiber, usefiber))
                    use_calib = calib[usefiber]
                    ref_calib_lst[fiber].append(use_calib)
                    ref_datetime_lst[fiber].append(use_calib['date-obs'])
                else:
                    print('Warning: "{}" is an invalid calib frame'.format(s))
                    succ = False
                    break
            if succ:
                break
            else:
                continue

    #################### Extract Science Spectrum ##############################

    for logitem in logtable:

        # logitem alias
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']

        if imgtype != 'sci':
            continue

        filename = os.path.join(rawdata, fileid+'.fits')

        logger.info('FileID: {} ({}) - start reduction: {}'.format(
            fileid, imgtype, filename))

        # read raw data
        data, head = fits.getdata(filename, header=True)
        mask = get_mask(data, head)

        # correct overscan
        data, card_lst = correct_overscan(data, head, readout_mode)
        for key, value in card_lst:
            head.append((key, value))
        message = 'Overscan corrected.'
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

        # correct flat
        data = data/flat_sens
        message = 'Flat corrected.'
        logger.info(logger_prefix + message)
        print(screen_prefix + message)

        # correct background
        fiberobj_lst = [v.strip().lower() for v in logitem['object'].split(';')]
        fig_sec = os.path.join(report,
                  'bkg_{}_sec.{}'.format(fileid, fig_format))

        # find apertureset list for this item
        apersets = {}
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if len(fiberobj_lst[ifiber])>0:
                apersets[fiber] = master_aperset[fiber]

        section = config['reduce.background']
        stray = find_background(data, mask,
                aperturesets = apersets,
                ncols        = section.getint('ncols'),
                distance     = section.getfloat('distance'),
                yorder       = section.getint('yorder'),
                fig_section  = fig_sec,
                )
        data = data - stray

        # plot stray light
        fig_stray = os.path.join(report,
                    'bkg_{}_stray.{}'.format(fileid, fig_format))
        plot_background_aspect1(data+stray, stray, fig_stray)

        message = 'FileID: {} - background corrected. max value = {}'.format(
                fileid, stray.max())
        logger.info(message)
        print(message)

        # extract 1d spectrum
        section = config['reduce.extract']
        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            if fiberobj_lst[ifiber]=='':
                # nothing in this fiber
                continue
            lower_limits = {'A':section.getfloat('lower_limit'), 'B':4}
            upper_limits = {'A':section.getfloat('upper_limit'), 'B':4}

            spectra1d = extract_aperset(data, mask,
                            apertureset = master_aperset[fiber],
                            lower_limit = lower_limits[fiber],
                            upper_limit = upper_limits[fiber],
                        )

            fmt_string = ('FileID: {}'
                            ' - fiber {}'
                            ' - 1D spectra of {} orders extracted')
            message = fmt_string.format(fileid, fiber, len(spectra1d))
            logger.info(message)
            print(message)

            # pack spectrum
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                item = (aper, 0, flux_sum.size,
                        np.zeros_like(flux_sum, dtype=np.float64),
                        flux_sum
                        )
                spec.append(item)
            spec = np.array(spec, dtype=spectype)

            # wavelength calibration
            weight_lst = get_time_weight(ref_datetime_lst[fiber],
                                        head[statime_key])

            message = ('FileID: {} - fiber {}'
                        ' - wavelength calibration weights: {}').format(
                        fileid, fiber,
                        ','.join(['%8.4f'%w for w in weight_lst])
                        )
            logger.info(message)
            print(message)

            spec, card_lst = reference_spec_wavelength(spec,
                                ref_calib_lst[fiber], weight_lst)

            prefix = 'HIERARCH GAMSE WLCALIB '
            for key, value in card_lst:
                head.append((prefix + key, value))

            # pack and save to fits
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        ])
            fname = '{}_{}_{}.fits'.format(fileid, fiber, oned_suffix)
            filename = os.path.join(onedspec, fname)
            hdu_lst.writeto(filename, overwrite=True)

            message = 'FileID: {} - Spectra written to {}'.format(
                        fileid, filename)
            logger.info(message)
            print(message)
