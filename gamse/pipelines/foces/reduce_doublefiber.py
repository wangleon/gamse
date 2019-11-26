import os
import re
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from .common import (all_columns, print_wrapper, get_mask, correct_overscan,
                     TraceFigure)
from .flat import (smooth_aperpar_A, smooth_aperpar_k, smooth_aperpar_c,
                   smooth_aperpar_bkg
                   )
from ..common import plot_background_aspect1, FormattedInfo
from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ...echelle.flat import (get_fiber_flat, mosaic_flat_auto, mosaic_images,
                             mosaic_speclist)
from ...echelle.extract import extract_aperset
from ...echelle.wlcalib import (wlcalib, recalib, select_calib_from_database,
                                get_time_weight, find_caliblamp_offset,
                                reference_wavelength,
                                reference_self_wavelength,
                                combine_fiber_cards,
                                combine_fiber_spec,
                                combine_fiber_identlist,
                                )
from ...echelle.background import find_background, simple_debackground

def reduce_doublefiber(logtable, config):
    """Data reduction for multiple-fiber configuration.

    Args:
        logtable (:class:`astropy.table.Table`): The observing log.
        config (:class:`configparser.ConfigParser`): The configuration of
            reduction.

    """

    # extract keywords from config file
    section = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')
    # if mulit-fiber, get fiber offset list from config file
    fiber_offsets = [float(v) for v in section.get('fiberoffset').split(',')]
    section = config['reduce']
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

    # find the maximum length of fileid
    maxlen_fileid = 0
    for fname in os.listdir(rawdata):
        if fname[-5:] == '.fits':
            fileid = fname[0:-5]
            maxlen_fileid = max(maxlen_fileid, len(fileid))
    # now the maxlen_fileid is the maximum length of fileid

    # initialize printing infomation
    pinfo1 = FormattedInfo(all_columns, ['frameid', 'fileid', 'imgtype',
                'object', 'exptime', 'obsdate', 'nsat', 'q95'])
    pinfo2 = pinfo1.add_columns([('overscan', 'float', '{:^8s}', '{1:8.2f}')])

    # count the number of fibers
    n_fiber = 1
    for logitem in logtable:
        n = len(logitem['object'].split(';'))
        n_fiber = max(n_fiber, n)
    message = ', '.join(['multi_fiber = True',
                         'number of fiber = {}'.format(n_fiber)])
    print(message)

    ################################ parse bias ################################
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        bias, head = fits.getdata(bias_file, header=True)
        bias_card_lst = [card for card in head.cards
                            if card.keyword[0:10]=='GAMSE BIAS']
        logger.info('Load bias from image: %s'%bias_file)
    else:
        # read each individual CCD
        bias_data_lst = []
        bias_head_lst = []
        bias_fileid_lst = []

        for logitem in logtable:
            if logitem['object'].strip().lower()=='bias':
                fname = logitem['fileid']+'.fits'
                filename = os.path.join(rawdata, fname)
                data, head = fits.getdata(filename, header=True)
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data)
                data, card_lst, overmean = correct_overscan(data, mask)
                # head['BLANK'] is only valid for integer arrays.
                if 'BLANK' in head:
                    del head['BLANK']
                for key, value in card_lst:
                    head.append((key, value))

                # print info
                if len(bias_data_lst) == 0:
                    print('* Combine Bias Images: {}'.format(bias_file))
                    print(' '*2 + pinfo2.get_separator())
                    print(' '*2 + pinfo2.get_title())
                    print(' '*2 + pinfo2.get_separator())
                string = pinfo2.get_format().format(logitem, overmean)
                print(' '*2 + print_wrapper(string, logitem))

                bias_data_lst.append(data)
                bias_head_lst.append(head)
                bias_fileid_lst.append(logitem['fileid'])

        n_bias = len(bias_data_lst)         # number of bias images
        has_bias = n_bias > 0

        if has_bias:
            # there is bias frames
            print(' '*2 + pinfo2.get_separator())

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            section = config['reduce.bias']
            bias = combine_images(bias_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('cosmic_clip'),
                    maxiter    = section.getint('maxiter'),
                    mask       = (None, 'max')[n_bias>=3],
                    )

            # initialize card list for bias header
            bias_card_lst = []
            bias_card_lst.append(('HIERARCH GAMSE BIAS NFILE', n_bias))

            # move cards related to OVERSCAN corrections from individual headers
            # to bias header
            for ifile, (head, fileid) in enumerate(zip(bias_head_lst,
                                                       bias_fileid_lst)):
                key_prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ifile)
                bias_card_lst.append((key_prefix+' FILEID', fileid))

                # move OVERSCAN cards to bias header
                for card in head.cards:
                    mobj = re.match('^GAMSE (OVERSCAN[\s\S]*)', card.keyword)
                    if mobj is not None:
                        newkey = key_prefix + ' ' + mobj.group(1)
                        bias_card_lst.append((newkey, card.value))

            ############## bias smooth ##################
            key_prefix = 'HIERARCH GAMSE BIAS SMOOTH'
            section = config['reduce.bias']
            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                if smooth_method in ['gauss', 'gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')

                    bias_smooth = gaussian_filter(bias,
                                    sigma=smooth_sigma, mode=smooth_mode)

                    # write information to FITS header
                    bias_card_lst.append((key_prefix,           True))
                    bias_card_lst.append((key_prefix+' METHOD', 'GAUSSIAN'))
                    bias_card_lst.append((key_prefix+' SIGMA',  smooth_sigma))
                    bias_card_lst.append((key_prefix+' MODE',   smooth_mode))
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth
            else:
                # bias not smoothed
                bias_card_lst.append((key_prefix, False))

            # create new FITS Header for bias
            head = fits.Header()
            for card in bias_card_lst:
                head.append(card)
            fits.writeto(bias_file, bias, header=head, overwrite=True)

            message = 'Bias image written to "{}"'.format(bias_file)
            logger.info(message)
            print(message)

        else:
            # no bias found
            pass

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
    flat_norm_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_mask_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    aperset_lst   = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_info_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_spec_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}

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
                flat_data  = hdu_lst[0].data
                exptime    = hdu_lst[0].header[exptime_key]
                mask_array = hdu_lst[1].data
                hdu_lst.close()
                aperset = load_aperture_set(aperset_filename)
            else:
                # if the above conditions are not satisfied, comine each flat
                data_lst = []
                head_lst = []
                exptime_lst = []

                print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
                print(' '*2 + pinfo2.get_separator())
                print(' '*2 + pinfo2.get_title())
                print(' '*2 + pinfo2.get_separator())

                for i_item, logitem in enumerate(item_lst):
                    # read each individual flat frame
                    filename = os.path.join(rawdata, logitem['fileid']+'.fits')
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
                    data, card_lst, overmean = correct_overscan(data, mask)
                    # head['BLANK'] is only valid for integer arrays.
                    if 'BLANK' in head:
                        del head['BLANK']
                    for key, value in card_lst:
                        head.append((key, value))

                    # correct bias for flat, if has bias
                    if has_bias:
                        data = data - bias
                        logger.info('Bias corrected')
                    else:
                        logger.info('No bias. skipped bias correction')

                    # print info
                    string = pinfo2.get_format().format(logitem, overmean)
                    print(' '*2 + print_wrapper(string, logitem))

                    data_lst.append(data)

                print(' '*2 + pinfo2.get_separator())

                if nflat == 1:
                    flat_data = data_lst[0]
                else:
                    data_lst = np.array(data_lst)
                    flat_data = combine_images(data_lst,
                                    mode       = 'mean',
                                    upper_clip = 10,
                                    maxiter    = 5,
                                    mask       = (None, 'max')[nflat>3],
                                    )

                # get mean exposure time and write it to header
                head = fits.Header()
                exptime = np.array(exptime_lst).mean()
                head[exptime_key] = exptime

                # find saturation mask
                sat_mask = allmask > nflat/2.
                mask_array = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

                # pack results and save to fits
                hdu_lst = fits.HDUList([
                            fits.PrimaryHDU(flat_data, head),
                            fits.ImageHDU(mask_array),
                            ])
                hdu_lst.writeto(flat_filename, overwrite=True)

                # now flt_data and mask_array are prepared

                # create the trace figure
                tracefig = TraceFigure()

                # if debackground before detecting the orders, then we lose the 
                # ability to detect the weak blue orders.
                #xnodes = np.arange(0, flat_data.shape[1], 200)
                #flat_debkg = simple_debackground(flat_data, mask_array, xnodes,
                # smooth=5)
                #aperset = find_apertures(flat_debkg, mask_array,
                section = config['reduce.trace']
                aperset = find_apertures(flat_data, mask_array,
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
                tracefig.suptitle('Trace for {}'.format(flat_filename), fontsize=15)
                tracefig.savefig(trace_figname)

                aperset.save_txt(aperset_filename)
                aperset.save_reg(aperset_regname, fiber=fiber,
                                color={'A':'green','B':'yellow'}[fiber])

            '''
            # correct background for flat
            fig_sec = os.path.join(report,
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
            fig_stray = os.path.join(report,
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
            flat_norm_lst[fiber][flatname] = flat_data/exptime
            flat_mask_lst[fiber][flatname] = mask_array
            aperset_lst[fiber][flatname]   = aperset
            flat_info_lst[fiber][flatname] = {'exptime': exptime}

    ########################### Get flat fielding ##############################
    flatmap_lst = {}

    for fiber, fiber_group in sorted(flat_groups.items()):
        for flatname in sorted(fiber_group.keys()):

            # get filename of flat
            flat_filename = os.path.join(midproc,
                    'flat_{}_{}.fits'.format(fiber, flatname))

            hdu_lst = fits.open(flat_filename, mode='update')
            if len(hdu_lst)>=3:
                # sensitivity map already exists in fits file
                flatmap = hdu_lst[2].data
                hdu_lst.close()
            else:
                # do flat fielding
                print('*** Start parsing flat fielding: %s ***'%flat_filename)
                fig_aperpar = {
                    'debug': os.path.join(report,
                            'flat_aperpar_{}_{}_%03d.{}'.format(
                                fiber, flatname, fig_format)),
                    'normal': None,
                    }[mode]

                fig_slit = os.path.join(report,
                                'slit_flat_{}_{}.{}'.format(
                                    fiber, flatname, fig_format))
    
                section = config['reduce.flat']
    
                flatmap, flatspec = get_fiber_flat(
                            data            = flat_data_lst[fiber][flatname],
                            mask            = flat_mask_lst[fiber][flatname],
                            apertureset     = aperset_lst[fiber][flatname],
                            slit_step       = section.getint('slit_step'),
                            nflat           = len(flat_groups[fiber][flatname]),
                            q_threshold     = section.getfloat('q_threshold'),
                            smooth_A_func   = smooth_aperpar_A,
                            smooth_k_func   = smooth_aperpar_k,
                            smooth_c_func   = smooth_aperpar_c,
                            smooth_bkg_func = smooth_aperpar_bkg,
                            fig_aperpar     = fig_aperpar,
                            fig_overlap     = None,
                            fig_slit        = fig_slit,
                            slit_file       = None,
                            )

                flat_spec_lst[fiber][flatname] = flatspec

                for aper, spec in sorted(flatspec.items()):
                    fig = plt.figure(dpi=150)
                    ax = fig.gca()
                    ax.plot(spec)
                    fig.savefig('flatspec_%s_%s_%d_simps.png'%(fiber, flatname, aper))
                    plt.close(fig)
                
                # append the sensivity map to fits file
                hdu_lst.append(fits.ImageHDU(flatmap))
                # write back to the original file
                hdu_lst.flush()
    
            # append the flatmap
            if fiber not in flatmap_lst:
                flatmap_lst[fiber] = {}
            flatmap_lst[fiber][flatname] = flatmap
    
            # continue to the next colored flat
        # continue to the next fiber

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    # master aperset is a dict of {fiber: aperset}.
    master_aperset = {}

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

            master_aperset[fiber] = mosaic_flat_auto(
                    aperture_set_lst = aperset_lst[fiber],
                    max_count        = section.getfloat('mosaic_maxcount'),
                    name_lst         = name_lst,
                    )
            # mosaic original flat images
            flat_data = mosaic_images(flat_data_lst[fiber],
                                        master_aperset[fiber])
            # mosaic flat mask images
            mask_data = mosaic_images(flat_mask_lst[fiber],
                                        master_aperset[fiber])
            # mosaic sensitivity map
            flat_map = mosaic_images(flatmap_lst[fiber],
                                        master_aperset[fiber])
            # mosaic exptime-normalized flat images
            flat_norm = mosaic_images(flat_norm_lst[fiber],
                                        master_aperset[fiber])
            # mosaic 1d spectra of flats
            flat_spec = mosaic_speclist(flat_spec_lst[fiber],
                                        master_aperset[fiber])

            # change contents of several lists
            flat_data_lst[fiber] = flat_data
            flat_mask_lst[fiber] = mask_data
            flatmap_lst[fiber]   = flat_map
            flat_norm_lst[fiber] = flat_norm
            flat_spec_lst[fiber] = flat_spec
    
            # pack and save to fits file
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data),
                        fits.ImageHDU(mask_data),
                        fits.ImageHDU(flat_map),
                        fits.ImageHDU(flat_norm),
                        ])
            hdu_lst.writeto(flat_fiber_file, overwrite=True)

            for aper, spec in flat_spec.items():
                fig = plt.figure(dpi=150)
                ax = fig.gca()
                ax.plot(spec)
                plt.savefig('tmp/flatspec_%s_%d.png'%(fiber,aper))
                plt.close(fig)

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
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
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
        master_flatmap[mask]  = flatmap_lst[fiber][mask]
        master_flatnorm[mask] = flat_norm_lst[fiber][mask]
        prev_line = next_line
    # parse the last order
    mask = yy >= prev_line
    master_flatdata[mask] = flat_data_lst[next_fiber][mask]
    master_flatmask[mask] = flat_mask_lst[next_fiber][mask]
    master_flatmap[mask] = flatmap_lst[next_fiber][mask]
    master_flatnorm[mask] = flat_norm_lst[next_fiber][mask]

    #ax.imshow(master_flatmap, alpha=0.6)
    #plt.show()
    #print(h, w)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(master_flatdata),
                fits.ImageHDU(master_flatmask),
                fits.ImageHDU(master_flatmap),
                fits.ImageHDU(master_flatnorm),
                ])
    hdu_lst.writeto(flat_file, overwrite=True)

    #################### Extract Flat Fielding Spectrum ########################

    # correct flat for flatfielding
    flat_spec_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        data = flat_norm_lst[fiber]/master_flatmap
        mask = flat_mask_lst[fiber]

        fits.writeto('flat_flt.{}.fits'.format(fiber), data, overwrite=True)

        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                        apertureset = master_aperset[fiber],
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )
        flat_spec_lst[fiber] = spectra1d

    ############################## Extract ThAr ################################

    # get the data shape
    h, w = flat_map.shape

    # define dtype of 1-d spectra for all fibers
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux',       (np.float32, w)),
            ('flat',       (np.float32, w)),
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
        exptime = logitem['exptime']

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
        fileid = logitem['fileid']
        print('Wavelength Calibration for {}'.format(fileid))

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)

        # correct overscan for ThAr
        data, card_lst, overmean = correct_overscan(data, mask)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']
        for key, value in card_lst:
            head.append((key, value))

        # correct bias for ThAr, if has bias
        if has_bias:
            data = data - bias
            logger.info('Bias corrected')
        else:
            logger.info('No bias. skipped bias correction')

        # initialize data for all fibers
        all_spec      = {}
        all_cards     = {}
        all_identlist = {}

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
            head = master_aperset[fiber].to_fitsheader(head, fiber=fiber)

            # pack to a structured array
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                spec.append((
                    aper,          # aperture
                    0,             # order (not determined yet)
                    flux_sum.size, # number of points
                    np.zeros_like(flux_sum, dtype=np.float64), # wavelengths (0)
                    flux_sum,      # fluxes
                    flat_spec_lst[fiber][aper]['flux_sum'],  # flat
                    ))
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
                    logger.info(message)

                    ref_spec, ref_calib = select_calib_from_database(
                        database_path, statime_key, head[statime_key])

                    if ref_spec is None or ref_calib is None:

                        message = ('Did not find any archive wavelength'
                                   'calibration file')
                        logger.info(message)

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
                        aperture_k = ((-1, 1)[direction[1]==ref_direction[1]],
                                        None)[direction[1]=='?']
                        pixel_k = ((-1, 1)[direction[2]==ref_direction[2]],
                                    None)[direction[2]=='?']
                        # determine the name of the output figure during lamp
                        # shift finding.
                        fig_ccf = {'normal': None,
                                    'debug': os.path.join(report,
                                        'lamp_ccf_{:+2d}_{:+03d}.png')}[mode]
                        fig_scatter = {'normal': None,
                                        'debug': os.path.join(report,
                                            'lamp_ccf_scatter.png')}[mode]

                        result = find_caliblamp_offset(ref_spec, spec,
                                    aperture_k  = aperture_k,
                                    pixel_k     = pixel_k,
                                    fig_ccf     = fig_ccf,
                                    fig_scatter = fig_scatter,
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
                        print(message)
                        logger.info(message)

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
                    logger.info(message)

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

            # append all spec, card list and ident lists
            all_spec[fiber]      = spec
            all_cards[fiber]     = card_lst
            all_identlist[fiber] = identlist

            # save calib results and the oned spec for this fiber
            head_fiber = head.copy()
            for key,value in card_lst:
                key = 'HIERARCH GAMSE WLCALIB '+key
                head_fiber.append((key, value))
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head_fiber),
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
        # combine different fibers
        # combine cards for FITS header
        newcards = combine_fiber_cards(all_cards)
        # combine spectra
        newspec = combine_fiber_spec(all_spec)
        # combine ident line list
        newidentlist = combine_fiber_identlist(all_identlist)
        # append cards to fits header
        for key, value in newcards:
            key = 'HIERARCH GAMSE WLCALIB '+key
            head.append((key, value))
        # pack and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(newspec),
                    fits.BinTableHDU(newidentlist),
                    ])
        filename = os.path.join(onedspec, '{}_{}.fits'.format(
                                            fileid, oned_suffix))
        hdu_lst.writeto(filename, overwrite=True)

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

        logger.info(
            'FileID: {} ({}) - start reduction: {}'.format(
            fileid, imgtype, filename)
            )

        # read raw data
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)

        # correct overscan
        data, card_lst, overmean = correct_overscan(data, mask)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']
        for key, value in card_lst:
            head.append((key, value))

        message = 'FileID: {} - overscan corrected'.format(fileid)

        logger.info(message)
        print(message)

        # correct bias
        if has_bias:
            data = data - bias
            message = 'FileID: {} - bias corrected. mean value = {}'.format(
                        fileid, bias.mean())
        else:
            message = 'FileID: {} - no bias'.format(fileid)
        logger.info(message)
        print(message)

        # correct flat
        data = data/master_flatmap
        message = 'FileID: {} - flat corrected'.format(fileid)
        logger.info(message)
        print(message)

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

        # generate two figures for each background
        #plot_background_aspect1_alt(data+stray, stray,
        #    os.path.join(report, 'bkg_%s_stray1.%s'%(fileid, fig_format)),
        #    os.path.join(report, 'bkg_%s_stray2.%s'%(fileid, fig_format)))

        message = 'FileID: {} - background corrected. max value = {}'.format(
                fileid, stray.max())
        logger.info(message)
        print(message)

        # extract 1d spectrum
        section = config['reduce.extract']
        all_spec  = {}   # use to pack final 1d spectrum
        all_cards = {}
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
                        flux_sum,
                        flat_spec_lst[fiber][aper]['flux_sum'],
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

            spec, card_lst = reference_wavelength(
                                spec,
                                ref_calib_lst[fiber],
                                weight_lst,
                                )
            all_spec[fiber] = spec
            all_cards[fiber] = card_lst

        newcards = combine_fiber_cards(all_cards)
        newspec = combine_fiber_spec(all_spec)
        for key, value in newcards:
            key = 'HIERARCH GAMSE WLCALIB '+key
            head.append((key, value))
        # pack and save to fits
        hdu_lst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(newspec),
                    ])
        filename = os.path.join(onedspec, '{}_{}.fits'.format(
                                            fileid, oned_suffix))
        hdu_lst.writeto(filename, overwrite=True)

        message = 'FileID: {} - Spectra written to {}'.format(
                    fileid, filename)
        logger.info(message)
        print(message)
