import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ..common import load_obslog, load_config
from .common import parse_image

def reduce_rawdata():
    """Reduce the Subaru/HDS spectra.
    """
    # read obslog and config
    config = load_config('HDS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

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

    ############ count different setups #############
    setup_lst = {}
    for logitem in logtable:
        setup   = logitem['setup']
        objtype = logitem['objtype']
        binning = logitem['binning']
        if (setup, binning) not in setup_lst:
            setup_lst[(setup, binning)] = {}
        if objtype not in setup_lst[(setup, binning)]:
            setup_lst[(setup, binning)][objtype] = 0
        setup_lst[(setup, binning)][objtype] += 1

    object_setup_lst = []
    for (setup, binning), objtype_lst in sorted(setup_lst.items()):
        print('Setup: {} Binning: {}'.format(setup, binning))
        count_total = 0
        for objtype, count in sorted(objtype_lst.items()):
            print(' - {:10s}: {:3d} Frames'.format(objtype, count))
            count_total += count
            if objtype=='OBJECT':
                object_setup_lst.append((setup, binning))
        print(' - {:10s}: {:3d} Frames'.format('Total', count_total))
    object_setup_lst = list(set(object_setup_lst))


    # loop over different setups and binnings
    for sel_setup, sel_binning in object_setup_lst:
        print('Selected setup={}; selected binning={}'.format(
            sel_setup, sel_binning))

        ############### parse bias #################
        bias_filter = lambda item: item['setup']==sel_setup \
                        and item['binning']==sel_binning \
                        and item['objtype']=='BIAS' \
                        and item['object']=='BIAS' \
                        and item['nsat_1']<100 \
                        and item['q95_1']<10000

        bias_file = config['reduce.bias'].get('bias_file')

        if mode=='debug' and os.path.exists(bias_file):
            pass

        else:
            bias_data_lst1 = []
            bias_data_lst2 = []
            bias_card_lst = []

            logitem_lst = list(filter(bias_filter, logtable))

            # get the number of bias images
            n_bias = len(logitem_lst)

            if n_bias == 0:
                pass

            fmt_str = ('  - {:>5s} {:12s} {:12s} {:<7s} {:<7s} {:1s}I2 {:>7}'
                    ' {:<7s} {:5}' # setup, binning
                    ' {:>7} {:>7} {:>5} {:>5}' # nsat_1, nsat_2, q95_1, q95_2
                    )
            head_str = fmt_str.format('FID', 'fileid1', 'fileid2', 'objtype',
                        'object', '', 'exptime', 'setup', 'binning',
                        'nsat_1', 'nsat_2', 'q95_1',  'q95_2')
            print(head_str)
            for ifile, logitem in enumerate(logitem_lst):
                fname1 = '{}.fits'.format(logitem['fileid1'])
                fname2 = '{}.fits'.format(logitem['fileid2'])
                filename1 = os.path.join(rawpath, fname1)
                filename2 = os.path.join(rawpath, fname2)
                data1, head1 = fits.getdata(filename1, header=True)
                data2, head2 = fits.getdata(filename2, header=True)
                data1 = parse_image(data1, head1)
                data2 = parse_image(data2, head2)
        
                string = fmt_str.format('[{:d}]'.format(logitem['frameid']),
                            logitem['fileid1'], logitem['fileid2'],
                            logitem['objtype'], logitem['object'],
                            logitem['i2'], logitem['exptime'],
                            logitem['setup'], logitem['binning'],
                            logitem['nsat_1'], logitem['nsat_2'],
                            logitem['q95_1'], logitem['q95_2'])
                print(print_wrapper(string, logitem))
        
                bias_data_lst1.append(data1)
                bias_data_lst2.append(data2)
        
                # append the file information
                prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ifile+1)
                card = (prefix+' FILEID1', logitem['fileid1'])
                bias_card_lst.append(card)
                card = (prefix+' FILEID2', logitem['fileid2'])
                bias_card_lst.append(card)

            prefix = 'HIERARCH GAMSE BIAS '
            bias_card_lst.append((prefix + 'NFILE', n_bias))
     
            # combine bias images
            bias_data_lst1 = np.array(bias_data_lst1)
            bias_data_lst2 = np.array(bias_data_lst2)
     
            combine_mode = 'mean'
            cosmic_clip  = section.getfloat('cosmic_clip')
            maxiter      = section.getint('maxiter')
            maskmode     = (None, 'max')[n_bias>=3]
     
            bias_combine1 = combine_images(bias_data_lst1,
                    mode        = combine_mode,
                    upper_clip  = cosmic_clip,
                    maxiter     = maxiter,
                    maskmode    = maskmode,
                    ncores      = ncores,
                    )
            bias_combine2 = combine_images(bias_data_lst2,
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
            # pack new card list into header and bias_card_lst
            for card in bias_card_lst:
                head.append(card)
            head['HIERARCH GAMSE FILECONTENT 0'] = 'BIAS COMBINED'
            hdu_lst.append(fits.PrimaryHDU(data=bias_combine1, header=head))
            hdu_lst.append(fits.ImageHDU(data=bias_combine2, header=head))
     
            # write to FITS file
            hdu_lst.writeto(bias_file, overwrite=True)
     
            message = 'Bias image written to "{}"'.format(bias_file)
            logger.info(message)
            print(message)

        ############### find flat groups #################

        flat_file_str = config['reduce.flat'].get('flat_file')
        flat_file = flat_file.format(sel_setup, sel_binning)

        if mode=='debug' and os.path.exists(flat_file):
            continue
            # pass
        else:
            filterfunc = lambda item: item['setup']==sel_setup \
                            and item['binning']==sel_binning \
                            and item['objtype']=='FLAT' \
                            and item['object']=='FLAT'
            logitem_lst = list(filter(filterfunc, logtable))

            fmt_str = ('  - {:>5s} {:12s} {:12s} {:<7s} {:<7s} {:1s}I2 {:>7}'
                    ' {:<7s} {:5} {:8}' # setup, binning, slitsize
                    ' {:>7} {:>7} {:>5} {:>5}' # nsat_1, nsat_2, q95_1, q95_2
                    )
            head_str = fmt_str.format('FID', 'fileid1', 'fileid2',
                        'objtype', 'object', '', 'exptime',
                        'setup', 'binning', 'slitsize',
                        'nsat_1', 'nsat_2', 'q95_1',  'q95_2')
            
            for logitem in logtable:
                objtype = logitem['objtype']
                objname = logitem['object']
