import os
import re
import shutil
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from .common import correct_overscan
from .trace import find_apertures

def reduce_data(config, logtable):
    """Reduce the single fiber data of ESPRESSO.

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
    readout_mode = section.get('readout_mode')

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

    # initialize general card list
    general_card_lst = {}

    # determine mode and binning
    instmode = logtable[0]['mode']
    binning = logtable[0]['binning']
    mobj = re.match('\((\d), (\d)\)', binning)
    binx = mobj.group(1)
    biny = mobj.group(2)
    instconfig = '{}_{}x{}'.format(instmode, binx, biny)

    ############################# parse bias ###################################

    bias_file = {channel: os.path.join(midpath,
                    'bias_{}_{}.fits'.format(instconfig, channel))
                    for channel in ['b', 'r']}
    bias_lst = {}
    if mode=='debug' and \
        os.path.exists(bias_file['b']) and os.path.exists(bias_file['r']):

        for channel in ['b', 'r']:
            filename = bias_file[channel]
            bias_lst[channel] = fits.getdata(filename)
            message = 'read bias from {}'.format(filename)
            print(message)
    else:
        bias_data_lst = {'b': [], 'r': []}
        bias_card_lst = []
        bias_items = list(filter(
                    lambda item: item['imgtype']=='BIAS' and \
                                 item['object']=='BIAS',
                    logtable))
        # get number of bias images
        n_bias = len(bias_items)

        if n_bias == 0:
            pass

        fmt_str = '  - {:>7s} {:^11} {:^8s} {:^7} {:^19s}'
        head_str = fmt_str.format('ExpID', 'FileID', 'Object', 'exptime',
                        'obsdate')
        for iframe, logitem in enumerate(bias_items):
            expid      = logitem['expid']
            fileid     = logitem['fileid']
            objectname = logitem['object']
            exptime    = logitem['exptime']
            obsdate    = logitem['obsdate']

            fname = '{}.fits'.format(fileid)
            filename = os.path.join(rawpath, fname)
            hdulst = fits.open(filename)
            head_b = hdulst[1].header
            data_b = hdulst[1].data
            head_r = hdulst[2].header
            data_r = hdulst[2].data
            hdulst.close()

            data_b = correct_overscan(data_b, head_b)
            data_r = correct_overscan(data_r, head_r)

            #fits.writeto('{}_ovr_b.fits'.format(fileid), data_b, overwrite=True)
            #fits.writeto('{}_ovr_r.fits'.format(fileid), data_r, overwrite=True)
            bias_data_lst['b'].append(data_b)
            bias_data_lst['r'].append(data_r)

            if iframe == 0:
                print('Combine Bias Images')
                print(head_str)
            message = fmt_str.format('[{:d}]'.format(expid),
                    fileid, objectname, exptime, obsdate)
            print(message)

        # combine bias images
        bias_data_lst['b'] = np.array(bias_data_lst['b'])
        bias_data_lst['r'] = np.array(bias_data_lst['r'])

        combine_mode = 'mean'
        section = config['reduce.bias']
        cosmic_clip  = section.getfloat('cosmic_clip')
        maxiter      = section.getint('maxiter')
        maskmode    = (None, 'max')[n_bias>=3]


        for channel in ['b', 'r']:
            bias_combine = combine_images(
                    bias_data_lst[channel],
                    mode        = combine_mode,
                    upper_clip  = cosmic_clip,
                    maxiter     = maxiter,
                    maskmode    = maskmode,
                    ncores      = ncores,
                    )
            
            filename = bias_file[channel]
            # save to fits
            fits.writeto(filename, bias_combine, overwrite=True)

            ####### smooth bias #####
            if section.getboolean('smooth'):
                bias_lst[channel] = bias_combine
            else:
                bias_lst[channel] = bias_combine

            message = 'Bias of {} channel written to {}'.format(
                        channel, filename)


    ##### order trace #######
    trace_item_lst = {'A': list(filter(lambda item:
                        item['object']=='ORDERDEF,LAMP,OFF', logtable)),
                      'B': list(filter(lambda item:
                        item['object']=='ORDERDEF,OFF,LAMP', logtable)),
                    }
    for fiber in ['A', 'B']:
        logitem = trace_item_lst[fiber][0]
        fileid = logitem['fileid']

        fname = '{}.fits'.format(fileid)
        filename = os.path.join(rawpath, fname)
        hdulst = fits.open(filename)
        head_lst = {'b': hdulst[1].header, 'r': hdulst[2].header}
        data_lst = {'b': hdulst[1].data,   'r': hdulst[2].data}
        hdulst.close()

        for channel in ['b', 'r']:
            data = correct_overscan(data_lst[channel], head_lst[channel])
            data = data - bias_lst[channel]
            fname = 'trace_{}_{}_{}.fits'.format(
                        instconfig, fiber, channel)
            filename = os.path.join(midpath, fname)
            fits.writeto(filename, data, overwrite=True)
