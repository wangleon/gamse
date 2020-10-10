import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ..common import load_obslog, load_config
from .common import get_bias

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
        bin_1   = eval(logitem['bin_1'])
        bin_2   = eval(logitem['bin_2'])
        binning = (bin_1, bin_2)
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

    if len(object_setup_lst)==1:
        sel_setup, sel_binning = object_setup_lst[0]
    print('Selected setup: {} binning: {}'.format(
            sel_setup, sel_binning))
    ############### parse bias #################
    filterfunc = lambda item: item['setup']==sel_setup \
                    and item['bin_1']==sel_binning[0] \
                    and item['bin_2']==sel_binning[1]
    get_bias(config, logtable, filterfunc)
