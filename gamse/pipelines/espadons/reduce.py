import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from .common import correct_overscan

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


    # parse bias

    flat_filename = os.path.join(midpath,
            )
    if mode='debug' and os.path.exists(
    flat_item_lst = [logitem for logitem in logtable
                            if logitem['obstype']=='FLAT']
    for logitem in flat_item_lst:
        filename = os.path.join(midpath, 
