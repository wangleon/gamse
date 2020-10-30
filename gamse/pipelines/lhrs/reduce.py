import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

from ...echelle.imageproc import combine_images
from ...echelle.trace import find_apertures, load_aperture_set
from ..common import load_obslog, load_config
from .common import print_wrapper, correct_overscan, get_mask, TraceFigure

def reduce_rawdata():
    """Reduc the LAMOST-HRS data.
    """

    # read obslog and config
    config = load_config('Levy\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')

    # extract keywords from config file
    section = config['data']
    rawpath     = section.get('rawpath')
    statime_key = 'DATE-OBS'
    exptime_key = 'EXPOSURE'

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

