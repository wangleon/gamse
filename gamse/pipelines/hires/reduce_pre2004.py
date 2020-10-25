import os

import numpy as np
import astropy.io.fits as fits

def reduce_pre2004(config, logtable):
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
