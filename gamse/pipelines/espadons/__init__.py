import os
import re
import sys
import datetime
import dateutil.parser
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

def make_config():
    pass

def make_obslog():
    pass


def reduce_rawdata():
    """2D to 1D pipeline for the CFHT/ESPaDOnS.
    """

    # read obslog and config
    config = load_config('ESPaDOnS\S*\.cfg$')
    logtable = load_obslog('\S*\.obslog$', fmt='astropy')
