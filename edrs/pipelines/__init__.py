import os
import logging
logger = logging.getLogger(__name__)

import astropy.io.fits as fits

from .reduction import Reduction
from .plot import plot_spectra1d
from ..utils.config import read_config, find_config
from . import foces, xinglong216hrs, levy

instrument_lst =  ['FOCES', 'Xinglong216HRS']

def reduce_echelle():
    '''Automatically select the instrument and reduce echelle spectra
    accordingly.

    Available instruments include:
        
        * *FOCES*: FOCES on 2m Fraunhofer Telescope in Wendelstein Observatory,
            Germany.
        * *Xinglong216HRS*: HRS on 2.16m telescope in Xinglong Station, China.

    Args:
        No args.
    Returns:
        No returns.

    '''

    key = get_instrument()

    if key == ('Fraunhofer', 'FOCES'):
        reduction = foces.FOCES()
    elif key == ('Xinglong2.16m', 'HRS'):
        reduction = xinglong216hrs.Xinglong216HRS()
    elif key == ('APF', 'Levy'):
        levy.reduce()
    else:
        print('Unknown Instrument: %s, %s'%(key[0], key[1]))
        exit()

    logger.info('Start reducing %s, %s data'%(key[0], key[1]))
    reduction.reduce()


def plot():
    '''Plot the 1-D spectra.

    Args:
        No args.
    Returns:
        No returns.
    '''
    plot_spectra1d()

def make_log():
    '''Scan the path to the raw FITS files and generate an observing log.
    
    Args:
        No args
    Returns:
        No returns.
    '''
    config_file = find_config('./')
    config = read_config(config_file)
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']
    rawdata    = section['rawdata']
    key = get_instrument()

    if key == ('Fraunhofer', 'FOCES'):
        foces.make_log(rawdata)
    elif key == ('Xinglong2.16m', 'HRS'):
        xinglong216hrs.make_log(rawdata)
    elif key == ('APF', 'Levy'):
        levy.make_log(rawdata)
    else:
        print('Unknown Instrument: %s, %s'%(telescope, instrument))
        exit()

def get_instrument():
    '''Find the telescope and instrument by checking the raw FITS files.

    Args:
        No args
    Returns:
        string: Name of the instrument.
    '''
    config_file = find_config('./')
    config = read_config(config_file)
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']
    rawdata    = section['rawdata']
    return telescope, instrument


def find_rawdata():
    '''Find the path to the raw images.

    Args:
        No args.
    Returns:
        string or None: Path to the raw images. Return *None* if path not found.
    '''

    if os.path.exists('rawdata'):
        return 'rawdata'
    else:
        config_file = find_config('./')
        config = read_config(config_file)
        if config.has_section('path') and \
           config.has_option('path', 'rawdata'):
            return config.get_option('path', 'rawdata')
        else:
            return None
