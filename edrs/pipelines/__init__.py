import os
import astropy.io.fits as fits

from .reduction import Reduction
from ..utils.config import read_config, find_config

instrument_lst =  ['FOCES', 'Xinglong216HRS']

def reduce_echelle(instrument):
    '''Automatically select the instrument and reduce echelle spectra
    accordingly.

    Available instruments include:
        
        * *FOCES*: FOCES on 2.2m Fraunhofer Telescope on Wendelstein
          Observatory.
        * *Xinglong216HRS*: HRS on 2.16m telescope in Xinglong Station.

    Args:
        instrument (string): Name of the instrument.
    Returns:
        No returns.

    '''
    if instrument == 'FOCES':
        from .foces import FOCES
        reduction = FOCES()
        reduction.reduce()
    elif instrument == 'XinglongHRS':
        from .xl216hrs import XinglongHRS
        reduction = XinglongHRS()
        reduction.reduce()
    else:
        print('Unknown Instrument: %s'%instrument)


def make_log(instrument, path):
    '''Scan the path to the raw FITS files and generate an observing log.
    
    Args:
        instrument (string): Name of the instrument.
        path (string): Path to the raw FITS files.
    Returns:
        No returns.
    '''
    if instrument == 'FOCES':
        from .foces import make_log
        make_log(path)
    elif instrument == 'Xinglong216HRS':
        from .xl216hrs import make_log
        make_log(path)
    else:
        print('Cannot recognize the instrument name: %s'%instrument)
        exit()

def get_instrument(path):
    '''Find the telescope and instrument by checking the raw FITS files.

    Args:
        path (string): Path to the raw files.
    Returns:
        string: Name of the instrument.
    '''
    fname_lst = [name for name in os.listdir(path) if name[-5:]=='.fits']
    head = fits.getheader(os.path.join(path, fname_lst[0]))
    if 'INSTRUME' in head and head['INSTRUME']=='FOCES':
        return 'FOCES'
    else:
        print('Cannot recognize the instrument')
        return ''

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
        if config.has_option('reduction', 'path.data'):
            return config.get_option('reduction', 'path.data')
        else:
            return None
