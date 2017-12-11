import os
import astropy.io.fits as fits

from .reduction import Reduction

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
        pass

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
    if head['INSTRUME']=='FOCES':
        return 'FOCES'
    else:
        print('Cannot recognize the instrument')
        return ''
