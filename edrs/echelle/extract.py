import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

def sum_extract(infilename, mskfilename, outfilename, channels, apertureset_lst,
    upper_limit=5, lowr_limit=5, figure=None):
    '''Extract spectra from an individual image.
    
    Args:
        infilename (string): Name of the input image.
        outfilename (string): Name of the output image.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of ApertureSet at different channels.
        upper_limit (float): Upper limit of the extracted aperture.
        lower_limit (float): Lower limit of the extracted aperture.
        figure (:class:`matplotlib.figure`): Figure to display the 1d spectra
    Returns:
        No returns
    '''

    data, head = fits.getdata(infilename, header=True)
    mdata = np.int16(fits.getdata(mskfilename))
    h, w = data.shape
    xx, yy = np.meshgrid(np.arange(w),np.arange(h))

    # seperate each type of mask
    #cov_mask = (mdata & 1)>0
    #bad_mask = (mdata & 2)>0
    sat_mask = (mdata & 4)>0
    
    # define a numpy structured array
    types = [
            ('aperture', np.int32),
            ('channel',  '|1S'),
            ('points',   np.int32),
            ('flux',    '(%d,)float32'%w),
            ('mask',    '(%d,)int16'%w),
            ]
    tmp = zip(*types)
    eche_spec = np.dtype({'names':tmp[0], 'formats':tmp[1]})

    spec = []
    for o, location in enumerate(order_lst):
        xdata = location['x']
        ydata = location['y']

        m1 = yy > ydata - lower_limit
        m2 = yy < ydata + upper_limit
        mask = m1*m2

        fluxdata = (data*mask).sum(axis=0)
        sat_flux = (sat_mask*mask).sum(axis=0)>0

        fluxmask = np.int16(sat_flux*4)
        item = np.array((o,fluxdata.size,fluxdata,fluxmask),dtype=eche_spec)
        spec.append(item)

        if figure != None:
            ax = figure.gca()
            ax.cla()
            ax.plot(np.arange(fluxdata.size),fluxdata,'r-')
            ax.set_xlim(0, fluxdata.size-1)
            ax.set_title('%s: Order %d'%(os.path.basename(infilename),o))
            figure.canvas.draw()

    spec = np.array(spec, dtype=eche_spec)

    pri_hdu = fits.PrimaryHDU(header=head)
    tbl_hdu = fits.BinTableHDU(spec)
    hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
    if os.path.exists(outfilename):
        os.remove(outfilename)
        logger.warning('File "%s" is overwritten'%outfilename)
    hdu_lst.writeto(outfilename)
    logger.info('Write 1D spectra file "%s"'%outfilename)
