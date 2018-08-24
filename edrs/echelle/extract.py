import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from .imageproc      import table_to_array

def sum_extract(infilename, mskfilename, outfilename, channels, apertureset_lst,
    upper_limit=5, lower_limit=5, figure=None):
    '''Extract spectra from an individual image.
    
    Args:
        infilename (string): Name of the input image.
        outfilename (string): Name of the output image.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of :class:`ApertureSet` instances for
            different channels.
        upper_limit (float): Upper limit of the extracted aperture.
        lower_limit (float): Lower limit of the extracted aperture.
        figure (:class:`matplotlib.figure`): Figure to display the 1d spectra
    Returns:
        No returns
    '''
    data, head = fits.getdata(infilename, header=True)
    h, w = data.shape

    # read data mask
    mask_table = fits.getdata(mskfilename)
    if mask_table.size==0:
        mask = np.zeros_like(data, dtype=np.int16)
    else:
        mask = table_to_array(mask_table, data.shape)
    data_mask = (np.int16(mask) & 4) > 0

    xx, yy = np.meshgrid(np.arange(w),np.arange(h))

    # seperate each type of mask
    #cov_mask = (mdata & 1)>0
    #bad_mask = (mdata & 2)>0
    #sat_mask = (mdata & 4)>0
    
    # define a numpy structured array
    types = [
            ('aperture', np.int32),
            ('channel',  '|1S'),
            ('points',   np.int32),
            ('flux',    '(%d,)float32'%w),
            ('mask',    '(%d,)int16'%w),
            ]
    tmp = list(zip(*types))
    eche_spec = np.dtype({'names':tmp[0], 'formats':tmp[1]})

    spec = []

    newx = np.arange(w)

    # find integration limits
    info_lst = []
    for channel in channels:
        for aper, aperloc in apertureset_lst[channel].items():
            center = aperloc.get_center()
            info_lst.append((center, channel, aper))
    # sort the info_lst
    newinfo_lst = sorted(info_lst, key=lambda item: item[0])

    # find the middle bounds for every adjacent apertures
    lower_bounds = {}
    upper_bounds = {}
    prev_channel  = None
    prev_aper     = None
    prev_position = None
    for item in newinfo_lst:
        channel = item[1]
        aper    = item[2]
        position = apertureset_lst[channel][aper].position(newx)
        if prev_position is not None:
            mid = (position + prev_position)/2.
            lower_bounds[(channel, aper)] = mid
            upper_bounds[(prev_channel, prev_aper)] = mid
        prev_position = position
        prev_channel  = channel
        prev_aper     = aper

    for channel in channels:
        for aper, aper_loc in apertureset_lst[channel].items():
            position = aper_loc.position(newx)
            # determine the lower and upper limits
            lower_line = position - lower_limit
            upper_line = position + upper_limit
            key = (channel, aper)
            if key in lower_bounds:
                lower_line = np.maximum(lower_line, lower_bounds[key])
            if key in upper_bounds:
                upper_line = np.minimum(upper_line, upper_bounds[key])
            lower_line = np.maximum(lower_line, np.zeros(w)-0.5)
            upper_line = np.minimum(upper_line, np.zeros(w)+h-1-0.5)
            lower_ints = np.int32(np.round(lower_line))
            upper_ints = np.int32(np.round(upper_line))
            m1 = yy > lower_ints
            m2 = yy < upper_ints
            mask = m1*m2
            mask = np.float32(mask)
            # determine the weight in the boundary
            mask[lower_ints, newx] = 1-(lower_line+0.5)%1
            mask[upper_ints, newx] = (upper_line+0.5)%1

            # determine the upper and lower row of summing
            r1 = int(lower_line.min())
            r2 = int(upper_line.max())+1
            mask = mask[r1:r2]

            # summing the data and mask
            fluxdata = (data[r1:r2,]*mask).sum(axis=0)
            sat_flux = (data_mask[r1:r2,]*mask).sum(axis=0)>0

            fluxmask = np.int16(sat_flux*4)
            item = np.array((aper, channel, fluxdata.size, fluxdata, fluxmask),
                    dtype=eche_spec)
            spec.append(item)

            # update header. Put coefficients of aperture locations into header.
            leading_string = 'HIERARCH EDRS TRACE CHANNEL %s APERTURE %d'%(
                    channel, aper)
            for ic, c in enumerate(aper_loc.position.coef):
                head[leading_string + ' COEFF %d'%ic] = c


    spec = np.array(spec, dtype=eche_spec)

    pri_hdu = fits.PrimaryHDU(header=head)
    tbl_hdu = fits.BinTableHDU(spec)
    hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
    hdu_lst.writeto(outfilename, overwrite=True)
    logger.info('Write 1D spectra file "%s"'%outfilename)

def extract_aperset(data, mask, apertureset, lower_limit=5, upper_limit=5):
    '''Extract 1-D spectra from the input image data following the input
    :class:`ApertureSet`.

    Args:
        data (:class:`numpy.array`): Input data image.
        mask (:class:`numpy.array`): Input mask.
        apertureset (:class:`ApertureSet`): Input :class:`ApertureSet` instance.
        lower_limit (float): Lower limit of the extracted aperture.
        upper_limit (float): Upper limit of the extracted aperture.
    Returns:
        dict: A dict of 1-d spectra with the aperture numbers as keys, and a
            dict of ("flux_sum", "flux_mean", "mask_sat") as values.
        
    '''
    h, w = data.shape

    # find saturation mask and bad pixel mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    yy, xx = np.mgrid[:h:,:w:]
    spectra1d = {}
    for aper, aper_loc in sorted(apertureset.items()):
        spectra1d[aper] = {}
        domain = aper_loc.position.domain
        d1, d2 = int(domain[0]), int(domain[1])+1
        newx = np.arange(d1, d2)
        position = aper_loc.position(newx)
        lower_line = position - lower_limit
        upper_line = position + upper_limit
        lower_line = np.maximum(lower_line, -0.5)
        lower_line = np.minimum(lower_line, h-1-0.5)
        upper_line = np.maximum(upper_line, -0.5)
        upper_line = np.minimum(upper_line, h-1-0.5)
        lower_ints = np.int32(np.round(lower_line))
        upper_ints = np.int32(np.round(upper_line))
        m1 = yy[:,d1:d2] > lower_ints
        m2 = yy[:,d1:d2] < upper_ints
        mask = np.zeros_like(data, dtype=np.bool)
        mask[:,d1:d2] = m1*m2
        mask = np.float32(mask)
        # determine the weight in the boundary
        mask[lower_ints, newx] = 1-(lower_line+0.5)%1
        mask[upper_ints, newx] = (upper_line+0.5)%1

        ## determine the upper and lower row of summing
        r1 = int(lower_line.min())
        r2 = int(upper_line.max())+1

        # summing the data and mask
        weight_sum = mask[r1:r2].sum(axis=0)
        # summing the flux
        fluxsum = (data[r1:r2]*mask[r1:r2]).sum(axis=0)
        # calculate mean flux
        # filter the zero values
        _m = weight_sum>0
        fluxmean = np.zeros_like(fluxsum)
        fluxmean[_m] = fluxsum[_m]/weight_sum[_m]
        spectra1d[aper]['flux_sum']  = fluxsum
        spectra1d[aper]['flux_mean'] = fluxmean

        # summing the masks
        fluxsat = (sat_mask[r1:r2]*mask[r1:r2]).sum(axis=0)>0
        spectra1d[aper]['mask_sat'] = fluxsat

    return spectra1d
