import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from .imageproc import table_to_array

def sum_extract(infilename, mskfilename, outfilename, channels, apertureset_lst,
    upper_limit=5, lower_limit=5, figure=None):
    """Extract 1-D spectra from an individual image.
    
    Args:
        infilename (str): Name of the input image.
        outfilename (str): Name of the output image.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of :class:`~gamse.echelle.trace.ApertureSet`
            instances for different channels.
        upper_limit (float): Upper limit of the extracted aperture.
        lower_limit (float): Lower limit of the extracted aperture.
        figure (:class:`matplotlib.figure.Figure`): Figure to display the
            extracted 1d spectra.
    """
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
    """Extract 1-D spectra from the input image data following the input
    :class:`~edrs.echelle.trace.ApertureSet`.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Input mask.
        apertureset (:class:`~edrs.echelle.trace.ApertureSet`): Input
            :class:`~edrs.echelle.trace.ApertureSet` instance.
        lower_limit (float): Lower limit of the extracted aperture.
        upper_limit (float): Upper limit of the extracted aperture.

    Returns:
        dict: A dict of 1-d spectra with the aperture numbers as keys, and a
            dict of ("flux_sum", "flux_mean", "mask_sat") as values.
        
    """
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

def optimal_extract(data, mask, apertureset):
    """Optimal extraction
    """

    daper = 10
    aper1_lst = np.arange(min(apertureset), max(apertureset), daper)
    aper2_lst = aper1_lst + daper
    if len(apertureset)%daper < daper/2:
        aper1_lst = np.delete(aper1_lst, -1)
        aper2_lst = np.delete(aper2_lst, -1)
    else:
        pass
    aper2_lst[-1] = max(apertureset)+1

    profy_lst = np.arange(-10, 10+1e-3, 0.5)
    h, w = data.shape

    for loop in range(2):
        apercen_lst = []
        profilesamp_lst = []
        for iregion, (aper1, aper2) in enumerate(zip(aper1_lst, aper2_lst)):
            print(aper1, aper2)
            apernode_x_lst = []
            apernode_y_lst = []
            for iaper, aper in enumerate(np.arange(aper1, aper2)):
                aperloc = apertureset[aper]
                ycen_lst = aperloc.position(np.arange(w))

                node_x_lst = []
                node_y_lst = []
                if loop > 0:
                    profile = allprofile_lst[aper]
                    interf = intp.InterpolatedUnivariateSpline(
                                profy_lst, profile, k=3, ext=1)

                for x in np.arange(w//2-200, w//2+200):
                    ycen = ycen_lst[x]
                    yceni = np.int(np.round(ycen))
                    yrows = np.arange(yceni-10, yceni+10+1)
                    flux = data[yrows, x]
                    negative_mask = flux<0
                    if negative_mask.sum()>0.5*flux.size:
                        continue

                    if loop == 0:
                        A = flux.sum()
                    else:
                        para = [flux.sum(),ycen]
                        mask = np.ones_like(flux, dtype=np.bool)
                        for ite in range(10):
                            result = opt.least_squares(errfunc, para,
                                        bounds=((-np.inf,ycen-2),(np.inf,ycen+2)),
                                        args=(flux[mask], interf, yrows[mask]))
                            newpara = result['x']
                            pro = fitfunc(newpara, interf, yrows)
                            res = flux - pro
                            std = res[mask].std()
                            new_mask = (res < 3*std)*(res > -3*std)
                            if new_mask.sum() == mask.sum():
                                break
                            mask = new_mask
                            para = newpara
                        A, ycen = newpara

                    if A<0:
                        continue
                    normflux = flux/A
                    for v in yrows-ycen:
                        node_x_lst.append(v)
                        apernode_x_lst.append(v)
                    for v in normflux:
                        node_y_lst.append(v)
                        apernode_y_lst.append(v)
                ### loop for x pixel ends here
            apernode_x_lst = np.array(apernode_x_lst)
            apernode_y_lst = np.array(apernode_y_lst)
            profile, profile_std, profile_snr = get_mean_profile(
                    apernode_x_lst, apernode_y_lst, profy_lst)
            profile_cen = (profy_lst*profile).sum()/profile.sum()
            func = intp.InterpolatedUnivariateSpline(
                    profy_lst-profile_cen, profile, k=3, ext=3)
            newprofile = func(profy_lst)
            profilesamp_lst.append(newprofile)
            apercen_lst.append((aper1+aper2)/2)

            # plot figure for this aper group
            fig1 = plt.figure(dpi=150, figsize=(12,8))
            ax1 = fig1.gca()
            ax1.scatter(apernode_x_lst, apernode_y_lst, c='gray', s=1, alpha=0.1)
            ax1.plot(profy_lst, profile, '-', lw=1, c='C1')
            ax1.plot(profy_lst, profile+profile_std, '--', lw=0.5, c='C1')
            ax1.plot(profy_lst, profile-profile_std, '--', lw=0.5, c='C1')
            ax1.set_xlim(-11, 11)
            ax1.set_ylim(-0.02, 0.13)
            ax1.axvline(x=0, color='k', ls='--', lw=1)
            ax1.axhline(y=0, color='k', ls='--', lw=1)
            ax1.grid(True, ls=':', color='k')
            fig1.savefig('img2/apergroup_%02d_%02d_loop%d.png'%(aper1, aper2, loop))
            plt.close(fig1)

        ### loop for aper region ends here
        # build interp functions for all apertures
        profilesamp_lst = np.array(profilesamp_lst)

        # get profiles for all apertures
        allprofile_lst = {}
        for aper in sorted(apertureset):
            profile = []
            for col in np.arange(profy_lst.size):
                func = intp.InterpolatedUnivariateSpline(
                        apercen_lst, profilesamp_lst[:,col], k=3, ext=0)
                profile.append(func(aper))
            allprofile_lst[aper] = np.array(profile)

        # plot interpolated profiles
        fig = plt.figure(dpi=150)
        ax = fig.gca()
        for aper in sorted(apertureset):
            ax.plot(profy_lst, allprofile_lst[aper], alpha=0.6, lw=0.5)
        ax.set_xlim(-11,11)
        ax.set_ylim(-0.01, 0.12)
        ax.grid(True, color='k', ls=':', lw=0.5)
        fig.savefig('img2/intp_profiles_loop%d.png'%loop)
        plt.close(fig)
    # profile loop ends here


