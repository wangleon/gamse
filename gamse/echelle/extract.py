import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from .imageproc import table_to_array
from ..utils.onedarray import pairwise, smooth
from ..utils.regression import get_clip_mean

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

def extract_aperset(data, mask, apertureset, lower_limit=5, upper_limit=5, variance=False):
    """Extract 1-D spectra from the input image data following the input
    :class:`~gamse.echelle.trace.ApertureSet`.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Input mask.
        apertureset (:class:`~gamse.echelle.trace.ApertureSet`): Input
            :class:`~gamse.echelle.trace.ApertureSet` instance.
        lower_limit (float): Lower limit of the extracted aperture.
        upper_limit (float): Upper limit of the extracted aperture.
        variance (bool)    : If a variance array is processed the weights 
                             need to be squared

    Returns:
        dict: A dict of 1-d spectra with the aperture numbers as keys, and a
            dict of ("flux_sum", "flux_mean", "mask_sat") as values.
        
    """
    h, w = data.shape

    # find saturation mask and bad pixel mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)
    gap_mask = (mask&1 > 0)

    yy, xx = np.mgrid[:h:,:w:]
    spectra1d = {}
    for aper, aper_loc in sorted(apertureset.items()):
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
        newmask = np.zeros_like(data, dtype=np.bool)
        newmask[:,d1:d2] = m1*m2
        newmask = np.float32(newmask)
        # determine the weight in the boundary
        if variance:
            newmask[lower_ints, newx] = (1-(lower_line+0.5)%1)**2
            newmask[upper_ints, newx] = ((upper_line+0.5)%1)**2
        else:
            newmask[lower_ints, newx] = 1-(lower_line+0.5)%1
            newmask[upper_ints, newx] = (upper_line+0.5)%1
        # filter the bad, saturated, and gap pixels
        newmask = newmask*(~sat_mask)
        newmask = newmask*(~bad_mask)
        newmask = newmask*(~gap_mask)

        ## determine the upper and lower row of summing
        r1 = int(lower_line.min())
        r2 = int(upper_line.max())+1

        # summing the data and mask
        weight_sum = newmask[r1:r2].sum(axis=0)
        # summing the flux
        fluxsum = (data[r1:r2]*newmask[r1:r2]).sum(axis=0)
        # calculate mean flux
        # filter the zero values
        _m = weight_sum>0
        fluxmean = np.zeros_like(fluxsum)
        fluxmean[_m] = fluxsum[_m]/weight_sum[_m]

        spectra1d[aper] = {
                'flux_sum':  fluxsum,
                'flux_mean': fluxmean,
                'mask':      ~_m,
                'nsum':      weight_sum,
                }


        # summing the masks
        fluxsat = (sat_mask[r1:r2]*newmask[r1:r2]).sum(axis=0)>0
        spectra1d[aper]['mask_sat'] = fluxsat

    return spectra1d

def get_mean_profile(nodex_lst, nodey_lst, profx_lst):
    """Calculate the mean profiles for a series of (*x*, *y*) data.

    Args:
        nodex_lst (:class:`numpy.ndarray`): Input *x* data.
        nodey_lst (:class:`numpy.ndarray`): Input *y* data with the same length
            as **nodex_lst**.
        profx_lst (:class:`numpy.ndarray`): X-coordinates of the mean profile.

    Returns:
        A tuple containing:
            **profile** (:class:`numpy.ndarray`): Mean profile.
            **profile_std** (:class:`numpy.ndarray`): Standard deviations of
                mean profile.


    """
    # find middle points
    mid_profx_lst = (profx_lst + np.roll(profx_lst, -1))/2
    mid_profx_lst = np.insert(mid_profx_lst, 0,
                    profx_lst[0]-(profx_lst[1]-profx_lst[0])/2)
    mid_profx_lst[-1] = profx_lst[-1] + (profx_lst[-1] - profx_lst[-2])/2

    # calculate mean profile
    mean_x_lst, mean_y_lst, std_y_lst = [], [], []
    for y1, y2 in pairwise(mid_profx_lst):
        mask = (nodex_lst > y1)*(nodex_lst < y2)
        if mask.sum() > 0:
            meany, std, _ = get_clip_mean(nodey_lst[mask], maxiter=20)
            #xcenter = (nodex_lst[mask]*nodey_lst[mask]).sum()/nodey_lst[mask].sum()
            #mean_x_lst.append(xcenter)
            mean_x_lst.append((y1+y2)/2)
            mean_y_lst.append(meany)
            std_y_lst.append(std)

    # convert to numpy arrays
    mean_x_lst = np.array(mean_x_lst)
    mean_y_lst = np.array(mean_y_lst)
    std_y_lst = np.array(std_y_lst)

    # fill the missing values with cubic interpolation
    if mean_x_lst.size < profx_lst.size:
        f1 = intp.InterpolatedUnivariateSpline(mean_x_lst, mean_y_lst, k=3)
        mean_y_lst = f1(profx_lst)
        f2 = intp.InterpolatedUnivariateSpline(mean_x_lst, std_y_lst, k=3)
        std_y_lst = f2(profx_lst)

    return mean_y_lst, std_y_lst

def optimal_extract(data, mask, apertureset):
    """Optimal extraction.

    Args:
        data (:class:`ndarray`):
        mask (:class:`ndarray`):
        apertureset ():

    Returns:
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

    profx_lst = np.arange(-10, 10+1e-3, 0.5)
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
                                profx_lst, profile, k=3, ext=1)

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

            # now calculate the mean profile
            apernode_x_lst = np.array(apernode_x_lst)
            apernode_y_lst = np.array(apernode_y_lst)
            profile, profile_std = get_mean_profile(
                    apernode_x_lst, apernode_y_lst, profx_lst)

            # smooth the profile
            profile     = smooth(profile,     points=5, deg=3)
            profile_std = smooth(profile_std, points=5, deg=3)

            # calculate the typical S/N of this profile
            snr_lst = profile/profile_std
            ic = profx_lst.size//2
            snr = snr_lst[ic-1:ic+2].mean()

            # calculate the center of mass
            profile_cen = (profx_lst*profile).sum()/profile.sum()

            # align the profile to the center of mass with cubic interpolataion
            func = intp.InterpolatedUnivariateSpline(
                    profx_lst-profile_cen, profile, k=3, ext=3)
            newprofile = func(profx_lst)

            # append the results
            profilesamp_lst.append(newprofile)
            apercen_lst.append((aper1+aper2)/2)

            # plot figure for this aper group
            fig1 = plt.figure(dpi=150, figsize=(12,8))
            ax1 = fig1.gca()
            ax1.scatter(apernode_x_lst, apernode_y_lst, c='gray', s=1, alpha=0.1)
            ax1.plot(profx_lst, profile, '-', lw=1, c='C1')
            ax1.plot(profx_lst, profile+profile_std, '--', lw=0.5, c='C1')
            ax1.plot(profx_lst, profile-profile_std, '--', lw=0.5, c='C1')
            ax1.set_xlim(profx_lst[0]-1, profx_lst[-1]+1)
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
            for col in np.arange(profx_lst.size):
                func = intp.InterpolatedUnivariateSpline(
                        apercen_lst, profilesamp_lst[:,col], k=3, ext=0)
                profile.append(func(aper))
            allprofile_lst[aper] = np.array(profile)

        # plot interpolated profiles
        fig = plt.figure(dpi=150)
        ax = fig.gca()
        for aper in sorted(apertureset):
            ax.plot(profx_lst, allprofile_lst[aper], alpha=0.6, lw=0.5)
        ax.set_xlim(-11,11)
        ax.set_ylim(-0.01, 0.12)
        ax.grid(True, color='k', ls=':', lw=0.5)
        fig.savefig('img2/intp_profiles_loop%d.png'%loop)
        plt.close(fig)
    # profile loop ends here

    ######################################################################
    flux_opt_lst = {}
    flux_sum_lst = {}
    #for aper, aperloc in sorted(aperset.items()):
    for aper in [10, 63]:
        aperloc = aperset[aper]
        print(aper)
        ycen_lst = aperloc.position(np.arange(w))
        profile = allprofile_lst[aper]
        interf = intp.InterpolatedUnivariateSpline(
                    profx_lst, profile, k=3, ext=1)
        flux_opt_lst[aper] = []
        flux_sum_lst[aper] = []
        newycen_lst = []
        for x in np.arange(w):
            ycen = ycen_lst[x]
            yceni = np.int(np.round(ycen))
            yrows = np.arange(yceni-10, yceni+10+1)
            flux = data[yrows, x]
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
            print(x, newpara, mask.size-mask.sum(), ite)
            newycen_lst.append(newpara[1])
            s_lst = 1/(np.maximum((flux+240),0)+1.0**2)
            normpro = pro/pro.sum()
            fopt = ((s_lst*normpro*flux)[mask].sum())/((s_lst*normpro**2)[mask].sum())
            flux_opt_lst[aper].append(fopt)
            flux_sum_lst[aper].append(flux.sum())

            ################################################
            if aper==63:
                if x%30==0:
                    fig1 = plt.figure(figsize=(18,10), dpi=150)
                irow = int((x%30)/6)
                icol = (x%30)%6
                _x = 0.04 + icol*0.16
                _y = 0.05 + (4-irow)*0.19
                ax = fig1.add_axes([_x, _y, 0.14, 0.17])
                ax.plot(yrows-para[1], pro, 'o-', color='w',
                        markeredgecolor='C0',ms=4, lw=0.8)
                ax.plot(yrows[mask]-para[1], pro[mask], 'o', color='C0',
                        ms=4)
                ax.plot(yrows-para[1], pro+1*std, '--', color='C0', lw=0.5)
                ax.plot(yrows-para[1], pro-1*std, '--', color='C0', lw=0.5)
                ax.plot(yrows[mask]-para[1], flux[mask], '-', color='C1', lw=0.8)
                x1, x2 = ax.get_xlim()
                y1, y2 = ax.get_ylim()
                ax.plot(yrows-para[1], flux, '--', color='C1', lw=0.8)
                ax.text(0.95*x1+0.05*x2, 0.1*y1+0.9*y2, 'X=%d'%x, fontsize=9)
                ax.text(0.35*x1+0.65*x2, 0.1*y1+0.9*y2, '%7g'%(flux.sum()), fontsize=9, color='C0')
                ax.text(0.35*x1+0.65*x2, 0.2*y1+0.8*y2, '%7g'%fopt, fontsize=9, color='C1')
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                ax.axhline(y=0, c='k', ls='--', lw=0.5)
                ax.axvline(x=0, c='k', ls='--', lw=0.5)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                if x%30 == 29 or x == w-1:
                    fig1.savefig('img4/fitting-%02d-%04d.png'%(aper,x))
                    plt.close(fig1)
            ################################################



        flux_opt_lst[aper] = np.array(flux_opt_lst[aper])
        flux_sum_lst[aper] = np.array(flux_sum_lst[aper])

        '''
        fig = plt.figure(dpi=150, figsize=(15,10))
        ax = fig.gca()
        ax.plot(flux_opt_lst[aper], ls='-', lw=0.5, color='C1')
        ax.set_xlim(0, w-1)
        fig.savefig('img4/flux_%02d.png'%aper)
        plt.close(fig)
        '''
        newycen_lst = np.array(newycen_lst)
        fig2 = plt.figure(dpi=150, figsize=(12,8))
        ax1 = fig2.add_subplot(211)
        ax2 = fig2.add_subplot(212)
        ax1.plot(ycen_lst, color='C0', ls='-')
        ax1.plot(newycen_lst, color='C1', ls='-')
        ax2.plot(newycen_lst-ycen_lst, color='C1', ls='-')
        fig2.savefig('img4/comp_center_aper%02d.png'%aper)
        plt.close(fig2)

    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux_sum',   (np.float32, w)),
            ('flux_opt',   (np.float32, w)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})
    spec = []
    for aper in sorted(flux_opt_lst):
        flux_sum = flux_sum_lst[aper]
        flux_opt = flux_opt_lst[aper]
        n = flux_sum.size
        spec.append((aper, 0, n,
                    np.zeros(n, dtype=np.float64),
                    flux_sum, flux_opt))
    spec = np.array(spec, dtype=spectype)

