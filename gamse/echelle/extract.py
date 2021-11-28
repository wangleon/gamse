import os
import math
import logging

logger = logging.getLogger(__name__)

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.integrate as intg
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from .imageproc import table_to_array
from ..utils.onedarray import pairwise, smooth
from ..utils.regression import get_clip_mean

def extract_aperset(data, mask, apertureset, lower_limit=5, upper_limit=5,
        variance=False):
    """Extract 1-D spectra from the input image data following the input
    :class:`~gamse.echelle.trace.ApertureSet`.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Input mask.
        apertureset (:class:`~gamse.echelle.trace.ApertureSet`): Input
            :class:`~gamse.echelle.trace.ApertureSet` instance.
        lower_limit (float): Lower limit of the extracted aperture.
        upper_limit (float): Upper limit of the extracted aperture.
        variance (bool): If a variance array is processed the weights 
                             need to be squared

    Returns:
        dict: A dict of 1-d spectra with the aperture numbers as keys, and a
            dict of ("flux_sum", "flux_mean", "mask_sat") as values.
        
    """
    ny, nx = data.shape

    # find saturation mask and bad pixel mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)
    gap_mask = (mask&1 > 0)

    allx = np.arange(nx)
    ally = np.arange(ny)
    yy, xx = np.mgrid[:ny:,:nx:]
    spectra1d = {}
    for aper, aper_loc in sorted(apertureset.items()):
        if aper_loc.direct==0:
            position = aper_loc.position(ally)
            lower_line = position - lower_limit
            upper_line = position + upper_limit
            lower_line = np.maximum(lower_line, -0.5)
            lower_line = np.minimum(lower_line, nx-1-0.5)
            upper_line = np.maximum(upper_line, -0.5)
            upper_line = np.minimum(upper_line, nx-1-0.5)
            lower_ints = np.int32(np.round(lower_line))
            upper_ints = np.int32(np.round(upper_line))
            m1 = xx > lower_ints.reshape((-1,1))
            m2 = xx < upper_ints.reshape((-1,1))
            newmask = np.float32(m1*m2)
            # determine the weight in the boundary
            newmask[ally, lower_ints] = 1-(lower_line+0.5)%1
            newmask[ally, upper_ints] = (upper_line+0.5)%1
            # filter the bad, saturated, and gap pixels
            newmask = newmask*(~sat_mask)
            newmask = newmask*(~bad_mask)
            newmask = newmask*(~gap_mask)
            # determine the left and right column of summing
            c1 = int(lower_line.min())
            c2 = int(np.ceil(upper_line.max()))+1
            # summing the data and mask
            weight_sum = newmask[:,c1:c2].sum(axis=1)
            # summing the flux and save it to fluxsum
            fluxsum = (data[:,c1:c2]*newmask[:,c1:c2]).sum(axis=1)
            # summing the masks
            fluxsat = (sat_mask[:,c1:c2]*newmask[:,c1:c2]).sum(axis=1)>0


        elif aper_loc.direct==1:
            domain = aper_loc.position.domain
            d1, d2 = int(domain[0]), int(domain[1])+1
            newx = np.arange(d1, d2)
            position = aper_loc.position(newx)
            lower_line = position - lower_limit
            upper_line = position + upper_limit
            lower_line = np.maximum(lower_line, -0.5)
            lower_line = np.minimum(lower_line, ny-1-0.5)
            upper_line = np.maximum(upper_line, -0.5)
            upper_line = np.minimum(upper_line, ny-1-0.5)
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
            
            # determine the upper and lower row of summing
            r1 = int(lower_line.min())
            r2 = int(np.ceil(upper_line.max()))+1
            
            # summing the data and mask
            weight_sum = newmask[r1:r2].sum(axis=0)
            # summing the flux and save it to fluxsum
            # method 1: direct sum
            fluxsum = (data[r1:r2]*newmask[r1:r2]).sum(axis=0)
            
            # method2: simpson integration
            #fluxsum = []
            #for x in np.arange(data.shape[1]):
            #    summask = newmask[:,x]>0
            #    samplex = np.arange(data.shape[0])[summask]
            #    if len(samplex)>0:
            #        fluxp = intg.simps(data[:,x][summask], samplex)
            #    else:
            #        fluxp = 0.0
            #    fluxsum.append(fluxp)
            #fluxsum = np.array(fluxsum)

            # summing the masks
            fluxsat = (sat_mask[r1:r2]*newmask[r1:r2]).sum(axis=0)>0

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
            'mask_sat':  fluxsat,
            }
            
    return spectra1d

def extract_aperset_optimal(data, mask, background, apertureset,
        ron, gain, profilex, disp_x_lst):
    """Extract 1-D spectra from the input image using the optimal method.

    Args:
        data ():
        mask ():
    """

    def get_profile(xnodes, ynodes, p1, p2, step):
        xlst, ylst = [], []
        for xmid in np.arange(p1, p2+1e-5, step):
            x1, x2 = xmid-step/2, xmid+step/2
            mask = (xnodes > x1)*(xnodes < x2)
            mask = mask * (ynodes>0)
            if mask.sum()<=3:
                xlst.append(xmid)
                ylst.append(0.0)
            else:
                ymean, _, _ = get_clip_mean(ynodes[mask], high=3, low=3,
                                            maxiter=15)
                xlst.append(xmid)
                ylst.append(ymean)
        xlst = np.array(xlst)
        ylst = np.array(ylst)
        return ylst

    def errfunc(p, flux, interf, x):
        return flux - fitfunc(p, interf, x)
    def fitfunc(p, interf, x):
        A, cen = p
        return A*interf(x-cen)

    p1 = profilex[0]
    p2 = profilex[-1]

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    profilefunc_lst = []

    for disp_x in disp_x_lst:
        fig_loop = plt.figure(dpi=150, figsize=(10,4))
        for iloop in range(2):
            apernode_x_lst, apernode_y_lst = [], []
            for y in np.arange(disp_x-20, disp_x+20+1e-3, 5):
                y = int(y)
                for aper, aperloc in sorted(apertureset.items()):
                    cen = aperloc.position(y)
                    intc = np.int(np.round(cen))
                    x1 = int(intc + p1)
                    x2 = int(intc + p2 + 1)
                    x1 = max(x1, 0)
                    x2 = min(x2, nx)
                    if x2-x1 < (p2-p1)/2:
                        continue
                    indx = np.arange(x1, x2)
                    flux = data[y, indx]
    
                    if iloop==0:
                        A = flux.sum()
                    else:
                        para = [flux.sum(), cen]
                        mask = np.ones_like(flux, dtype=np.bool)
                        for ite in range(10):
                            result = opt.least_squares(errfunc, para,
                                        bounds=((-np.inf, cen-2),
                                                (np.inf, cen+2)),
                                        args = (flux[mask], interf, indx[mask]))
                            newpara = result['x']
                            fitprof = fitfunc(newpara, interf, indx)
                            resprof = flux - fitprof
                            std = resprof[mask].std()
                            new_mask = (resprof < 3*std)*(resprof > -3*std)
                            if new_mask.sum() == mask.sum():
                                break
                            mask = new_mask
                            para = newpara
                        A, cen = newpara
    
                    if A < 0:
                        continue
                    newx = indx - cen
                    normflux = flux/A
                    for vx, vy in zip(newx, normflux):
                        apernode_x_lst.append(vx)
                        apernode_y_lst.append(vy)
        
            apernode_x_lst = np.array(apernode_x_lst)
            apernode_y_lst = np.array(apernode_y_lst)
            profile = get_profile(
                        apernode_x_lst, apernode_y_lst, p1, p2, 0.5)
            interf = intp.InterpolatedUnivariateSpline(
                        profilex, profile, k=3, ext=0)
            newprofilex = np.arange(p1, p2+1e-3, 0.1)
            newprofile = interf(newprofilex)
        
            ## plot in fig_loop
            ax_loop = fig_loop.add_axes([0.07+iloop*0.5, 0.1, 0.4, 0.8])
            ax_loop.scatter(apernode_x_lst, apernode_y_lst,
                        c='gray', s=5, alpha=0.2)
            ax_loop.plot(newprofilex, newprofile, '-', lw=1, c='C1')
            #ax_loop.plot(profilex_lst, profile+profile_std, '--', lw=0.5, c='C1')
            #ax_loop.plot(profilex_lst, profile-profile_std, '--', lw=0.5, c='C1')
            ax_loop.set_xlim(profilex[0]-1, profilex[-1]+1)
            ax_loop.set_ylim(-0.02, 0.15)
            ax_loop.axvline(x=0, color='k', ls='--', lw=0.6)
            ax_loop.axhline(y=0, color='k', ls='--', lw=0.6)
            ax_loop.grid(True, ls='--', color='gray', lw=0.6)
            ax_loop.set_axisbelow(True)
            ax_loop.set_title('Loop {}'.format(iloop))

            '''
            fig_loop2 = plt.figure(dpi=150, figsize=(6,4))
            ax_loop2 = fig_loop2.add_axes([0.1,0.1,0.8,0.8])
            ax_loop2.scatter(apernode_x_lst, apernode_y_lst,
                        c='gray', s=5, alpha=0.2)
            ax_loop2.plot(newprofilex, newprofile, '-', lw=1, c='C1')
            ax_loop2.set_xlim(profilex[0]-1, profilex[-1]+1)
            ax_loop2.set_ylim(-0.02, 0.15)
            ax_loop2.axvline(x=0, color='k', ls='--', lw=0.6)
            ax_loop2.axhline(y=0, color='k', ls='--', lw=0.6)
            ax_loop2.grid(True, ls='--', color='gray', lw=0.6)
            ax_loop2.set_axisbelow(True)
            title = 'Profile loop {} for Row {:04d}'.format(iloop, disp_x)
            fig_loop2.suptitle(title)
            figname = 'rowprofile_{:04d}_loop_{}.png'.format(disp_x, iloop)
            fig_loop2.savefig(figname)
            plt.close(fig_loop2)
            '''

        # loop ends here
        profilefunc_lst.append(profile)

        fig_loop.suptitle('Profile loop for Row {:04d}'.format(disp_x))
        fig_loop.savefig('row_{:04d}_loops.png'.format(disp_x, iloop))
        plt.close(fig_loop)

    profilenode_lst = disp_x_lst
    profilefunc_lst = np.array(profilefunc_lst)
    npoints = profilefunc_lst.shape[1]

    interprofilefunc_lst = {}
    for y in ally:
        profile = np.zeros(npoints)
        for i in np.arange(npoints):
            f = intp.InterpolatedUnivariateSpline(
                    profilenode_lst, profilefunc_lst[:, i], k=3)
            profile[i] = f(y)
        interprofilefunc = intp.InterpolatedUnivariateSpline(
                profilex, profile, k=3, ext=1)
        interprofilefunc_lst[y] = interprofilefunc

    fig_allprofile = plt.figure(figsize=(6,8), dpi=150)
    ax_allprofile = fig_allprofile.add_axes([0.1, 0.1, 0.8, 0.8])
    for y in range(0, ny, 16):
        newx = np.arange(p1, p2, 0.1)
        interf = interprofilefunc_lst[y]
        ax_allprofile.plot(newx, interf(newx)+0.0001*y,
                lw=0.6, alpha=0.4, color='C0')
    ax_allprofile.grid(True, ls='--', lw=0.5)
    ax_allprofile.set_axisbelow(True)
    fig_allprofile.savefig('profile_stack.png')
    plt.close(fig_allprofile)

    flux_sum_lst = {}
    flux_opt_lst = {}
    flux_err_lst = {}
    back_sum_lst = {}
    for aper, aperloc in sorted(apertureset.items()):
        flux_sum_lst[aper] = []
        flux_opt_lst[aper] = []
        flux_err_lst[aper] = []
        back_sum_lst[aper] = []
        dcen_lst = []
        for y in ally:
            interf = interprofilefunc_lst[y]
            cen = aperloc.position(y)
            intc = np.int(np.round(cen))
            x1 = int(intc + p1)
            x2 = int(intc + p2 + 1)
            x1 = max(x1, 0)
            x2 = min(x2, nx)
            if x2-x1 < (p2-p1)/2:
                flux_opt_lst[aper].append(0.0)
                flux_sum_lst[aper].append(0.0)
                flux_err_lst[aper].append(0.0)
                back_sum_lst[aper].append(0.0)
                continue
            indx = np.arange(x1, x2)
            flux = data[y, indx]
            back = background[y, indx]
            para = [flux.sum(), cen]
            mask = np.ones_like(flux, dtype=np.bool)
            for ite in range(10):
                result = opt.least_squares(errfunc, para,
                        bounds=((-np.inf, cen-0.5), (np.inf, cen+0.5)),
                        args=(flux[mask], interf, indx[mask]))
                newpara = result['x']
                fitprof = fitfunc(newpara, interf, indx)
                resprof = flux - fitprof
                std = resprof[mask].std()
                new_mask = resprof < 3*std
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask
                para = newpara

            #var = np.maximum(flux+back, 0)+(ron/gain)**2
            var = np.maximum(fitprof+back, 0)+(ron/gain)**2
            mask = resprof < 3*np.sqrt(var)
            s_lst = 1/var
            normprof = fitprof/fitprof.sum()
            fopt = (  (s_lst*normprof*flux)[mask].sum()  )/(
                      (s_lst*normprof**2)[mask].sum()    )
            vopt = (  normprof[mask].sum()               )/(
                      (s_lst*normprof**2)[mask].sum()    )
            ferr = math.sqrt(vopt)
            fsum = flux.sum()
            bsum = back.sum()
            flux_opt_lst[aper].append(fopt)
            flux_sum_lst[aper].append(fsum)
            flux_err_lst[aper].append(ferr)
            back_sum_lst[aper].append(bsum)

            dcen_lst.append(cen-newpara[1])
            #if aper==4 or aper==0:
            if False:
                if y%30==0:
                    fig_pix = plt.figure(figsize=(18,10), dpi=150)
                irow = int((y%30)/6)
                icol = (y%30)%6
                _x = 0.04 + icol*0.16
                _y = 0.05 + (4-irow)*0.19
                ax = fig_pix.add_axes([_x, _y, 0.14, 0.17])
                #ax.plot(indx-para[1], fitprof, 'o-',
                #        color='w', mec='C0', ms=4, lw=0.8)
                #ax.plot(indx[mask]-para[1], fitprof[mask], 'o',
                #        color='C0', ms=4)
                newx = np.arange(indx[0], indx[-1], 0.1)
                fitprofnew = fitfunc(newpara, interf, newx)
                ax.plot(newx-newpara[1], fitprofnew, '-', color='C0', lw=0.5)
                #ax.plot(newx-para[1], fitprofnew+1*std, '--', color='C0', lw=0.5)
                #ax.plot(newx-para[1], fitprofnew-1*std, '--', color='C0', lw=0.5)
                ax.fill_between(newx-newpara[1], fitprofnew+1*std, fitprofnew-1*std,
                        facecolor='C0', alpha=0.1)
                #ax.plot(indx[mask]-newpara[1], flux[mask], ls='-',
                #        color='C1', lw=0.8, zorder=1, ms=2)
                x1, x2 = ax.get_xlim()
                y1, y2 = ax.get_ylim()
                #ax.plot(indx-newpara[1], flux, '--', color='C1', lw=0.8, zorder=-1)
                ax.errorbar(indx-newpara[1], flux, yerr=np.sqrt(var),
                        fmt='o', mec='C1', mew=0.6, mfc='w', ecolor='C1',
                        ms=4, lw=0.6, zorder=-1)
                ax.plot(indx[mask]-newpara[1], flux[mask], 'o',
                        color='C1', zorder=1, ms=4)
                ax.plot(indx-newpara[1], np.zeros_like(indx), '^',
                        zorder=-1, ms=4, mec='C1', mew=0.6, mfc='w')
                ax.plot(indx[mask]-newpara[1], np.zeros_like(indx[mask]), '^',
                        zorder=1, ms=3, color='C1')
                ax.text(0.95*x1+0.05*x2, 0.1*y1+0.9*y2, 'Y=%d'%y,
                        fontsize=9)
                ax.text(0.35*x1+0.65*x2, 0.1*y1+0.9*y2, '%7g'%fsum,
                        fontsize=9, color='C0')
                ax.text(0.35*x1+0.65*x2, 0.2*y1+0.8*y2, '%7g'%fopt,
                        fontsize=9, color='C1')
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                ax.axhline(y=0, c='k', ls='--', lw=0.5)
                ax.axvline(x=0, c='k', ls='--', lw=0.5)
                ax.axvline(x=cen-newpara[1], ls='-', lw=0.5, color='C1')
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                if y%30 == 29 or y == ny-1:
                    if not os.path.exists('images'):
                        os.mkdir('images')
                    fig_pix.savefig('images/fitting_%02d_%04d_var.png'%(aper, y))
                    plt.close(fig_pix)

        flux_sum_lst[aper] = np.array(flux_sum_lst[aper])
        flux_opt_lst[aper] = np.array(flux_opt_lst[aper])
        back_sum_lst[aper] = np.array(back_sum_lst[aper])
        flux_err_lst[aper] = np.array(flux_err_lst[aper])

        #if aper==4:
        if False:
            fig2 = plt.figure()
            ax2 = fig2.gca()
            ax2.plot(dcen_lst)

            fig_comp = plt.figure(figsize=(12,6), dpi=150)
            ax_comp1 = fig_comp.add_axes([0.1,0.4,0.8,0.5])
            ax_comp2 = fig_comp.add_axes([0.1,0.1,0.8,0.25])
            ax_comp1.plot(flux_sum_lst[aper],
                    lw=0.6, color='C0', alpha=0.6, label='Standard')
            ax_comp1.plot(flux_opt_lst[aper],
                    lw=0.6, color='C3', alpha=0.6, label='Optimal')
            ax_comp2.plot(back_sum_lst[aper],
                    lw=0.6, color='C0', alpha=0.6, label='Standard Background')
            for ax in fig_comp.get_axes():
                ax.set_xlim(0, ny-1)
                ax.grid(True, ls='--', lw=0.5)
                ax.set_axisbelow(True)
                ax.legend(loc='upper right')
            fig_comp.savefig('flux_comp_{:03d}.png'.format(aper))
            plt.close(fig_comp)

            for x in [600, 1050, 1400, 1960, 2700, 3300, 4000, 4200]:
                fig_comp2 = plt.figure(figsize=(12,6), dpi=150)
                ax_comp21 = fig_comp2.gca()
                i1 = x-120
                i2 = x+120
                ax_comp21.plot(np.arange(i1, i2), flux_sum_lst[aper][i1:i2],
                        lw=0.6, color='C0', alpha=0.6, label='Standard')
                ax_comp21.plot(np.arange(i1, i2), flux_opt_lst[aper][i1:i2],
                        lw=0.6, color='C1', alpha=0.6, label='Optimal')
                ax_comp21.grid(True, ls='--', lw=0.5)
                ax_comp21.set_axisbelow(True)
                ax_comp21.set_xlim(i1, i2)
                fig_comp2.savefig('flux_comp_{:03d}_zoom_{:04d}.png'.format(
                    aper, x))
                plt.close(fig_comp2)

        print('Spectrum of Aperture {:3d} extracted'.format(aper))


    return flux_opt_lst, flux_err_lst, flux_sum_lst, back_sum_lst




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

