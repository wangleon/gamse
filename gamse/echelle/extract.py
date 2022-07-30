import os
import math
import time
import logging

logger = logging.getLogger(__name__)

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.integrate as intg
from scipy.signal import savgol_filter
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
        ron, gain, profilex, disp_x_lst, main_disp, profile_lst=None,
        recenter=False,
        upper_clipping=3.0, mode='normal', figpath='debug',
        plot_apertures=[]):
    """Extract 1-D spectra from the input image using the optimal method.

    Args:
        data (:class:`numpy.ndarray`): Input Image data with background
            subtracted.
        mask (:class:`numpy.ndarray`): Image mask with the same shape as
            **data**.
        background (:class:`numpy.ndarray`): Backgroud Image with the same
            shape as **data**.
        apertureset (:class:`~gamse.echelle.trace.ApertureSet`): Apertures.
        ron (float): Readout Noise in unit of e-.
        gain (float): CCD gain in unit of e-/ADU.
        profilex (:class:`numpy.ndarray`): The sampling points of cross-order
            profiles.
        disp_x_lst (:class:`numpy.ndarray`): An array of the profile sampling
            position along the main-dispersion direction.
        main_disp (str): Axes of the main dispersion direction ("x" or "y").
        profile_lst (:class:`numpy.ndarray`): Cross order profile array.

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

    # fitting and error function of profile fitting, with central positions
    # and amplitude as the free parameters.
    def errfunc(p, flux, interf, x):
        return flux - fitfunc(p, interf, x)
    def fitfunc(p, interf, x):
        A, cen = p
        return A*interf(x-cen)

    # fitting and error function of profile fitting, with central positions
    # fixed, and only the amplitudes as the free parameter.
    def errfunc2(p, flux, interf, x):
        return flux - fitfunc2(p, interf, x)
    def fitfunc2(p, interf, x):
        A = p[0]
        return A*interf(x)

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    # determine pixel number along the main-dispersion and cross-dispesion
    # directions
    ndisp = {'x':nx, 'y':ny}[main_disp]
    ncros = {'x':ny, 'y':nx}[main_disp]

    # left and right ends of profile sampling
    p1 = profilex[0]
    p2 = profilex[-1]

    if profile_lst is None:
        # cross-order profile is not given. generate profile with the input
        # image.

        profile_lst = []

        for idisp_c in disp_x_lst:
            fig_loop = plt.figure(dpi=150, figsize=(10,4))
            for iloop in range(2):
                apernode_x_lst, apernode_y_lst = [], []
                for idisp in np.arange(idisp_c-20, idisp_c+20+1e-3, 5):
                    idisp = int(idisp)
                    for aper, aperloc in sorted(apertureset.items()):
                        cen = aperloc.position(idisp)
                        intc = np.int(np.round(cen))
                        i1 = int(intc + p1)
                        i2 = int(intc + p2 + 1)
                        i1 = max(i1, 0)
                        i2 = min(i2, ncros)
                        if i2-i1 < (p2-p1)/2:
                            continue
                        indx = np.arange(i1, i2)
    
                        # slice the image array
                        if main_disp == 'x':
                            flux = data[indx, idisp]
                        elif main_disp == 'y':
                            flux = data[idisp, indx]
                        else:
                            raise ValueError
        
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
            profile_lst.append(profile)
    
            fig_loop.suptitle('Profile loop for Row {:04d}'.format(idisp_c))
            fig_loop.savefig('row_{:04d}_loops.png'.format(idisp_c, iloop))
            plt.close(fig_loop)
    else:
        pass

    # get a list of cross-order profile interpolation function for every x
    # along the main-dispersion direction.
    profilex_lst = disp_x_lst
    profile_lst = np.array(profile_lst)
    npoints = profile_lst.shape[1]

    interprofilefunc_lst = {}
    for idisp in np.arange(ndisp):
        profile = np.zeros(npoints)
        for i in np.arange(npoints):
            f = intp.InterpolatedUnivariateSpline(
                    profilex_lst, profile_lst[:, i], k=3, ext=3)
            profile[i] = f(idisp)
        interprofilefunc = intp.InterpolatedUnivariateSpline(
                profilex, profile, k=3, ext=3)
        interprofilefunc_lst[idisp] = interprofilefunc

    #fig_allprofile = plt.figure(figsize=(6,8), dpi=150)
    #ax_allprofile = fig_allprofile.add_axes([0.1, 0.1, 0.8, 0.8])
    #for idisp in range(0, ndisp, 16):
    #    newx = np.arange(p1, p2, 0.1)
    #    interf = interprofilefunc_lst[idisp]
    #    ax_allprofile.plot(newx, interf(newx)+0.0001*idisp,
    #            lw=0.6, alpha=0.4, color='C0')
    #ax_allprofile.grid(True, ls='--', lw=0.5)
    #ax_allprofile.set_axisbelow(True)
    #fig_allprofile.savefig('profile_stack.png')
    #plt.close(fig_allprofile)


    # generate idisp_lst, from center to the limbs
    a = allx[0:ndisp//2]
    b = allx[ndisp//2:]
    idisp_lst = np.transpose(np.vstack((a[::-1],b))).flatten()

    # initialize result arrays
    flux_sum_lst = {}
    flux_opt_lst = {}
    flux_err_lst = {}
    back_sum_lst = {}
    back_opt_lst = {}
    for aper, aperloc in sorted(apertureset.items()):
        t1 = time.time()
        flux_sum = np.zeros(ndisp, dtype=np.float32)
        flux_opt = np.zeros(ndisp, dtype=np.float32)
        flux_err = np.zeros(ndisp, dtype=np.float32)
        back_sum = np.zeros(ndisp, dtype=np.float32)
        back_opt = np.zeros(ndisp, dtype=np.float32)
        dcen_lst = np.zeros(ndisp, dtype=np.float32)
        qsnr_lst = np.zeros(ndisp, dtype=np.float32)

        if recenter:
            for idisp in idisp_lst:
                interf = interprofilefunc_lst[idisp]
                cen = aperloc.position(idisp)
                intc = np.int(np.round(cen))
                i1 = int(intc + p1)
                i2 = int(intc + p2 + 1)
                i1 = max(i1, 0)
                i2 = min(i2, ncros)
                if i2-i1 < (p2-p1)/2:
                    continue
                indx = np.arange(i1, i2)
                if main_disp == 'x':
                    flux = data[indx, idisp]
                    back = background[indx, idisp]
                elif main_disp == 'y':
                    flux = data[idisp, indx]
                    back = background[idisp, indx]
                else:
                    raise ValueError

                # initialize iteration mask for optimal extraction
                mask = flux < np.median(flux) + 3*np.std(flux)
                A0 = flux[mask].max()
                para0 = [A0, cen]

                for ite in range(3):
                    result = opt.least_squares(errfunc, para0,
                        args=(flux[mask], interf, indx[mask]))
                    para = result['x']
                    fitprof = fitfunc(para, interf, indx)
                    resprof = flux - fitprof
                    std = resprof[mask].std()
                    new_mask = resprof < 3*std
                    if new_mask.sum() == mask.sum():
                        break
                    mask *= new_mask

                dcen_lst[idisp] = para[1]-cen
                qsnr_lst[idisp] = para[0]/std

            deg = 5
            m = np.ones_like(dcen_lst, dtype=bool)
            for ite in range(5):
                binx = []
                biny = []
                binstep = 128
                for i1 in np.arange(0, ndisp, binstep):
                    i2 = i1 + binstep
                    binx.append(i1 + binstep/2)
                    m2 = m[i1:i2]
                    biny.append(np.mean(dcen_lst[i1:i2][m2]))
                binx = np.array(binx)
                biny = np.array(biny)

                coeff = np.polyfit(binx/ndisp, biny, deg=deg)
                newdcen = np.polyval(coeff, allx/ndisp)
                yres = dcen_lst - newdcen
                std = yres[m].std()
                newm = m*(yres<3*std)*(yres>-3*std)
                if newm.sum()==m.sum():
                    break
                m = newm

            #loglikelihood = (-(yres[m]/std)**2/2).sum()
            #bic = -2*loglikelihood + (deg+1)*np.log(m.sum())
            if mode == 'debug':
                fig3 = plt.figure(figsize=(8,4))
                ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
                ax3.plot(allx[m], dcen_lst[m], lw=0.5, color='C0',
                        alpha=0.8, zorder=3)
                _y1, _y2 = ax3.get_ylim()
                #_y1 = max(_y1, -1.8)
                #_y2 = min(_y2, +1.8)
                ax3.plot(allx, dcen_lst, lw=0.5, color='C0',
                        alpha=0.2, zorder=1)
                ax3.plot(binx, biny, 'o', ms=3, color='C1',
                        alpha=0.8, zorder=4)
                ax3.plot(newdcen, '-',  lw=0.7, color='C1')
                ax3.plot(newdcen-std,   '--', lw=0.7, color='C1')
                ax3.plot(newdcen+std,   '--', lw=0.7, color='C1')
                ax3.plot(newdcen-3*std, '--', lw=0.7, color='C1')
                ax3.plot(newdcen+3*std, '--', lw=0.7, color='C1')
                ax3.set_ylim(_y1, _y2)
                ax3.set_xlim(0, ndisp-1)
                ax3.grid(True, ls='--', lw=0.5)
                ax3c = ax3.twinx()
                ax3c.plot(qsnr_lst, '-', color='k', alpha=0.2, lw=0.5)
                ax3c.set_xlim(0, ndisp-1)
                fig3.suptitle('Aperture {}'.format(aper))
                fig3.savefig('debug/fitcen_{:03d}.png'.format(aper), dpi=150)
                plt.close(fig3)


        count = 0
        prev_para = None
        for idisp in idisp_lst:
            interf = interprofilefunc_lst[idisp]
            cen = aperloc.position(idisp) + newdcen[idisp]
            intc = np.int(np.round(cen))
            i1 = int(intc + p1)
            i2 = int(intc + p2 + 1)
            i1 = max(i1, 0)
            i2 = min(i2, ncros)
            if i2-i1 < (p2-p1)/2:
                flux_opt[idisp] = 0.0
                flux_sum[idisp] = 0.0
                flux_err[idisp] = 0.0
                back_sum[idisp] = 0.0
                continue
            indx = np.arange(i1, i2)
            if main_disp == 'x':
                flux = data[indx, idisp]
                back = background[indx, idisp]
            elif main_disp == 'y':
                flux = data[idisp, indx]
                back = background[idisp, indx]
            else:
                raise ValueError

            # sum extraction
            fsum = flux.sum()
            bsum = back.sum()
            flux_sum[idisp] = fsum
            back_sum[idisp] = bsum

            # initialize iteration mask for optimal extraction
            mask = flux < np.median(flux) + 3*np.std(flux)
            A0 = flux[mask].max()

            para0 = [A0]

            for ite in range(3):
                result = opt.least_squares(errfunc2, para0,
                    args=(flux[mask], interf, indx[mask]-cen))
                para = result['x']
                fitprof = fitfunc2(para, interf, indx-cen)
                resprof = flux - fitprof
                std = resprof[mask].std()
                qsnr = para[0]/std
                new_mask = resprof < 3*std
                if new_mask.sum() == mask.sum():
                    break
                mask *= new_mask


            #if aper==0:
            #    print(idisp, ite, A0, para[0])

            # Horne 1986, PASP, 98, 609 Formula 7-9
            #var = np.maximum(flux+back, 0)+(ron/gain)**2
            var = np.maximum(fitprof+back, 0)+(ron/gain)**2
            #mask = resprof < upper_clipping*np.sqrt(var)
            s_lst = 1/var
            normprof = fitprof/fitprof.sum()
            ssum = (s_lst*normprof**2)[mask].sum()
            fopt = ((s_lst*normprof*flux)[mask].sum())/ssum
            vopt = (normprof[mask].sum()             )/ssum
            bopt = ((s_lst*normprof*back)[mask].sum())/ssum
            ferr = math.sqrt(vopt)

            flux_opt[idisp] = fopt
            flux_err[idisp] = ferr
            back_opt[idisp] = bopt

            if mode=='debug' and aper in plot_apertures:
                if count%30==0:
                    fig_pix = plt.figure(figsize=(18,10), dpi=150)
                irow = int((count%30)/6)
                icol = (count%30)%6
                _x = 0.04 + icol*0.16
                _y = 0.05 + (4-irow)*0.19
                ax = fig_pix.add_axes([_x, _y, 0.14, 0.17])
                newx = np.arange(indx[0], indx[-1], 0.1)
                fitprofnew = fitfunc2(para, interf, newx-cen)

                ax.plot(newx-cen, fitprofnew, '-', color='C0', lw=0.5)
                ax.fill_between(newx-cen, fitprofnew+1*std, fitprofnew-1*std,
                        facecolor='C0', alpha=0.1)
                x1, x2 = ax.get_xlim()
                y1, y2 = ax.get_ylim()
                #ax.plot(indx[mask]-cen, flux[mask], ls='-',
                #        color='C1', lw=0.8, zorder=1, ms=2)
                ax.errorbar(indx-cen, flux, yerr=np.sqrt(var),
                        fmt='o-',mew=0.6, mfc='w', ms=4, lw=0.6, zorder=-1,
                        color='C1', mec='C1', ecolor='C1',
                        )
                ax.plot(indx[mask]-cen, flux[mask], 'o',
                        color='C1', zorder=1, ms=4)
                ax.axhline(np.median(flux), x1, x2, ls=':', lw=0.5, c='C2')
                ax.fill_between([x1, x2], np.median(flux), np.median(flux)+flux.std(),
                        facecolor='C2', alpha=0.1)
                ax.plot(indx[~mask]-cen, np.zeros_like(indx)[~mask], 'x',
                        zorder=-1, ms=4, c='C1')
                ax.text(0.95*x1+0.05*x2, 0.1*y1+0.9*y2, 'X=%d'%idisp,
                        fontsize=9)
                ax.text(0.05*x1+0.95*x2, 0.1*y1+0.9*y2, '%7g'%fsum,
                        fontsize=9, color='C0', ha='right')
                ax.text(0.05*x1+0.95*x2, 0.2*y1+0.8*y2, '%7g'%fopt,
                        fontsize=9, color='C1', ha='right')
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                ax.axhline(y=0, c='k', ls='--', lw=0.5)
                ax.axvline(x=0, c='k', ls='--', lw=0.5)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                if count%30 == 29 or count == ndisp-1:
                    if not os.path.exists(figpath):
                        os.mkdir(figpath)
                    fname = 'extopt_{:03d}_{:03d}.png'.format(
                            aper, int(count/30))
                    figfilename = os.path.join(figpath, fname)
                    fig_pix.savefig(figfilename)
                    plt.close(fig_pix)

            # save this para as prev_para for next column
            prev_para = para

            #idisp += direction
            count += 1

        flux_sum_lst[aper] = flux_sum
        flux_opt_lst[aper] = flux_opt
        back_sum_lst[aper] = flux_err
        back_opt_lst[aper] = back_sum
        flux_err_lst[aper] = back_opt

        if False:
        #mode=='debug':
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(flux_opt, lw=0.5)
            ax2.plot(dcen_lst, lw=0.5, alpha=0.3)
            # plot Qsnr list
            ax1c = ax1.twinx()
            ax1c.plot(qsnr_lst, lw=0.5, c='k', alpha=0.2)
            ax1c.axhline(y=10, lw=0.5, ls='--', c='k')
            ax1c.axhline(y=5, lw=0.5, ls='--', c='k')

            m = np.ones_like(dcen_lst, dtype=bool)
            allx = np.arange(ndisp)

            #winlen = ndisp//10
            #if winlen%2==0:
            #    winlne += 1
            #for ite in range(5):
            #    f = intp.InterpolatedUnivariateSpline(allx[m], dcen_lst[m], k=3)
            #    ysmooth = savgol_filter(f(allx), window_length=winlen,
            #               polyorder=3)
            #    yres = dcen_lst - ysmooth
            #    std = yres[m].std()
            #    newm = m*(yres<3*std)*(yres>-3*std)
            #    if newm.sum()==m.sum():
            #        break
            #    m = newm

            ax2.plot(allx[m], dcen_lst[m], lw=0.5, color='C0', alpha=0.8)
            ax2.plot(binx, biny, 'o', ms=3, color='C1', alpha=0.9)
            ax2.plot(ysmooth,       '-',  lw=0.7, color='C1',
                        label='Order = {}'.format(deg))
            ax2.plot(ysmooth-std,   '--', lw=0.7, color='C1')
            ax2.plot(ysmooth+std,   '--', lw=0.7, color='C1')
            ax2.plot(ysmooth-3*std, '--', lw=0.7, color='C1')
            ax2.plot(ysmooth+3*std, '--', lw=0.7, color='C1')

            
            _y1, _y2 = ax2.get_ylim()
            _y1 = max(_y1, -1.8)
            _y2 = min(_y2, 1.8)
            ax2.set_ylim(_y1, _y2)
            ax1.set_xlim(0, ndisp-1)
            ax2.set_xlim(0, ndisp-1)
            ax2.legend(loc='upper left')
            ax1c.set_xlim(0, ndisp-1)
            fig.savefig('debug/checkcen_{:03d}.png'.format(aper), dpi=200)
            plt.close(fig)

        t2 = time.time()
        print('Spectrum of Aperture {:3d} extracted'.format(aper), t2-t1)


    return flux_opt_lst, flux_err_lst, back_opt_lst, flux_sum_lst, back_sum_lst


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


def extract_aperset_optimal_multifiber(data, mask, background,
        apertureset_lst, ron, gain, main_disp, profilex, disp_x_lst,
        extract_fiber, all_profile_lst=None):
    """Extract 1-D spectra from the input image using the optimal method.

    Args:
        data (:class:`numpy.ndarray`): Input Image data with background
            subtracted.
        mask (:class:`numpy.ndarray`): Image mask with the same shape as
            **data**.
        background (:class:`numpy.ndarray`): Backgroud Image with the same
            shape as **data**.
        apertureset_lst (list): Apertures of different fibers. A list of
            :class:`~gamse.echelle.trace.ApertureSet` instances.
        ron (float): Readout Noise in unit of e-.
        gain (float): CCD gain in unit of e-/ADU.
        main_disp (str): Axes of the main dispersion direction ("x" or "y").
        profilex (:class:`numpy.ndarray`): The sampling points of cross-order
            profiles.
        disp_x_lst (:class:`numpy.ndarray`): An array of the profile sampling
            position along the main-dispersion direction.
        extract_fiber (str): The fiber for which the 1-d spectra to be
            extracted.
        all_profile_lst (:class:`numpy.ndarray`): A dict of cross order
            profile array in different orders.
    """

    # list of all fibers in apertureset_lst
    fiber_lst = sorted(list(apertureset_lst.keys()))

    # fitting and error function of profile fitting, with central positions
    # and amplitude as the free parameters.
    def errfunc(p, flux, interf, x):
        return flux - fitfunc(p, interf, x)
    def fitfunc(p, interf, x):
        A, cen = p
        return A*interf(x-cen)

    # fitting and error function of profile fitting, with central positions
    # fixed, and only the amplitudes as the free parameter.
    def errfunc2(p, flux, interf_lst, cen_lst, x):
        return flux - fitfunc2(p, interf_lst, cen_lst, x)
    def fitfunc2(p, interf_lst, cen_lst, x):
        y = np.zeros_like(x, dtype=np.float32)
        for i in range(len(interf_lst)):
            A = p[i]
            interf = interf_lst[i]
            cen = cen_lst[i]
            y += A*interf(x-cen)
        return y

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    # determine pixel number along the main-dispersion and cross-dispesion
    # directions
    ndisp = {'x':nx, 'y':ny}[main_disp]
    ncros = {'x':ny, 'y':nx}[main_disp]

    # left and right ends of profile sampling
    p1 = profilex[0]
    p2 = profilex[-1]

    if all_profile_lst is None:
        pass
    else:
        pass

    profilex_lst = disp_x_lst

    # convert all elements in all_profile_lst into numpy array
    for fiber in fiber_lst:
        all_profile_lst[fiber] = np.array(all_profile_lst[fiber])


    # interprofilefunc list should be an fiber dict
    all_interprofilefunc_lst = {}
    for fiber in fiber_lst:
        # get a list of cross-order profile interpolation function for every x
        # along the main-dispersion direction.
        profile_lst = all_profile_lst[fiber]
        npoints = profile_lst.shape[1]

        interprofilefunc_lst = {}
        for idisp in np.arange(ndisp):
            profile = np.zeros(npoints)
            for i in np.arange(npoints):
                f = intp.InterpolatedUnivariateSpline(
                        profilex_lst, profile_lst[:, i], k=3, ext=3)
                profile[i] = f(idisp)
            interprofilefunc = intp.InterpolatedUnivariateSpline(
                    profilex, profile, k=3, ext=3)
            interprofilefunc_lst[idisp] = interprofilefunc
        all_interprofilefunc_lst[fiber] = interprofilefunc_lst

    # sort all orders based on their positions along the cross-order direction
    allpos_lst = []
    for fiber in fiber_lst:
        apertureset = apertureset_lst[fiber]
        for aper, aperloc in sorted(apertureset.items()):
            pos = aperloc.position(ndisp//2)
            allpos_lst.append((fiber, aper, pos))
    allpos_lst = sorted(allpos_lst, key=lambda item: item[2])
    # sort apertures based on pos
    allaper_lst = [(fiber, aper) for fiber, aper, pos in allpos_lst]
    # number of all orders (apertures)
    naper = len(allaper_lst)

    # focus on the fiber to be extracted.
    apertureset = apertureset_lst[extract_fiber]
    interprofilefunc_lst = all_interprofilefunc_lst[extract_fiber]

    # initialize result arrays
    flux_sum_lst = {}
    flux_opt_lst = {}
    flux_err_lst = {}
    back_sum_lst = {}
    back_opt_lst = {}
    for aper, aperloc in sorted(apertureset.items()):
        flux_sum_lst[aper] = []
        flux_opt_lst[aper] = []
        flux_err_lst[aper] = []
        back_sum_lst[aper] = []
        back_opt_lst[aper] = []

        idx = allaper_lst.index((extract_fiber, aper))

        for idisp in np.arange(ndisp):
            interf = interprofilefunc_lst[idisp]
            interf_lst = [interf]
            cen = aperloc.position(idisp)
            cen_lst = [cen]
            intc = np.int(np.round(cen))
            i1 = int(intc + p1)
            i2 = int(intc + p2 + 1)
            i1 = max(i1, 0)
            i2 = min(i2, ncros)
            if i2-i1 < (p2-p1)/2:
                flux_opt_lst[aper].append(0.0)
                flux_sum_lst[aper].append(0.0)
                flux_err_lst[aper].append(0.0)
                back_sum_lst[aper].append(0.0)
                continue
            indx = np.arange(i1, i2)
            if main_disp == 'x':
                flux = data[indx, idisp]
                back = background[indx, idisp]
            elif main_disp == 'y':
                flux = data[idisp, indx]
                back = background[idisp, indx]
            else:
                raise ValueError
            para_lst = [flux.max()]
            # extend the interf_lst and cen_lst

            # get the fiber and aperture of the left order
            if idx > 0:
                fiber1, aper1 = allaper_lst[idx-1]
                interf1 = all_interprofilefunc_lst[fiber1][idisp]
                interf_lst.append(interf1)
                aperloc1 = apertureset_lst[fiber1][aper1]
                cen1 = aperloc1.position(idisp)
                cen_lst.append(cen1)
                intc1 = np.int(np.round(cen1))
                i1 = max(intc1 - 2, 0)
                indx = np.arange(i1, i2)
                if main_disp == 'x':
                    flux = data[indx, idisp]
                    back = background[indx, idisp]
                elif main_disp == 'y':
                    flux = data[idisp, indx]
                    back = background[idisp, indx]
                else:
                    raise ValueError
                para_lst.append(flux[2])

            # get the fiber and aperture of the left order
            if idx < naper-1:
                fiber2, aper2 = allaper_lst[idx+1]
                interf2 = all_interprofilefunc_lst[fiber2][idisp]
                interf_lst.append(interf2)
                aperloc2 = apertureset_lst[fiber2][aper2]
                cen2 = aperloc2.position(idisp)
                cen_lst.append(cen2)
                intc2 = np.int(np.round(cen2))
                i2 = min(intc2 + 2, ncros)
                indx = np.arange(i1, i2)
                if main_disp == 'x':
                    flux = data[indx, idisp]
                    back = background[indx, idisp]
                elif main_disp == 'y':
                    flux = data[idisp, indx]
                    back = background[idisp, indx]
                else:
                    raise ValueError
                para_lst.append(flux[-2])


            mask = np.ones_like(flux, dtype=np.bool)
            for ite in range(10):
                result = opt.least_squares(errfunc2, para_lst,
                        args=(flux[mask], interf_lst, cen_lst, indx[mask]))
                newpara_lst = result['x']
                fitprof = fitfunc2(newpara_lst, interf_lst, cen_lst, indx)
                resprof = flux - fitprof
                std = resprof[mask].std()
                new_mask = resprof < 3*std
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask
                para_lst = newpara_lst

            # calculate fraction of light from this order.
            single_prof = fitfunc2(newpara_lst[0:1], interf_lst[0:1],
                            cen_lst[0:1], indx)
            frac = np.maximum(single_prof/fitprof, 0)

            # Horne 1986, PASP, 98, 609 Formula 7-9
            var = np.maximum(fitprof+back, 0)+(ron/gain)**2
            mask = resprof < 3*np.sqrt(var)
            s_lst = 1/var
            normprof = frac*fitprof/fitprof.sum()
            ssum = (s_lst*normprof**2)[mask].sum()
            fopt = ((s_lst*normprof*flux)[mask].sum())/ssum
            vopt = (normprof[mask].sum()             )/ssum
            bopt = ((s_lst*normprof*back)[mask].sum())/ssum
            ferr = math.sqrt(np.maximum(vopt,0))
            fsum = flux.sum()
            bsum = back.sum()
            flux_opt_lst[aper].append(fopt)
            flux_sum_lst[aper].append(fsum)
            flux_err_lst[aper].append(ferr)
            back_sum_lst[aper].append(bsum)
            back_opt_lst[aper].append(bopt)
        print('Spectrum of Fiber {}, Aperture {:3d} extracted'.format(
                extract_fiber, aper))

    return (flux_opt_lst, flux_err_lst, back_opt_lst,
            flux_sum_lst, back_sum_lst)
