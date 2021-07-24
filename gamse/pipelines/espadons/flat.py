import os
import sys
import time
import math
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.signal import savgol_filter
import scipy.interpolate as intp
import scipy.optimize as opt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import astropy.io.fits as fits

from ...echelle.flat import SpatialProfile
from ...utils.regression import iterative_polyfit
from ...utils.onedarray import iterative_savgol_filter, get_local_minima
from .common import norm_profile, get_mean_profile

def get_flat2(data, mask, apertureset, nflat,
        smooth_A_func,
        smooth_c_func,
        smooth_bkg_func,
        q_threshold=30, mode='normal',
        fig_spatial=None,
        ):
    """ Get flat.
    """

    if mode == 'debug':
        dbgpath = 'debug'
        if not os.path.exists(dbgpath):
            os.mkdir(dbgpath)
        plot_aperpar = True
        figname_aperpar = lambda aper: os.path.join(dbgpath,
                                'aperpar_{:03d}.png'.format(aper))
        save_aperpar = True
        filename_aperpar = lambda aper: os.path.join(dbgpath,
                                'aperpar_{:03d}.dat'.format(aper))
    else:
        plot_aperpar = False
        save_aperpar = False

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    # calculate order positions and boundaries and save them in dicts
    all_positions  = apertureset.get_positions(ally)
    all_boundaries = apertureset.get_boundaries(ally)

    p1, p2, pstep = -16, 16, 0.1
    profile_x = np.arange(p1+1, p2-1+1e-4, pstep)
    profilex_lst = []
    profiley_lst = []

    # find saturation mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    y0 = 54
    winsize = 500
    yc_lst = np.arange(y0, ny, winsize)
    # n = 10

    fig_show = plt.figure(figsize=(12, 3), dpi=200)

    for iyc, yc in enumerate(yc_lst):
        yc = int(yc)
        y = yc
        # initialize the mean profiles
        all_xnodes = []
        all_ynodes = []


        ax = fig_spatial.get_axes()[iyc]

        # has 10 panels, draw 0, 4, 9
        if iyc==0:
            _x = 0.05
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        elif iyc==4:
            _x = 0.05 + 1*0.32
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        elif iyc==9:
            _x = 0.05 + 2*0.32
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        else:
            ax2 = None


        order_cent_lst = np.array([aperloc.position(y)
            for aper, aperloc in sorted(apertureset.items())])
        order_dist_lst = np.diff(order_cent_lst)
        # now size of order_dist_lst = size of order_cent_lst - 1
        order_dist_lst = np.insert(order_dist_lst, 0, order_dist_lst[0])
        order_dist_lst = np.append(order_dist_lst, order_dist_lst[-1])
        # now size of order_dist_lst = size of order_cent_lst + 1
        order_hwin_lst = order_dist_lst/2
        # size of order_hwin_lst = size of order_cent_lst + 1

        for iaper, (aper, aperloc) in enumerate(
                sorted(apertureset.items())):
            cent = order_cent_lst[iaper]
            lwin = order_hwin_lst[iaper]
            rwin = order_hwin_lst[iaper+1]

            ceni = int(round(cent))
            i1 = cent - min(lwin, abs(p1))
            i2 = cent + min(rwin, abs(p2))
            i1, i2 = int(round(i1)), int(round(i2))
            i1 = max(i1, 0)
            i2 = min(i2, nx)
            if i2 - i1 < 10:
                continue

            xnodes = np.arange(i1, i2)
            ynodes = data[y, i1:i2]
            mnodes = mask[y, i1:i2]

            results = norm_profile(xnodes, ynodes)
            # in case returns None results
            if results is None:
                continue

            # unpack the results
            newx, newy, param = results
            v0, _p1, _p2, A, bkg = param

            if A>bkg and bkg>0 and bkg<np.percentile(ynodes, 20):
                for _newx, _newy in zip(newx, newy):
                    all_xnodes.append(_newx)
                    all_ynodes.append(_newy)

                '''
                fig0 = plt.figure()
                ax01 = fig0.add_subplot(211)
                ax02 = fig0.add_subplot(212)
                label = 'A={:.2f}\nv0={:.1f}\np1={:.2f}\np2={:.2f}\nbkg={:.2f}'.format(
                        param[3], param[0], param[1], param[2], param[4])
                ax01.plot(xnodes, ynodes, 'o', label=label)
                ax02.plot(newx, newy, 'o')
                ax01.legend()
                figname = 'test-x_{:04d}_y_{:04d}_{:04d}.png'.format(y, i1, i2)
                fig0.savefig(figname)
                plt.close(fig0)
                '''

                # plotting
                ax.scatter(newx, newy, s=5, alpha=0.3, lw=0)
                if ax2 is not None:
                    ax2.scatter(newx, newy, s=10, alpha=0.3, lw=0)

        # print a progress bar in terminal
        n_finished = iyc + 1
        n_total    = yc_lst.size
        ratio = min(n_finished/n_total, 1.0)
        term_size = os.get_terminal_size()
        nchar = term_size.columns - 60

        string = '>'*int(ratio*nchar)
        string = string.ljust(nchar, '-')
        prompt = 'Constructing slit function'
        string = '\r {:<30s} |{}| {:6.2f}%'.format(prompt, string, ratio*100)
        sys.stdout.write(string)
        sys.stdout.flush()

        all_xnodes = np.array(all_xnodes)
        all_ynodes = np.array(all_ynodes)

        _m = (all_ynodes>-0.05)*(all_ynodes<1.2)
        all_xnodes = all_xnodes[_m]
        all_ynodes = all_ynodes[_m]

        spatial_profile = SpatialProfile(all_xnodes, all_ynodes)
        profile_y = spatial_profile(profile_x)

        ax.plot(profile_x, profile_y, color='k', lw=0.6)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_axisbelow(True)
        _x1, _x2 = p1-2, p2+2
        _y1, _y2 = -0.2, 1.2
        _text = 'Y={:4d}/{:4d}'.format(yc, ny)
        ax.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2, _text)
        ax.set_xlim(_x1, _x2)
        ax.set_ylim(_y1, _y2)
        # temperary
        if ax2 is not None:
            ax2.plot(profile_x, profile_y, color='k', lw=1)
            ax2.grid(True, ls='--', lw=0.5)
            ax2.set_axisbelow(True)
            _x1, _x2 = p1-2, p2+2
            _y1, _y2 = -0.2, 1.2
            _text = 'Y={:4d}/{:4d}'.format(yc, ny)
            ax2.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2, _text)
            ax2.set_xlim(_x1, _x2)
            ax2.set_ylim(_y1, _y2)


        profilex_lst.append(yc)
        profiley_lst.append(profile_y)

    # use light green color
    print(' \033[92m Completed\033[0m')

    fig_show.savefig('spatialprofile_espadons.pdf')
    plt.close(fig_show)

    profilex_lst = np.array(profilex_lst)
    profiley_lst = np.array(profiley_lst)
    npoints = profiley_lst.shape[1]

    interprofilefunc_lst = {}
    corr_mask_array = []
    for y in ally:
        # calculate interpolated profie in this x
        profile = np.zeros(npoints)
        for i in np.arange(npoints):
            f = InterpolatedUnivariateSpline(
                    profilex_lst, profiley_lst[:, i], k=3, ext=3)
            profile[i] = f(y)
        interprofilefunc = InterpolatedUnivariateSpline(
                profile_x, profile, k=3, ext=3)
        interprofilefunc_lst[y] = interprofilefunc

        ilst = np.nonzero(profile>0.1)[0]
        il = profile_x[ilst[0]]
        ir = profile_x[ilst[-1]]
        corr_mask_array.append((il, ir))

    ##################

    flatdata = np.ones_like(data, dtype=np.float32)
    flatspec_lst = {aper: np.full(ny, np.nan) for aper in apertureset}

    # define fitting and error functions
    def fitfunc(param, xdata, interf):
        A, c, b = param
        return A*interf(xdata-c) + b
    def errfunc(param, xdata, ydata, interf):
        return ydata - fitfunc(param, xdata, interf)

    # prepare an x list
    newy_lst = np.arange(0, ny-1, 10)
    if newy_lst[-1] != ny-1:
        newy_lst = np.append(newy_lst, ny-1)

    ###################### loop for every aperture ########################

    for iaper, (aper, aperloc) in enumerate(sorted(apertureset.items())):
        fitpar_lst  = [] # stores (A, c, b). it has the same length as newx_lst
        aperpar_lst = []

        # prepare the figure for plotting the parameters of each aperture
        if plot_aperpar:
            if iaper%5==0:
                fig_aperpar = plt.figure(figsize=(15,8), dpi=150)

        aper_position = all_positions[aper]
        aper_lbound, aper_ubound = all_boundaries[aper]

        t1 = time.time()

        is_first_correct = False
        break_aperture = False

        # loop for every newx. find the fitting parameters for each column
        # prepar the blank parameter for insert
        blank_p = np.array([np.NaN, np.NaN, np.NaN])

        for iy, y in enumerate(newy_lst):
            pos      = aper_position[y]
            # skip this column if central position excess the CCD range
            if pos<0 or pos>nx:
                fitpar_lst.append(blank_p)
                continue

            # determine lower and upper bounds
            lbound = aper_lbound[y]
            ubound = aper_ubound[y]
            x1 = int(max(0,  lbound))
            x2 = int(min(nx, ubound))

            if x2-x1<=4:
                fitpar_lst.append(blank_p)
                continue

            # construct fitting data (xdata, ydata)
            xdata = np.arange(x1, x2)
            ydata = data[y, x1:x2]

            # calculate saturation mask and bad-pixel mask
            _satmask = sat_mask[y, x1:x2]
            _badmask = bad_mask[y, x1:x2]
            # skip this column if too many saturated or bad pixels
            if _satmask.sum()>=3 or _badmask.sum()>=3:
                fitpar_lst.append(blank_p)
                continue
            # estimate the SNR
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(nx, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-x1:_i2-x1])*nflat))
            # skip this column if sn is too low
            if sn < q_threshold:
                fitpar_lst.append(blank_p)
                continue

            # begin fitting
            interf = interprofilefunc_lst[y]
            p0 = [ydata.max()-ydata.min(), pos, max(0,ydata.min())]

            # find A, c, bkg
            _m = (~_satmask)*(~_badmask)
            for ite in range(10):
                p, ier = opt.leastsq(errfunc, p0,
                            args=(xdata[_m], ydata[_m], interf))
                ydata_fit = fitfunc(p, xdata, interf)
                ydata_res = ydata - ydata_fit
                std = ydata_res[_m].std(ddof=1)
                _new_m = (np.abs(ydata_res) < 5*std)*_m
                if _new_m.sum() == _m.sum():
                    break
                _m = _new_m
                p0 = p
            snr = p[0]/std

            # p[0]: amplitude; p[1]: pos, p[2]:background
            succ = p[0]>0 and x1<p[1]<x2 and snr>5 and ier<=4

            if succ:
                if not is_first_correct:
                    is_first_correct = True
                    if y > 0.25*ny:
                        break_aperture = True
                        break
                fitpar_lst.append(p)
            else:
                fitpar_lst.append(blank_p)

        if break_aperture:
            message = ('Aperture {:3d}: Skipped because of '
                       'break_aperture=True').format(aper)
            logger.debug(message)
            print(message)
            continue

        fitpar_lst = np.array(fitpar_lst)

        if np.isnan(fitpar_lst[:,0]).sum()>0.5*ny:
            message = ('Aperture {:3d}: Skipped because of too many NaN '
                       'values in aperture parameters').format(aper)
            logger.debug(message)
            print(message)
            continue

        if (~np.isnan(fitpar_lst[:,0])).sum()<10:
            message = ('Aperture {:3d}: Skipped because of too few real '
                       'values in aperture parameters').format(aper)
            logger.debug(message)
            print(message)
            continue

        # pick up NaN positions in fitpar_lst and generate fitmask.
        # NaN = False. Real number = True
        fitmask = ~np.isnan(fitpar_lst[:,0])
        # divide the whole order into several groups
        xx = np.nonzero(fitmask)[0]
        group_lst = np.split(xx, np.where(np.diff(xx) > 4)[0]+1)
        # group_lst is composed of (x1, x2, ..., xN), where xi is index in
        # **newx_lst**
        # 4 means the maximum tolerance skipping value in fitmask is 3
        # filter out short segments
        # every index in group is index in newx_lst, NOT real pixel numbers
        group_lst = [group for group in group_lst
                     if newy_lst[group[-1]] - newy_lst[group[0]] > ny/10]

        if len(group_lst) == 0:
            message = ('Aperture {:3d}: Skipped'.format(aper))
            print(message)
            logger.debug(message)
            continue

        # loop for A, c, bkg. Smooth these parameters
        for ipara in range(3):
            ypara = fitpar_lst[:,ipara]

            if ipara == 0:
                # fit for A
                res = smooth_A_func(newy_lst, ypara, fitmask, group_lst, ny)
            elif ipara == 1:
                # fit for c
                res = smooth_c_func(newy_lst, ypara, fitmask, group_lst, ny)
            else:
                # fit for bkg
                res = smooth_bkg_func(newy_lst, ypara, fitmask, group_lst, ny)

            # extract smoothing results
            aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst = res

            # pack this parameter for every pixels
            aperpar_lst.append(aperpar)

            if plot_aperpar:
                ########### plot flat parametres every 5 orders ##############
                has_aperpar_fig = True
                i1, i2 = newy_lst[group_lst[0][0]], newy_lst[group_lst[-1][-1]]
                # plot the parameters

                # create ax1 for plotting parameters
                irow = iaper%5
                _x, _y = 0.04+ipara*0.32, (4-irow)*0.19+0.05
                ax1 = fig_aperpar.add_axes([_x, _y, 0.28, 0.17])

                # make a copy of ax1 and plot the residuals in the background
                ax2 = ax1.twinx()
                ax2.plot(xpiece_lst, ypiece_res_lst, color='gray', lw=0.5,
                        alpha=0.4, zorder=-2)
                ax2.axhline(y=0, color='gray', ls='--', lw=0.5,
                        alpha=0.4, zorder=-3)
                # plot rejected points with gray dots
                _m = mask_rej_lst>0
                if _m.sum()>0:
                    ax2.plot(xpiece_lst[_m], ypiece_res_lst[_m], 'o',
                            color='gray', lw=0.5, ms=2, alpha=0.4, zorder=-1)

                # plot data points
                ax1.plot(newy_lst, ypara, '-', color='C0', lw=0.5, zorder=1)
                # plot fitted value
                ax1.plot(ally[i1:i2], aperpar[i1:i2], '-', color='C1',
                    lw=1, alpha=0.8, zorder=2)

                #ax1.plot(newy_lst[~fitmask], ypara[~fitmask], 'o', color='C3',
                #        lw=0.5, ms=3, alpha=0.5)
                _y1, _y2 = ax1.get_ylim()
                if ipara == 0:
                    ax1.text(0.05*ny, 0.15*_y1+0.85*_y2, 'Aperture %d'%aper,
                            fontsize=10)
                ax1.text(0.9*ny, 0.15*_y1+0.85*_y2, 'ACB'[ipara], fontsize=10)

                # fill the fitting regions
                for group in group_lst:
                    i1, i2 = newy_lst[group[0]], newy_lst[group[-1]]
                    ax1.fill_betweenx([_y1, _y2], i1, i2, color='C0', alpha=0.1)

                ax1.set_xlim(0, ny-1)
                ax1.set_ylim(_y1, _y2)
                if iaper%5<4:
                    ax1.set_xticklabels([])

                for tick in ax1.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax2.yaxis.get_major_ticks():
                    tick.label2.set_fontsize(4)
                    tick.label2.set_color('gray')
                    tick.label2.set_alpha(0.6)
                for tickline in ax2.yaxis.get_ticklines():
                    tickline.set_color('gray')
                    tickline.set_alpha(0.6)
                if ny<3000:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(100))
                else:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(500))

                ########### plot flat parametres for every order ##############
                if False:
                    pass

        if plot_aperpar:
            # save and close the figure
            if iaper%5==4:
                fig_aperpar.savefig(figname_aperpar(aper))
                plt.close(fig_aperpar)
                has_aperpar_fig = False

        # find columns to be corrected in this order
        correct_y_lst = []
        for y in ally:
            pos    = aper_position[y]
            lbound = aper_lbound[y]
            ubound = aper_ubound[y]

            x1 = int(max(0,  lbound))
            x2 = int(min(nx, ubound))
            if (x2-x1)<5:
                continue
            xdata = np.arange(x1, x2)
            ydata = data[y, x1:x2]
            _satmask = sat_mask[y, x1:x2]
            _badmask = bad_mask[y, x1:x2]
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(nx, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-x1:_i2-x1])*nflat))
            if sn>q_threshold and _satmask.sum()<3 and _badmask.sum()<3:
                correct_y_lst.append(y)

        # find the left and right boundaries of the correction region
        y1, y2 = correct_y_lst[0], correct_y_lst[-1]

        # now loop over columns in correction region
        for y in correct_y_lst:
            interf = interprofilefunc_lst[y]
            pos    = aper_position[y]
            lbound = aper_lbound[y]
            ubound = aper_ubound[y]

            x1 = int(max(0,  lbound))
            x2 = int(min(nx, ubound))
            xdata = np.arange(x1, x2)
            ydata = data[y, x1:x2]
            _satmask = sat_mask[y, x1:x2]
            _badmask = bad_mask[y, x1:x2]

            # correct flat for this column
            A = aperpar_lst[0][y]
            c = aperpar_lst[1][y]
            b = aperpar_lst[2][y]

            lcorr, rcorr = corr_mask_array[y]
            normx = xdata-c
            corr_mask = (normx > lcorr)*(normx < rcorr)
            flat = ydata/fitfunc([A,c,b], xdata, interf)
            flatmask = corr_mask*~_satmask*~_badmask

            flatdata[y, x1:x2][flatmask] = flat[flatmask]

            # extract the 1d spectra of the modeled flat using super-sampling
            # integration
            x1s = max(0,  np.round(lbound-2, 1))
            x2s = min(nx, np.round(ubound+2, 1))
            xdata2 = np.arange(x1s, x2s, 0.1)
            flatmod = fitfunc([A,c,bkg], xdata2, interf)
            # use trapezoidal integration
            # np.trapz(flatmod, x=xdata2)
            # use simpson integration
            flatspec_lst[aper][y] = simps(flatmod, x=xdata2)

        t2 = time.time()
        message = ('Aperture {:3d}: {:2d} group{:1s}; '
                   'correct {:4d} pixels from {:4d} to {:4d}; '
                   't = {:6.1f} ms').format(
                    aper, len(group_lst), (' ','s')[len(group_lst)>1],
                    len(correct_y_lst),
                    correct_y_lst[0], correct_y_lst[-1],
                    (t2-t1)*1e3
                    )
        print(message)

    ###################### aperture loop ends here ########################
    if plot_aperpar and has_aperpar_fig:
        # there's unsaved figure in memory. save and close the figure
        fig_aperpar.savefig(figname_aperpar(aper))
        plt.close(fig_aperpar)
        has_aperpar_fig = False

    # pack the final 1-d spectra of flat
    flatspectable = [(aper, flatspec_lst[aper])
                     for aper in sorted(apertureset.keys())]

    # define the datatype of flat 1d spectra
    flatspectype = np.dtype(
                    {'names':   ['aperture', 'flux'],
                     'formats': [np.int32, (np.float32, ny)],}
                    )
    flatspectable = np.array(flatspectable, dtype=flatspectype)

    return flatdata, flatspectable

def get_flat(data, aperture_set, mode='normal'):
    """Get flat fielding for CFHT/ESPaDOnS data.

    Args:
        data ():
        aperture_set ():
        mode (str):

    """

    if mode == 'debug':
        dbgpath = 'debug'
        if not os.path.exists(dbgpath):
            os.mkdir(dbgpath)
        plot_aperpar = True
        figname_aperpar = lambda aper: os.path.join(dbgpath,
                                'aperpar_{:03d}.png'.format(aper))
        save_aperpar = True
        filename_aperpar = lambda aper: os.path.join(dbgpath,
                                'aperpar_{:03d}.dat'.format(aper))
    else:
        plot_aperpar = False
        save_aperpar = False

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    x_lst = {}
    A_lst = {}
    c_lst = {}
    b_lst = {}
   
    profilex_lst = []
    profilefunc_lst = []
    p1, p2 = -15, 15
    pstep = 0.5

    winsize = 128
    yc_lst = np.arange(winsize/2, ny, winsize)

    for iyc, yc in enumerate(yc_lst):
        yc = int(yc)
        # initialize the mean profiles
        all_xnodes = np.array([])
        all_ynodes = np.array([])
    
        fig = plt.figure(dpi=200)
        ax = fig.gca()
    
        miniyc_lst = np.arange(yc-48, yc+48+1e-3, 16)
        for iy, y in enumerate(miniyc_lst):
            y = int(y)
            if y == 0 or y == ny:
                continue


            order_cent_lst = np.array([aperloc.position(y)
                    for aper, aperloc in sorted(aperture_set.items())])
            order_dist_lst = np.diff(order_cent_lst)
            # now size of order_dist_lst = size of order_cent_lst - 1
            order_dist_lst = np.insert(order_dist_lst, 0, order_dist_lst[0])
            order_dist_lst = np.append(order_dist_lst, order_dist_lst[-1])
            # now size of order_dist_lst = size of order_cent_lst + 1
            order_hwin_lst = order_dist_lst/2
            # size of order_hwin_lst = size of order_cent_lst + 1
    
            for iaper, (aper, aperloc) in enumerate(
                    sorted(aperture_set.items())):
                cent = order_cent_lst[iaper]
                lwin = order_hwin_lst[iaper]
                rwin = order_hwin_lst[iaper+1]
    
                ceni = int(round(cent))
                i1 = cent - min(lwin, 18)
                i2 = cent + min(rwin, 18)
                i1, i2 = int(round(i1)), int(round(i2))
                i1 = max(i1, 0)
                i2 = min(i2, nx)
                if i2 - i1 < 10:
                    continue
                xnodes = np.arange(i1, i2)
                ynodes = data[y, i1:i2]

                results = norm_profile(xnodes, ynodes)
                # in case returns None results
                if results is None:
                    continue

                # unpack the results
                newx, newy, param = results

                # pack A and background to result list
                if aper not in x_lst:
                    x_lst[aper] = []
                    A_lst[aper] = []
                    c_lst[aper] = []
                    b_lst[aper] = []
    
                x_lst[aper].append(y)
                c_lst[aper].append(param[0]-cent)
                A_lst[aper].append(param[3])
                b_lst[aper].append(param[4])
    
                all_xnodes = np.append(all_xnodes, newx)
                all_ynodes = np.append(all_ynodes, newy)
    
                # plotting
                ax.scatter(newx, newy, s=1, alpha=0.1)

            # print a progress bar in terminal
            n_finished = iyc*6 + iy + 1
            n_total    = yc_lst.size*6
            ratio = min(n_finished/n_total, 1.0)
            term_size = os.get_terminal_size()
            nchar = term_size.columns - 60

            string = '>'*int(ratio*nchar)
            string = string.ljust(nchar, '-')
            prompt = 'Constructing slit function'
            string = '\r {:<30s} |{}| {:6.2f}%'.format(prompt, string, ratio*100)
            sys.stdout.write(string)
            sys.stdout.flush()
    
        all_xnodes = np.array(all_xnodes)
        all_ynodes = np.array(all_ynodes)
        xlst, ylst = get_mean_profile(all_xnodes, all_ynodes, p1, p2, pstep)
        # filter out NaN values
        m = np.isnan(ylst)
    
        # for plotting a smoothed curve in the figure
        f = intp.InterpolatedUnivariateSpline(xlst[~m], ylst[~m], k=3, ext=1)
        newx = np.arange(p1-2, p2+2+1e-5, 0.1)
        newy = f(newx)
        ax.plot(newx, newy, ls='-', color='k', lw=0.7)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-18, 18)
        ax.set_ylim(-0.2, 1.2)
        fig.savefig('slit_{:04d}.png'.format(yc))
        plt.close(fig)

        profilex = yc
        profilex_lst.append(profilex)
        profilefunc_lst.append(f(xlst))

    # use light green color
    print(' \033[92m Completed\033[0m')

    profilenode_lst = xlst
    profilex_lst = np.array(profilex_lst)
    profilefunc_lst = np.array(profilefunc_lst)
    npoints = profilefunc_lst.shape[1]

    interprofilefunc_lst = {}
    for y in ally:
        profile = np.zeros(npoints)
        for i in np.arange(npoints):
            f = intp.InterpolatedUnivariateSpline(
                    profilex_lst, profilefunc_lst[:, i], k=3)
            profile[i] = f(y)
        interprofilefunc = intp.InterpolatedUnivariateSpline(
                profilenode_lst, profile, k=3)
        interprofilefunc_lst[y] = interprofilefunc

    flatdata = np.ones_like(data, dtype=np.float32)
    flatspec_lst = {aper: np.full(ny, np.nan) for aper in aperture_set}

    ally = np.arange(0, ny)
    for iaper, (aper, aperloc) in enumerate(sorted(aperture_set.items())):

        # in debug mode, save aperpar in ascii files
        if save_aperpar:
            filename = filename_aperpar(aper)
            outfile = open(filename, 'w')
            for x, A, b, c in zip(x_lst[aper], A_lst[aper],
                                  b_lst[aper], c_lst[aper]):
                outfile.write('{:4d} {:+16.8e} {:+16.8e} {:+16.8e}'.format(
                                x, A, b, c)+os.linesep)
            outfile.close()
            title = 'aperpar for Aperture {:03d}'.format(aper)
            message = 'savefile: "{}": {}'.format(filename, title)
            logger.info(message)


        x_lst[aper] = np.array(x_lst[aper])
        A_lst[aper] = np.array(A_lst[aper])
        c_lst[aper] = np.array(c_lst[aper])
        b_lst[aper] = np.array(b_lst[aper])

        ypos = aperloc.position(ally)
        m = (ypos > 0)*(ypos < nx)
        newx = ally[m]
        newA = smooth_aperpar_A(x_lst[aper], A_lst[aper],
                                npoints=ny, newx=newx)
        newb = smooth_aperpar_bkg(x_lst[aper], b_lst[aper],
                                npoints=ny, newx=newx)
    
        ################## plot aper par ####################
        if plot_aperpar:
            fig = plt.figure(dpi=150, figsize=(8, 6))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(x_lst[aper], A_lst[aper], lw=0.5)
            ax2.plot(x_lst[aper], b_lst[aper], lw=0.5)
            ax1.plot(newx, newA, lw=0.5, color='C3')
            ax2.plot(newx, newb, lw=0.5, color='C3')
            for ax in fig.get_axes():
                ax.set_xlim(0, ny-1)
                ax.grid(True, ls='--')
                ax.set_axisbelow(True)
            ax1.set_ylabel('A')
            ax2.set_ylabel('background')
            title = 'Aperture Parameters for Aperture {:03d}'.format(aper)
            figname = figname_aperpar(aper)
            fig.savefig(figname)
            plt.close(fig)
            message = 'savefig: "{}": {}'.format(figname, title)
            logger.info(message)
    
        # find columns to be corrected
        ypos = aperloc.position(ally)
        correct_y_lst = []
        for y in ally:
            pos = ypos[y]
            correct_y_lst.append(y)

        # loop over rows in the correction region
        for x, A, b in zip(newx, newA, newb):
            c = aperloc.position(x)
            intc = int(round(c))
            x1 = max(intc-15, 0)
            x2 = min(intc+15, nx)
            #index = np.searchsorted(profilex_lst, x)
            #index = min(max(index, 1), profilex_lst.size-2)
            #i1 = index-1
            #i2 = index+2
            #interx_lst = profilex_lst[i1:i2]
            #intery_lst = np.zeros((3,x2-x1))
            #for i, profilex in enumerate(interx_lst):
            #    profilefunc = profilefunc_lst[profilex]
            #    intery_lst[i,:] = profilefunc(np.arange(x1, x2)-c)

            #intery_lst = np.zeros((profilex_lst.size, x2-x1))
            #for i, profilex in enumerate(profilex_lst):
            #    profilefunc = profilefunc_lst[profilex]
            #    intery_lst[i,:] = profilefunc(np.arange(x1, x2)-c)

            #profile = np.zeros(x2-x1)
            #for ix in np.arange(x2-x1):
            #    f = intp.InterpolatedUnivariateSpline(
            #            profilex_lst, intery_lst[:,ix], k=3)
            #    profile[ix] = f(x)

            interprofilefunc = interprofilefunc_lst[x]
            profile = interprofilefunc(np.arange(x1, x2)-c)
            newprofile = profile*A + b
            flatdata[x, x1:x2] = data[x, x1:x2]/newprofile
            densex = np.arange(x1, x2, 0.1)
            densep = interprofilefunc(densex-c)
            flatspec_lst[aper][x] = simps(densep*A+b, x=densex)

        ratio = min(iaper/(len(aperture_set)-1), 1.0)
        term_size = os.get_terminal_size()
        nchar = term_size.columns - 60

        string = '>'*int(ratio*nchar)
        string = string.ljust(nchar, '-')
        prompt = 'Calculating flat field'
        string = '\r {:<30s} |{}| {:6.2f}%'.format(prompt, string, ratio*100)
        sys.stdout.write(string)
        sys.stdout.flush()

    # use light green color
    print(' \033[92m Completed\033[0m')

    return flatdata, flatspec_lst

def smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, npoints):
    has_fringe_lst = []
    aperpar = np.array([np.nan]*npoints)
    xpiece_lst     = np.array([np.nan]*newx_lst.size)
    ypiece_res_lst = np.array([np.nan]*newx_lst.size)
    mask_rej_lst   = np.array([np.nan]*newx_lst.size)
    allx = np.arange(npoints)
    # the dtype of xpiece_lst and ypiece_lst is np.float64

    # first try, scan every segment. find fringe by checking the local maximum
    # points after smoothing. Meanwhile, save the smoothing results in case the
    # smoothing will be used afterwards.
    for group in group_lst:
        i1, i2 = group[0], group[-1]
        p1, p2 = newx_lst[i1], newx_lst[i2]
        m = fitmask[group]
        xpiece = newx_lst[group]
        ypiece = ypara[group]
        # now fill the NaN values in ypiece
        if (~m).sum() > 0:
            f = InterpolatedUnivariateSpline(xpiece[m], ypiece[m], k=3)
            ypiece = f(xpiece)
        # now xpiece and ypiece are ready

        _m = np.ones_like(ypiece, dtype=np.bool)
        for ite in range(3):
            f = InterpolatedUnivariateSpline(xpiece[_m], ypiece[_m], k=3)
            ypiece2 = f(xpiece)
            win_len = (11, 21)[ypiece2.size>23]
            ysmooth = savgol_filter(ypiece2, window_length=win_len, polyorder=3)
            res = ypiece - ysmooth
            std = res.std()
            _new_m = np.abs(res) < 3*std

            # prevent extrapolation at the boundaries
            if _new_m.size > 3:
                _new_m[0:3] = True
                _new_m[-3:] = True
            _new_m = _m*_new_m

            if _new_m.sum() == _m.sum():
                break
            _m = _new_m
        # now xpiece, ypiece, ypiece2, ysmooth, res, and _m have the same
        # lengths and meanings on their positions of elements

        f = InterpolatedUnivariateSpline(xpiece, ysmooth, k=3)
        _x = np.arange(p1, p2+1)

        aperpar[_x] = f(_x)
        xpiece_lst[group] = xpiece
        ypiece_res_lst[group] = res
        mask_rej_lst[group] = ~_m

        # find out if this order is affected by fringes, by checking the
        # distribution of local maximum points
        imax, ymax = get_local_minima(-ysmooth, window=5)
        if len(imax) > 0:
            x = xpiece[imax]
        else:
            x = []
        # determine how many pixels in each bin.
        # if w=4000, then 500 pix. if w=2000, then 250 pix.
        npixbin = npoints//8
        bins = np.linspace(p1, p2, int(p2-p1)//npixbin+2)
        hist, _ = np.histogram(x, bins)

        n_nonzerobins = np.nonzero(hist)[0].size
        n_zerobins = hist.size - n_nonzerobins

        if p2-p1 < npoints/8 or n_zerobins <= 1 or \
            n_zerobins < n_nonzerobins or n_nonzerobins >= 3:
            # there's fringe
            has_fringe = True
        else:
            # no fringe
            has_fringe = False
        has_fringe_lst.append(has_fringe)

    # use global polynomial fitting if this order is affected by fringe and the
    # following conditions are satisified
    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2 \
        and has_fringe_lst.count(True) == len(has_fringe_lst):
        # fit polynomial over the whole order

        # prepare xpiece and y piece
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # fit with polynomial
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)
        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst = ~_m
    else:
        # scan again
        # fit polynomial for every segment
        for group, has_fringe in zip(group_lst, has_fringe_lst):
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            if has_fringe:
                if   xspan > npoints/2: deg = 4
                elif xspan > npoints/4: deg = 3
                elif xspan > npoints/8: deg = 2
                else:                   deg = 1
            else:
                deg = 7
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, np.log(ypiece), deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)
            ypiece_fit = np.exp(ypiece_fit)
            ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.exp(np.polyval(coeff, ii/npoints))
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst


def smooth_aperpar_A_old(x, y, npoints, newx):
    #coeff, yfit, _, mask, _ = iterative_polyfit(x, y, deg=11)
    p1, p2 = x[0], x[-1]
    xspan = p2 - p1

    f = intp.InterpolatedUnivariateSpline(x, y, k=3)
    xnew = np.arange(x[0], x[-1]+1e-3)
    ynew = f(xnew)
    ysmooth, _, _, _ = iterative_savgol_filter(ynew, winlen=101, order=3,
                    upper_clip=3, lower_clip=3)
    imax, ymax = get_local_minima(-ysmooth, window=5)

    if len(imax)>0:
        xmax = xnew[imax]
    else:
        xmax = []

    npixbin = npoints//8
    bins = np.linspace(p1, p2, int(p2-p1)//npixbin+2)
    hist, _ = np.histogram(xmax, bins)

    n_nonzerobins = np.nonzero(hist)[0].size
    n_zerobins = hist.size - n_nonzerobins

    if n_zerobins <=1 or n_zerobins < n_nonzerobins or n_nonzerobins >=3:
        has_fringe = True
    else:
        has_fringe = False

    if has_fringe:
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1
    else:
        deg = 5

    # pick up the points with y>0
    posmask = (y>0)*(x<4200)

    coeff, _, _, m, std = iterative_polyfit(
            x[posmask]/npoints, np.log(y[posmask]), deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)
    aperpar = np.exp(np.polyval(coeff, newx/npoints))

    return aperpar

def smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *k* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    """

    allx = np.arange(npoints)

    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2:

        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # fit with polynomial
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*npoints)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]

            # determine the polynomial degree
            xspan = xpiece[-1] - xpiece[0]
            if   xspan > npoints/2: deg = 4
            elif xspan > npoints/4: deg = 3
            elif xspan > npoints/8: deg = 2
            else:                   deg = 1

            # fit with polynomial
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/npoints)
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *bkg* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
    """

    allx = np.arange(npoints)

    # fit for bkg
    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the degree of polynomial
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # polynomial fitting
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*npoints)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]

            # determine the degree of polynomial
            xspan = xpiece[-1] - xpiece[0]
            if   xspan > npoints/2: deg = 4
            elif xspan > npoints/4: deg = 3
            elif xspan > npoints/8: deg = 2
            else:                   deg = 1

            scale = ('linear','log')[(ypiece<=0).sum()==0]
            if scale=='log':
                ypiece = np.log(ypiece)

            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, ypiece, deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)

            if scale=='log':
                ypiece = np.exp(ypiece)
                ypiece_fit = np.exp(ypiece_fit)
                ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/npoints)
            if scale=='log':
                aperpar[ii] = np.exp(aperpar[ii])
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_bkg_old(x, y, npoints, newx):
    p1, p2 = x[0], x[-1]
    xspan = p2 - p1

    if   xspan > npoints/2: deg = 4
    elif xspan > npoints/4: deg = 3
    elif xspan > npoints/8: deg = 2
    else:                   deg = 1

    # pick up the points with y>0
    posmask = (y>0)*(x<4200)

    coeff, _, _, m, std = iterative_polyfit(
            x[posmask]/npoints, np.log(y[posmask]), deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)
    aperpar = np.exp(np.polyval(coeff, newx/npoints))
    return aperpar

