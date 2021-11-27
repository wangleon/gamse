import os
import sys
import time
import math
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.signal import savgol_filter
import scipy.optimize as opt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.backends.backend_pdf import PdfPages

from ...utils.onedarray import get_local_minima
from ...utils.regression import iterative_polyfit
from .common import norm_profile, ProfileNormalizer

class AperparSinglePlotter(object):
    def __init__(self, ndisp, plot=False):
        self.plot = plot
        self.ndisp = ndisp

    def update_figure(self):
        if self.plot:
            self.fig = plt.figure(figsize=(12, 3.5), dpi=200)

    def plot_aperpar(self, aper, ipara, newx_lst, ypara, group_lst,
            aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst):

        if self.plot:
            allx = np.arange(self.ndisp)
            i1, i2 = newx_lst[group_lst[0][0]], newx_lst[group_lst[-1][-1]]
            ax1 = self.fig.add_axes([0.05+ipara*0.32, 0.35, 0.27, 0.62])
            ax2 = self.fig.add_axes([0.05+ipara*0.32, 0.06, 0.27, 0.26])
            ax2.plot(xpiece_lst, ypiece_res_lst, color='k', lw=0.5,
                   alpha=0.4, zorder=-2)
            ax2.axhline(y=0, color='k', ls='--', lw=0.5,
                    alpha=0.4, zorder=-3)

            # plot rejected points with gray dots
            _m = mask_rej_lst>0
            if _m.sum()>0:
                ax2.plot(xpiece_lst[_m], ypiece_res_lst[_m], 'o',
                    color='gray', lw=0.5, ms=2, alpha=0.4, zorder=-1)

            ax1.plot(newx_lst, ypara, '-', color='C0', lw=1, zorder=1)
            ax1.plot(allx[i1:i2], aperpar[i1:i2], '-', color='C1',
                lw=1, alpha=0.8, zorder=2)

            _y1, _y2 = ax1.get_ylim()
            if ipara == 0:
                ax1.text(0.05*self.ndisp, 0.15*_y1+0.85*_y2,
                        'Aperture {}'.format(aper), fontsize=10)

            ax1.text(0.9*self.ndisp, 0.15*_y1+0.85*_y2,
                    'ACB'[ipara], fontsize=10)
            ax1.set_xlim(0, self.ndisp-1)
            ax1.set_ylim(_y1, _y2)

            for tick in ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(10)
            for tick in ax1.yaxis.get_major_ticks():
                tick.label1.set_fontsize(10)
            for tick in ax2.yaxis.get_major_ticks():
                tick.label2.set_fontsize(10)
            if self.ndisp<3000:
                ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
                ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
                ax2.xaxis.set_major_locator(tck.MultipleLocator(500))
                ax2.xaxis.set_minor_locator(tck.MultipleLocator(100))
            else:
                ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
                ax1.xaxis.set_minor_locator(tck.MultipleLocator(500))
                ax2.xaxis.set_major_locator(tck.MultipleLocator(1000))
                ax2.xaxis.set_minor_locator(tck.MultipleLocator(500))
            ax1.set_xticklabels([])
            ax1.set_xlim(0, self.ndisp-1)
            ax2.set_xlim(0, self.ndisp-1)

    def savefig(self, figname):
        if self.plot:
            self.fig.savefig(figname)
            plt.close(self.fig)

class AperparPlotter(object):
    def __init__(self, ndisp, figname):
        self.ndisp = ndisp
        self.pdf = PdfPages(figname)
        self.allx = np.arange(self.ndisp)

    def close(self):
        self.pdf.close()

    def plot_aperpar(self, aper, ipara, newx_lst, ypara, group_lst,
            aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst):

        if ipara==0:
            self.fig = plt.figure(figsize=(14.14, 10), dpi=200)

        i1, i2 = newx_lst[group_lst[0][0]], newx_lst[group_lst[-1][-1]]

        # for 5-row plots
        ## create ax1 for plotting parameters
        #irow = self.iaper%5
        #_x, _y = 0.04+ipara*0.32, (4-irow)*0.19+0.05
        #ax1 = self.fig.add_axes([_x, _y, 0.28, 0.17])

        ax1 = self.fig.add_axes([0.06, 0.1+(3-ipara)*0.32, 0.82, 0.27])

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
        ax1.plot(newx_lst, ypara, '-', color='C0', lw=0.5, zorder=1)
        # plot fitted value
        ax1.plot(self.allx[i1:i2], aperpar[i1:i2], '-', color='C1',
            lw=1, alpha=0.8, zorder=2)

        _y1, _y2 = ax1.get_ylim()
        if ipara == 0:
            ax1.text(0.05*self.ndisp, 0.15*_y1+0.85*_y2,
                    'Aperture {}'.format(aper), fontsize=10)

        ax1.text(0.9*self.ndisp, 0.15*_y1+0.85*_y2,
                'ACB'[ipara], fontsize=10)

        # fill the fitting regions
        for group in group_lst:
            i1, i2 = newx_lst[group[0]], newx_lst[group[-1]]
            ax1.fill_betweenx([_y1, _y2], i1, i2, color='C0', alpha=0.1)

        ax1.set_xlim(0, self.ndisp-1)
        ax1.set_ylim(_y1, _y2)
        #if self.iaper%5<4:
        #    ax1.set_xticklabels([])

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
        if self.ndisp<3000:
            ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax2.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax2.xaxis.set_minor_locator(tck.MultipleLocator(100))
        else:
            ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
            ax1.xaxis.set_minor_locator(tck.MultipleLocator(500))
            ax2.xaxis.set_major_locator(tck.MultipleLocator(1000))
            ax2.xaxis.set_minor_locator(tck.MultipleLocator(500))

        if ipara==2:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)


def get_flat(data, mask, apertureset, nflat,
        #smooth_A_func, smooth_c_func, smooth_bkg_func,
        slit_step = 0,
        q_threshold=30, mode='normal',
        fig_spatial=None,
        flatname = '',
        ):
    """ Get flat.
    """

    if mode == 'debug':
        dbgpath = 'debug'
        if not os.path.exists(dbgpath):
            os.mkdir(dbgpath)
        plot_aperpar = True
        
        figname_aperpar = os.path.join(dbgpath,
                            'aperpar_{}.pdf'.format(flatname))
        save_aperpar = True
        filename_aperpar = os.path.join(dbgpath,
                            'aperpar_{}.dat'.format(flatname))

    else:
        plot_aperpar = False
        save_aperpar = False

    ny, nx = data.shape
    allx = np.arange(nx)

    # calculate order positions and boundaries and save them in dicts
    all_positions  = apertureset.get_positions(allx)
    all_boundaries = apertureset.get_boundaries(allx)

    p1, p2, pstep = -10, 10, 0.1
    profile_x = np.arange(p1+1, p2-1+1e-4, pstep)
    profilex_lst = []
    profiley_lst = []

    # find saturation mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    x0 = 32
    winsize = 400
    xc_lst = np.arange(x0, nx, winsize)
    # n = 6

    fig_show = plt.figure(figsize=(12, 3), dpi=200)

    for ixc, xc in enumerate(xc_lst):
        xc = int(xc)
        x = xc
        # initialize the mean profiles
        all_xnodes = []
        all_ynodes = []

        ax = fig_spatial.get_axes()[ixc]

        # has 6 panels, draw 0, 3, 5
        if ixc==0:
            _x = 0.05
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        elif ixc==3:
            _x = 0.05 + 1*0.32
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        elif ixc==5:
            _x = 0.05 + 2*0.32
            ax2 = fig_show.add_axes([_x, 0.07, 0.28, 0.9])
        else:
            ax2 = None

        #minixc_lst = np.arange(xc-8, xc+8+1e-3, 8)
        #for ix, x in enumerate(minixc_lst):
        #    x = int(x)
        #    if x == 0 or x == nx:
        #        continue

        order_cent_lst = np.array([aperloc.position(x)
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
            i2 = min(i2, ny)
            if i2 - i1 < 10:
                continue

            xnodes = np.arange(i1, i2)
            ynodes = data[i1:i2, x]
            mnodes = mask[i1:i2, x]
            if np.nonzero(mnodes>0)[0].size>3:
                continue

            normed_prof = ProfileNormalizer(xnodes, ynodes, mnodes)


            newx = normed_prof.x
            newy = normed_prof.y
            newm = normed_prof.m
            '''
            param = normed_prof.param
            std   = normed_prof.std
            A, center, sigma, bkg = param

            fig0 = plt.figure()
            ax01 = fig0.add_subplot(211)
            ax02 = fig0.add_subplot(212)
            ax01.plot(xnodes, ynodes, 'o')
            ax01.plot(xnodes[mnodes>0], ynodes[mnodes>0], 'x')
            plotx, ploty = normed_prof.linspace()
            label = 'A={:.2f}\nc={:.1f}\ns={:.2f}\nbkg={:.2f}\nstd={:.2f}'.format(
                    param[0], param[1], param[2], param[3], std)
            if normed_prof.is_succ():
                ls = '-'
            else:
                ls = '--'
            ax01.plot(plotx, ploty, ls=ls, label=label)
            ax02.plot(newx, newy, 'o', mec='C0', mfc='none')
            ax02.plot(newx[newm], newy[newm], 'o', c='C0')
            ax01.legend()
            figname = 'test-x_{:04d}_y_{:04d}_{:04d}.png'.format(x, i1, i2)
            if flatname is not None:
                figname = '{}_{}'.format(flatname, figname)
            figname = os.path.join(dbgpath, figname)
            fig0.savefig(figname)
            plt.close(fig0)
            '''

            if normed_prof.is_succ():
                for _newx, _newy in zip(newx[newm], newy[newm]):
                    all_xnodes.append(_newx)
                    all_ynodes.append(_newy)

                # plotting
                ax.scatter(newx[newm], newy[newm], s=5, alpha=0.3, lw=0)
                if ax2 is not None:
                    ax2.scatter(newx[newm], newy[newm], s=10, alpha=0.3, lw=0)


        # print a progress bar in terminal
        #n_finished = ixc*6 + ix + 1
        #n_total    = xc_lst.size*6
        n_finished = ixc + 1
        n_total    = xc_lst.size
        ratio = min(n_finished/n_total, 1.0)
        term_size = os.get_terminal_size()
        nchar = term_size.columns - 60

        string = '>'*int(ratio*nchar)
        string = string.ljust(nchar, '-')
        prompt = 'Constructing spatial profile'
        string = '\r {:<30s} |{}| {:6.2f}%'.format(prompt, string, ratio*100)
        sys.stdout.write(string)
        sys.stdout.flush()

        all_xnodes = np.array(all_xnodes)
        all_ynodes = np.array(all_ynodes)

        # spatial profile modeling with Gaussian Process Regression (GPR)
        input_train_x = all_xnodes.reshape(-1,1)
        input_train_y = all_ynodes.reshape(-1,1)
        kernel = RBF(length_scale=3.0) + WhiteKernel(0.1)
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(input_train_x, input_train_y)
        test_x = profile_x.reshape(-1,1)
        mu, cov = gpr.predict(test_x, return_cov=True)
        profile_y = mu.flatten()

        # plotting
        ax.plot(profile_x, profile_y, color='k', lw=0.6)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_axisbelow(True)
        _x1, _x2 = p1-2, p2+2
        _y1, _y2 = -0.2, 1.2
        _text = 'X={:4d}/{:4d}'.format(xc, nx)
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
            _text = 'X={:4d}/{:4d}'.format(xc, nx)
            ax2.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2, _text)
            ax2.set_xlim(_x1, _x2)
            ax2.set_ylim(_y1, _y2)

        profilex_lst.append(xc)
        profiley_lst.append(profile_y)

    # use light green color
    print(' \033[92m Completed\033[0m')

    fig_show.savefig('spatialprofile_foces.pdf')
    plt.close(fig_show)

    profilex_lst = np.array(profilex_lst)
    profiley_lst = np.array(profiley_lst)
    npoints = profiley_lst.shape[1]

    interprofilefunc_lst = {}
    corr_mask_array = []
    for x in allx:
        # calculate interpolated profie in this x
        profile = np.zeros(npoints)
        for i in np.arange(npoints):
            f = InterpolatedUnivariateSpline(
                    profilex_lst, profiley_lst[:, i], k=3, ext=3)
            profile[i] = f(x)
        interprofilefunc = InterpolatedUnivariateSpline(
                profile_x, profile, k=3, ext=3)
        interprofilefunc_lst[x] = interprofilefunc
        
        ilst = np.nonzero(profile>0.1)[0]
        il = profile_x[ilst[0]]
        ir = profile_x[ilst[-1]]
        corr_mask_array.append((il, ir))

    ##################

    flatdata = np.ones_like(data, dtype=np.float32)
    flatspec_lst = {aper: np.full(nx, np.nan) for aper in apertureset}

    # define fitting and error functions
    def fitfunc(param, xdata, interf):
        A, c, b = param
        return A*interf(xdata-c) + b
    def errfunc(param, xdata, ydata, interf):
        return ydata - fitfunc(param, xdata, interf)

    # prepare an x list
    newx_lst = np.arange(0, nx-1, 10)
    if newx_lst[-1] != nx-1:
        newx_lst = np.append(newx_lst, nx-1)


    ###################### loop for every aperture ########################

    # define the function of refreshing the second progress bar
    def refresh_progressbar2(iaper):
        ratio = min(iaper/(len(apertureset)-1), 1.0)
        term_size = os.get_terminal_size()
        nchar = term_size.columns - 60
        string = '>'*int(ratio*nchar)
        string = string.ljust(nchar, '-')
        prompt = 'Calculating flat field'
        string = '\r {:<30s} |{}| {:6.2f}%'.format(prompt, string, ratio*100)
        sys.stdout.write(string)
        sys.stdout.flush()

    if plot_aperpar:
        #aperpar_plotter = AperparPlotter(ndisp=nx, figname=figname_aperpar)
        aperpar_plotter = PdfPages(figname_aperpar)

    for iaper, (aper, aperloc) in enumerate(sorted(apertureset.items())):
        fitpar_lst  = [] # stores (A, c, b). it has the same length as newx_lst
        aperpar_lst = []

        # plot aperpar for specified orders
        aperpar_single_plotter = AperparSinglePlotter(
                                   plot=(aper in [83, 21]), ndisp=nx)
        aperpar_single_plotter.update_figure()

        aper_position = all_positions[aper]
        aper_lbound, aper_ubound = all_boundaries[aper]

        t1 = time.time()

        is_first_correct = False
        break_aperture = False

        # loop for every newx. find the fitting parameters for each column
        # prepar the blank parameter for insert
        blank_p = np.array([np.NaN, np.NaN, np.NaN])

        for ix, x in enumerate(newx_lst):
            pos      = aper_position[x]
            # skip this column if central position excess the CCD range
            if pos<0 or pos>ny:
                fitpar_lst.append(blank_p)
                continue

            # determine lower and upper bounds
            lbound = aper_lbound[x]
            ubound = aper_ubound[x]
            y1 = int(max(0,  lbound))
            y2 = int(min(ny, ubound))

            if y2-y1<=4:
                fitpar_lst.append(blank_p)
                continue

            # construct fitting data (xdata, ydata)
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]

            # calculate saturation mask and bad-pixel mask
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            # skip this column if too many saturated or bad pixels
            if _satmask.sum()>=3 or _badmask.sum()>=3:
                fitpar_lst.append(blank_p)
                continue
            # estimate the SNR
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(ny, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))
            # skip this column if sn is too low
            if sn < q_threshold:
                fitpar_lst.append(blank_p)
                continue

            # begin fitting
            interf = interprofilefunc_lst[x]
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
            succ = p[0]>0 and y1<p[1]<y2 and snr>5 and ier<=4

            if succ:
                if not is_first_correct:
                    is_first_correct = True
                    if x > 0.25*nx:
                        break_aperture = True
                        break
                fitpar_lst.append(p)
            else:
                fitpar_lst.append(blank_p)

        if break_aperture:
            message = ('Aperture {:3d}: Skipped because of '
                       'break_aperture=True').format(aper)
            logger.info(message)
            refresh_progressbar2(iaper)
            continue

        fitpar_lst = np.array(fitpar_lst)

        if np.isnan(fitpar_lst[:,0]).sum()>0.5*nx:
            message = ('Aperture {:3d}: Skipped because of too many NaN '
                       'values in aperture parameters').format(aper)
            logger.info(message)
            refresh_progressbar2(iaper)
            continue

        if (~np.isnan(fitpar_lst[:,0])).sum()<10:
            message = ('Aperture {:3d}: Skipped because of too few real '
                       'values in aperture parameters').format(aper)
            logger.info(message)
            refresh_progressbar2(iaper)
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
                     if newx_lst[group[-1]] - newx_lst[group[0]] > nx/10]

        if len(group_lst) == 0:
            message = ('Aperture {:3d}: Skipped'.format(aper))
            logger.info(message)
            refresh_progressbar2(iaper)
            continue

        # loop for A, c, bkg. Smooth these parameters
        if plot_aperpar:
            fig = plt.figure(figsize=(10, 14.14), dpi=100)

        for ipara in range(3):
            ypara = fitpar_lst[:,ipara]

            if ipara == 0:
                # fit for A
                res = smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, nx)
            elif ipara == 1:
                # fit for c
                res = smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, nx)
            else:
                # fit for bkg
                res = smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, nx)

            # extract smoothing results
            aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst = res

            # pack this parameter for every pixels
            aperpar_lst.append(aperpar)

            # plot aperpar
            if plot_aperpar:
                #aperpar_plotter.plot_aperpar(aper, ipara, newx_lst, ypara,
                #        group_lst, aperpar, xpiece_lst,
                #        ypiece_res_lst, mask_rej_lst)
                aperpar_single_plotter.plot_aperpar(aper, ipara, newx_lst,
                        ypara, group_lst, aperpar, xpiece_lst, ypiece_res_lst,
                        mask_rej_lst)
                ##############################
                allx = np.arange(nx)
                i1, i2 = newx_lst[group_lst[0][0]], newx_lst[group_lst[-1][-1]]
                ax1 = fig.add_axes([0.06, 0.06+(2-ipara)*0.30, 0.82, 0.26])
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
                ax1.plot(newx_lst, ypara, '-', color='C0', lw=0.5, zorder=1)
                # plot fitted value
                ax1.plot(allx[i1:i2], aperpar[i1:i2], '-', color='C1',
                    lw=1, alpha=0.8, zorder=2)
         
                _y1, _y2 = ax1.get_ylim()
                if ipara == 0:
                    ax1.text(0.05*nx, 0.15*_y1+0.85*_y2,
                            'Aperture {}'.format(aper), fontsize=10)
         
                ax1.text(0.9*nx, 0.15*_y1+0.85*_y2,
                        'ACB'[ipara], fontsize=10)
         
                # fill the fitting regions
                for group in group_lst:
                    i1, i2 = newx_lst[group[0]], newx_lst[group[-1]]
                    ax1.fill_betweenx([_y1, _y2], i1, i2, color='C0', alpha=0.1)
         
                ax1.set_xlim(0, nx-1)
                ax1.set_ylim(_y1, _y2)
                #if self.iaper%5<4:
                #    ax1.set_xticklabels([])
         
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
                if nx<3000:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(100))
                else:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(500))

                ##################################
        if plot_aperpar:
            aperpar_plotter.savefig(fig)
            plt.close(fig)

        # save aperpar figures
        #aperpar_plotter.close()
        #aperpar_plotter.savefig(figname_aperpar)
        #aperpar_single_plotter.savefig('aperpar2var_{}.png'.format(aper))

        '''
        # find columns to be corrected in this order
        correct_x_lst = []
        for x in allx:
            pos    = aper_position[x]
            lbound = aper_lbound[x]
            ubound = aper_ubound[x]

            y1 = int(max(0,  lbound))
            y2 = int(min(ny, ubound))
            if (y2-y1)<5:
                continue
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(ny, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))
            if sn>q_threshold and _satmask.sum()<3 and _badmask.sum()<3:
                correct_x_lst.append(x)

        # find the left and right boundaries of the correction region
        x1, x2 = correct_x_lst[0], correct_x_lst[-1]
        '''
        correct_x_lst = allx

        # now loop over columns in correction region
        for x in correct_x_lst:
            interf = interprofilefunc_lst[x]
            pos    = aper_position[x]
            lbound = aper_lbound[x]
            ubound = aper_ubound[x]

            y1 = int(max(0,  lbound))
            y2 = int(min(ny, ubound))
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]

            # correct flat for this column
            A = aperpar_lst[0][x]
            c = aperpar_lst[1][x]
            b = aperpar_lst[2][x]

            lcorr, rcorr = corr_mask_array[x]
            normx = xdata-c
            corr_mask = (normx > lcorr)*(normx < rcorr)
            flat = ydata/fitfunc([A,c,b], xdata, interf)
            flatmask = corr_mask*~_satmask*~_badmask

            flatdata[y1:y2, x][flatmask] = flat[flatmask]

            ## for debug
            #if aper==11 and x>400 and x<600:
            #    print(x, A, c, b, fitfunc([A,c,b], xdata, interf))
            #    print(corr_mask)
            #    print(_satmask)
            #    print(_badmask)
            #    print(flat[flatmask])


            # extract the 1d spectra of the modeled flat using super-sampling
            # integration
            y1s = max(0,  np.round(lbound-2, 1))
            y2s = min(ny, np.round(ubound+2, 1))
            xdata2 = np.arange(y1s, y2s, 0.1)
            flatmod = fitfunc([A,c,b], xdata2, interf)
            # use trapezoidal integration
            # np.trapz(flatmod, x=xdata2)
            # use simpson integration
            flatspec_lst[aper][x] = simps(flatmod, x=xdata2)

        t2 = time.time()
        message = ('Aperture {:3d}: {:2d} group{:1s}; '
                   'correct {:4d} pixels from {:4d} to {:4d}; '
                   't = {:6.1f} ms').format(
                    aper, len(group_lst), (' ','s')[len(group_lst)>1],
                    len(correct_x_lst),
                    correct_x_lst[0], correct_x_lst[-1],
                    (t2-t1)*1e3
                    )
        logger.info(message)
        #print(message)

        refresh_progressbar2(iaper)

    aperpar_plotter.close()
    # use light green color
    print(' \033[92m Completed\033[0m')

    ###################### aperture loop ends here ########################

    return flatdata, flatspec_lst

def smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *A* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    
    """

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
        # if npoints=4000, then 500 pix. if w=2000, then 250 pix.
        npixbin = npoints//8
        bins = np.linspace(p1, p2, int(p2-p1)//npixbin+2)
        # now the whole order is splitted into aproximately 8 bins
        hist, _ = np.histogram(x, bins)
        # hist is the number of local maxima points (LMP) in each bin

        n_nonzerobins = np.nonzero(hist)[0].size
        n_zerobins = hist.size - n_nonzerobins
        # n_nonzerobins is the number of bins with local maxima points (LMP)
        # n_zerobins is the number of bins without local maxima points (LMP)

        if p2-p1 < npoints/8 or n_zerobins <= 1 or \
            n_zerobins < n_nonzerobins or n_nonzerobins >= 3:
            # there's fringe if bins without LMP are less than 2, or
            # bins with LMP is more than without LMP, or
            # bins with LMP are more than 3
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

def smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *k* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

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

def smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *c* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_bkg`
    """
    return smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, npoints)

def smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *bkg* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

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

            # polynomial fitting
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


def gaussian_bkg(A, center, sigma, bkg, x):
    return A*np.exp(-(x-center)**2/2./sigma**2) + bkg
