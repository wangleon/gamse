import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt

import astropy.io.fits as fits

from ...utils.regression import iterative_polyfit
from ...utils.onedarray import iterative_savgol_filter, get_local_minima
from .common import norm_profile

def get_flat(data, aperture_set):
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

    hwidth = 256
    for x1 in np.arange(0, ny, hwidth):
        x2 = x1 + hwidth
    
        # initialize the mean profiles
        all_xnodes = np.array([])
        all_ynodes = np.array([])
    
        fig = plt.figure(dpi=200)
        ax = fig.gca()
    
        for x in np.arange(x1, x2, 16):
        #for x in np.arange((x1+x2)//2-8, (x1+x2)//2+8, 1):
            # skip the first column
            if x == 0:
                continue
    
            print(x)
            order_cent_lst = np.array([aperloc.position(x)
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
                ynodes = data[x, i1:i2]
    
                newx, newy, param = norm_profile(xnodes, ynodes)
    
                # pack A and background to result list
                if aper not in x_lst:
                    x_lst[aper] = []
                    A_lst[aper] = []
                    c_lst[aper] = []
                    b_lst[aper] = []
    
                x_lst[aper].append(x)
                c_lst[aper].append(param[0]-cent)
                A_lst[aper].append(param[3])
                b_lst[aper].append(param[4])
    
                all_xnodes = np.append(all_xnodes, newx)
                all_ynodes = np.append(all_ynodes, newy)
    
                # plotting
                ax.scatter(newx, newy, s=1, alpha=0.1)

    
        all_xnodes = np.array(all_xnodes)
        all_ynodes = np.array(all_ynodes)
        xlst, ylst = get_mean_profile(all_xnodes, all_ynodes, p1, p2, pstep)
    
        # for plotting a smoothed curve in the figure
        f = intp.InterpolatedUnivariateSpline(xlst, ylst, k=3, ext=1)
        newx = np.arange(p1-2, p2+2+1e-5, 0.1)
        newy = f(newx)
        ax.plot(newx, newy, ls='-', color='k', lw=0.7)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-18, 18)
        ax.set_ylim(-0.2, 1.2)
        fig.savefig('slit_{:04d}-{:04d}.png'.format(x1, x2))
        plt.close(fig)

        profilex = (x1+x2)//2
        profilex_lst.append(profilex)
        profilefunc_lst.append(ylst)

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
    flatspec = {}

    ally = np.arange(0, ny)
    for aper, aperloc in sorted(aperture_set.items()):
        '''
        outfile = open('aperpar_{:03d}.dat'.format(aper), 'w')
        for x, A, b, c in zip(x_lst[aper], A_lst[aper],
                              b_lst[aper], c_lst[aper]):
            outfile.write('{:4d} {:+16.8e} {:+16.8e} {:+16.8e}'.format(
                            x, A, b, c)+os.linesep)
        outfile.close()
        '''
        x_lst[aper] = np.array(x_lst[aper])
        A_lst[aper] = np.array(A_lst[aper])
        c_lst[aper] = np.array(c_lst[aper])
        b_lst[aper] = np.array(b_lst[aper])


        ypos = aperloc.position(ally)
        m = (ypos > 0)*(ypos < nx)
        newx = ally[m]
        newA = smooth_aperpar_A(x_lst[aper], A_lst[aper], npoints=ny, newx=newx)
        newb = smooth_aperpar_bkg(x_lst[aper], b_lst[aper], npoints=ny, newx=newx)
    
        ######### plot aper par #########
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
        fig.savefig('aperpar_{:03d}.png'.format(aper))
        plt.close(fig)
    
        # find columns to be corrected
        ypos = aperloc.position(ally)
        correct_y_lst = []
        for y in ally:
            pos = ypos[y]
            correct_y_lst.append(y)


        flatspec[aper] = np.zeros(ny, dtype=np.float32)

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
            print(aper, x, x1, x2)
            newprofile = profile*A + b
            flatdata[x, x1:x2] = data[x, x1:x2]/newprofile
            densex = np.arange(x1, x2, 0.1)
            densep = interprofilefunc(densex-c)
            flatspec[aper][x] = simps(densep*A+b, x=densex)

    # pack the final 1-d spectra of flat
    flatspectable = [(aper, flatspec[aper])
                     for aper in sorted(aperture_set.keys())]

    # define the datatype of flat 1d spectra
    flatspectype = np.dtype(
                    {'names':   ['aperture', 'flux'],
                     'formats': [np.int32, (np.float32, ny)],}
                    )
    flatspectable = np.array(flatspectable, dtype=flatspectype)

    hdu_lst = fits.HDUList([
                fits.PrimaryHDU(flatdata),
                fits.BinTableHDU(flatspectable),
            ])
    hdu_lst.writeto('flatnew.fits', overwrite=True)


    fig = plt.figure()
    ax = fig.gca()
    for aper, aperloc in aperture_set.items():
        newx = np.arange(0, ny)
        ax.plot(newx, aperloc.position(newx), '-')
    plt.show()



def smooth_aperpar_A(x, y, npoints, newx):
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

def smooth_aperpar_bkg(x, y, npoints, newx):
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

