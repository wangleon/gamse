
import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import matplotlib.pyplot as plt

from ...utils.onedarray import get_local_minima
from ...echelle.trace import ApertureSet, ApertureLocation

def gaussian(A, c, s, x):
    return A*np.exp(-(x-c)**2/s**2/2)
def fitfunc(p, x):
    A1, c1, s1, A2, c2, s2, b = p
    return gaussian(A1, c1, s1, x) + gaussian(A2, c2, s2, x) + b
def errfunc(p, x, y):
    return y - fitfunc(p, x)

def find_double_peak(x, y):
    """Find positions of peak for ESPRESSO dual-peak profile.
    """

    n = x.size
    y1 = y[0:n//2]
    y2 = y[n//2:]
    A01 = y1.max() - y1.min()
    A02 = y2.max() - y2.min()
    c01 = y1.argmax() + x[0]
    c02 = y2.argmax() + x[n//2]
    p0 = [A01, c01, 2.0, A02, c02, 2.0, y.min()]

    fitres = opt.least_squares(errfunc, p0, args=(x, y))
    p = fitres['x']
    A1, c1, s1, A2, c2, s2, b = p

    '''
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'o', ms=4, alpha=0.7)
    newx = np.linspace(x[0], x[-1], 500)
    newy = fitfunc(p, newx)
    ax.plot(newx, newy, '-')
    ax.axvline(c1, ls='--')
    ax.axvline(c2, ls='--')
    plt.show()
    plt.close(fig)
    '''

    return A1, c1, s1, A2, c2, s2, b

def find_apertures(data, scan_step=100, align_deg=2, separation=20,
        minimum=5, 
        fig_align=None, channel='b'):
    fig_align = plt.figure()
    fig_align.ax1 = fig_align.add_subplot(121)
    fig_align.ax2 = fig_align.add_subplot(122)

    fig_trace = plt.figure()
    fig_trace.ax1 = fig_trace.add_subplot(131)
    fig_trace.ax2 = fig_trace.add_subplot(132)
    fig_trace.ax3 = fig_trace.add_subplot(133)

    mode = 'debug'
    plot_alignfit  = False
    plot_orderalign = True
    plot_allorders = True
    figname_orderalign = 'order_alignment.png'
    figname_allorders  = 'order_all.png'

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    logdata = np.log10(np.maximum(data, minimum))

    try:
        separation = float(separation)
    except:
        x_lst, y_lst = [], []
        for substring in separation.split(','):
            g = substring.split(':')
            x, y = float(g[0]), float(g[1])
            x_lst.append(x)
            y_lst.append(y)
        x_lst = np.array(x_lst)
        y_lst = np.array(y_lst)
        # sort x and y
        index = x_lst.argsort()
        x_lst = x_lst[index]
        y_lst = y_lst[index]
        separation = Polynomial.fit(x_lst, y_lst, deg=x_lst.size-1)

    # initialize fsep as an ufunc.
    if isinstance(separation, int) or isinstance(separation, float):
        func = lambda x: separation
        # convert it to numpy ufunc
        fsep = np.frompyfunc(func, 1, 1)
    elif isinstance(separation, Polynomial):
        # separation is alreay a numpy ufunc
        fsep = separation
    else:
        print('cannot understand the meaning of separation')
        exit()


    def forward(x, p):
        deg = len(p)-1
        res = p[0]
        for i in range(deg):
            res = res*x + p[i+1]
        return res
    def forward_der(x, p):
        deg = len(p)-1
        p_der = [(deg-i)*p[i] for i in range(deg)]
        return forward(x, p_der)
    def backward(y, p):
        x = y
        for ite in range(20):
            dy = forward(x, p) - y
            y_der = forward_der(x, p)
            dx = dy/y_der
            x = x - dx
            if (np.abs(dx) < 1e-7).all():
                break
        return x
    def fitfunc(p, interfunc, n):
        #return p[-2]*interfunc(forward(np.arange(n), p[0:-2]))+p[-1]
        return interfunc(forward(np.arange(n), p[0:-1]))+p[-1]
    def resfunc(p, interfunc, flux0, mask=None):
        res_lst = flux0 - fitfunc(p, interfunc, flux0.size)
        if mask is None:
            mask = np.ones_like(flux0, dtype=bool)
        return res_lst[mask]
    def find_shift(flux0, flux1, deg):
        #p0 = [1.0, 0.0, 0.0]
        #p0 = [0.0, 1.0, 0.0, 0.0]
        #p0 = [0.0, 0.0, 1.0, 0.0, 0.0]

        p0 = [0.0 for i in range(deg+1)]
        p0[-3] = 1.0

        interfunc = intp.InterpolatedUnivariateSpline(
                    np.arange(flux1.size), flux1, k=3, ext=3)
        mask = np.ones_like(flux0, dtype=bool)
        clipping = 5.
        for i in range(10):
            p, _ = opt.leastsq(resfunc, p0, args=(interfunc, flux0, mask))
            res_lst = resfunc(p, interfunc, flux0)
            std  = res_lst.std()
            mask1 = res_lst <  clipping*std
            mask2 = res_lst > -clipping*std
            new_mask = mask1*mask2
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask
        return p, mask
    def find_local_peak(xdata, ydata, mask, smooth=None, figname=None):
        if figname is not None:
            fig = plt.figure(dpi=150, figsize=(8,6))
            ax = fig.gca()
            ax.plot(xdata, ydata, color='C0')
        if smooth is not None:
            core = np.hanning(min(smooth, ydata.size))
            core = core/core.sum()
            # length of core should not be smaller than length of ydata
            # otherwise the length of ydata after convolution is reduced
            ydata = np.convolve(ydata, core, mode='same')
        n = x.size
        y1 = y[0:n//2]
        y2 = y[n//2:]
        A01 = y1.max() - y1.min()
        A02 = y2.max() - y2.min()
        c01 = y1.argmax() + x[0]
        c02 = y2.argmax() + x[n//2]
        p0 = [A01, c01, 2.0, A02, c02, 2.0, y.min()]

        fitres = opt.least_squares(errfunc, p0, args=(x, y))
        p = fitres['x']
        A1, c1, s1, A2, c2, s2, b = p

        return c1, c2



    x0 = ny//2
    y_lst = {-1:[], 1:[]}
    param_lst = {-1:[], 1:[]}
    x1 = x0
    direction = -1
    density = 10
    icol = 0

    peak_lst = []

    csec_i1 = -nx//2
    csec_i2 = nx + nx//2

    csec_lst    = np.zeros(csec_i2 - csec_i1)
    csec_nlst   = np.zeros(csec_i2 - csec_i1, dtype=np.int32)
    csec_maxlst = np.zeros(csec_i2 - csec_i1)

    param_lst = {-1:[], 1:[]}
    nodes_lst = {}


    # generate a window list
    dense_y = np.linspace(0, nx-1, (nx-1)*10+1)
    separation_lst = fsep(dense_y)
    separation_lst = np.int32(np.round(separation_lst))
    window = 2*separation_lst*density+1

    while(True):
        nodes_lst[x1] = []
        flux1 = logdata[x1, :]
        linflux1 = np.median(data[x1-2:x1+3,:], axis=0)

        convflux1 = flux1.copy()
        if icol==0:
            convflux1_center = convflux1

        # find peaks
        n = convflux1.size
        f = intp.InterpolatedUnivariateSpline(np.arange(n), convflux1, k=3)
        convflux2 = f(dense_y)
        imax, fmax = get_local_minima(-convflux2, window=window)
        ymax = dense_y[imax]
        fmax = -fmax

        if icol==0:
            for y,f in zip(ymax, fmax):
                peak_lst.append((y, f))
                nodes_lst[x1].append(y)
            # convert to
            i1 = 0 - csec_i1
            i2 = nx - csec_i1
            csec_lst[i1:i2] += linflux1
            csec_nlst[i1:i2] += 1
            csec_maxlst[i1:i2] = np.maximum(csec_maxlst[i1:i2],linflux1)

            if fig_align is not None:
                q01 = np.percentile(flux1, 1)
                q99 = np.percentile(flux1, 99)
                # take a small portion of the cross-section and
                # calculate the constrast of peak
                sflux = flux1[int(0.45*nx):int(0.55*nx)]
                contrast = sflux.max() - sflux.min()
                amp = scan_step*(q99-q01)/contrast*2
                # define the transform function
                plottrans = lambda flux: (flux - q01)/(q99 - q01)*amp

                # use the transform function
                fig_align.ax1.plot(allx, plottrans(flux1)+x1,
                        c='C0', lw=0.5)
                fig_align.ax2.plot(allx, plottrans(flux1)+x1,
                        c='C0', lw=0.5)
        else:
            # aperture
            param, _ = find_shift(convflux0, convflux1, deg=align_deg)
            param_lst[direction].append(param[0:-1])

            for y, f in zip(ymax, fmax):
                ystep = y
                for param in param_lst[direction][::-1]:
                    ystep = backward(ystep, param)
                peak_lst.append((ystep,f))
                nodes_lst[x1].append(y)

            # find ysta & yend, the start and point pixel after aperture
            # alignment
            ysta, yend = 0., nx-1.
            for param in param_lst[direction][::-1]:
                ysta = backward(ysta, param)
                yend = backward(yend, param)
            # interplote the new csection, from ysta to yend
            ynew = np.linspace(ysta, yend, nx)
            interfunc = intp.InterpolatedUnivariateSpline(ynew, linflux1, k=3)
            # find the starting and ending indices for the new csection
            ysta_int = int(round(ysta))
            yend_int = int(round(yend))
            fnew = interfunc(np.arange(ysta_int, yend_int+1))
            # stack the new cross-section into the stacked cross-section
            # first, conver to the stacked cross-section coordinate
            i1 = ysta_int - csec_i1
            i2 = yend_int + 1 - csec_i1
            # then, stack the cross-sections
            csec_lst[i1:i2] += fnew
            csec_nlst[i1:i2] += 1
            csec_maxlst[i1:i2] = np.maximum(csec_maxlst[i1:i2],fnew)
            # for debug purpose
            #fig_trace.ax2.plot(np.arange(ysta_int, yend_int+1), fnew,
            #    'y-', alpha=0.2)
            fig_trace.ax2.plot(np.arange(ysta_int, yend_int+1), fnew,
                        '-', lw=0.2, alpha=0.05)

            # plot in the order alignment figure
            if fig_align is not None:
                # calculate the ally after alignment
                aligned_allx = allx.copy()
                for param in param_lst[direction][::-1]:
                    aligned_allx = backward(aligned_allx, param)
                # plot in the align figure
                fig_align.ax1.plot(allx, plottrans(flux1)+x1,
                        c='k', lw=0.5, alpha=0.2)
                fig_align.ax2.plot(aligned_allx, plottrans(flux1)+x1,
                        c='k', lw=0.5, alpha=0.2)

        nodes_lst[x1] = np.array(nodes_lst[x1])

        x1 += direction*scan_step
        if x1 <= 1200:
        #if x1 <= scan_step:
            # turn to the other direction
            direction = +1
            x1 = x0 + direction*scan_step
            y_lst[direction].append(x1)
            convflux0 = convflux1_center
            icol += 1
            continue
        elif x1 >= ny - 1100:
        #elif x1 >= w - scan_step:
            # scan ends
            break
        else:
            y_lst[direction].append(x1)
            convflux0 = convflux1
            icol += 1
            continue

    # filter the consecutive zero elements at the beginning and the end
    i_nonzero = np.nonzero(csec_nlst)[0]
    istart, iend = i_nonzero[0], i_nonzero[-1]
    csec_ylst = np.arange(csec_lst.size) + csec_i1
    # now csec_ylst starts from -h//2

    # set the zero elements to 1, preparing for the division
    csec_nlst = np.maximum(csec_nlst, 1)
    csec_lst /= csec_nlst

    fig_trace.ax2.plot(csec_ylst[istart:iend], csec_lst[istart:iend],
                '-', color='C0', lw=0.8)
    fig_trace.ax2.set_yscale('log')
    fig_trace.ax2.set_xlabel('X')
    fig_trace.ax2.set_ylabel('Count')
    fig_trace.ax2.set_ylim(0.5, )


    sectionx = csec_ylst[istart:iend]
    section = csec_lst[istart:iend]
    nsec = section.size

    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.plot(sectionx, section, c='C0')
    # find aperture positions
    min_width = 20
    xnodes = [200, 9000]
    wnodes = [300, 130]   # lengths of order distance in xnodes
    if channel == 'b':
        snodes = [200, 80]    # lengths of order gaps in xnodes
    elif channel == 'r':
        snodes = [250, 80]    # lengths of order gaps in xnodes
    c1 = np.polyfit(xnodes, wnodes, deg=1)
    c2 = np.polyfit(xnodes, snodes, deg=1)

    get_winlen = lambda x: 0 if x<0 else np.polyval(c1, x)
    get_gaplen = lambda x: 0 if x<0 else np.polyval(c2, x)

    winmask = np.ones_like(section, dtype=bool)

    for i1 in np.arange(0, nsec, 30):
        s1 = sectionx[i1]
        winlen = get_winlen(s1)
        gaplen = get_gaplen(s1)
        i2 = i1 + int(winlen)
        s2 = sectionx[min(i2, nsec-1)]
        if winlen <= 0 or gaplen <= 0:
            winmask[i1] = False
            continue
        percent = gaplen/winlen*100
        v = np.percentile(section[i1:i2], percent)
        idx = np.nonzero(section[i1:i2]>v)[0]
        winmask[idx+i1] = False

    ax3.plot(sectionx[winmask], section[winmask], 'o', c='C1', ms=3,
            alpha=0.4, zorder=-1)
    ax3.set_yscale('log')

    bkgmask = winmask.copy()
    aper_idx = np.nonzero(~bkgmask)[0]
    
    # determine the order edges

    order_index_lst = []
    for group in np.split(aper_idx, np.where(np.diff(aper_idx)>=3)[0]+1):
        i1 = group[0]
        i2 = group[-1]+1
        if i2 - i1 < min_width:
            continue

        order_index_lst.append((i1, i2))

    for (i1, i2) in order_index_lst:
        y1, y2 = ax3.get_ylim()
        ax3.fill_betweenx([y1, y2], sectionx[i1], sectionx[i2],
                color='C0', alpha=0.1, lw=0)
        ax3.set_ylim(y1, y2)

    mid_lst = []
    for (i1, i2) in order_index_lst:
        xdata = sectionx[i1:i2+1]
        ydata = section[i1:i2+1]
        A1, c1, s1, A2, c2, s2, b = find_double_peak(xdata, ydata)
        mid_lst.append((c1, c2, s1, s2, i1, i2))

    aperture_set1 = ApertureSet(shape=(ny, nx))
    aperture_set2 = ApertureSet(shape=(ny, nx))
    fig4 = plt.figure()
    ax4 = fig4.gca()
    ax4.imshow(logdata, cmap='gray')

    for aperture, (c1, c2, s1, s2, i1, i2) in enumerate(mid_lst):
        # central column
        c0 = (c1 + c2)/2
        xfit1, yfit1 = [], []
        xfit2, yfit2 = [], []
        for direction in [-1, 1]:
            ystep0 = c0
            ystep1 = c1
            ystep2 = c2
            for iy, param in enumerate(param_lst[direction]):
                ystep0 = forward(ystep0, param)
                ystep1 = forward(ystep1, param)
                ystep2 = forward(ystep2, param)
                if ystep1 <0 or ystep2 > nx-1:
                    # order center out of CCD boundaries
                    continue
                y1 = y_lst[direction][iy]
                x1 = int(ystep1 - 3*s1)
                x2 = int(ystep2 + 3*s2)
                x1 = max(x1, 0)
                x2 = min(x2, nx-1)
                xdata = np.arange(x1, x2)
                ydata = data[y1-3:y1+2, x1:x2].mean(axis=0)
                result = find_double_peak(xdata, ydata)
                _A1, _c1, _s1 = result[0:3]
                _A2, _c2, _s2 = result[3:6]
                xfit1.append(y1)
                yfit1.append(_c1)
                xfit2.append(y1)
                yfit2.append(_c2)

        xfit1, yfit1 = np.array(xfit1), np.array(yfit1)
        argsort1 = xfit1.argsort()
        xfit1, yfit1 = xfit1[argsort1], yfit1[argsort1]

        xfit2, yfit2 = np.array(xfit2), np.array(yfit2)
        argsort2 = xfit2.argsort()
        xfit2, yfit2 = xfit2[argsort2], yfit2[argsort2]

        ax4.plot(yfit1, xfit1, 'o', color='C{}'.format(aperture%10), lw=0.5, ms=3)
        ax4.plot(yfit2, xfit2, 'o', color='C{}'.format(aperture%10), lw=0.5, ms=3)

        # fit cheybshev polynomial
        if xfit1[0] < 1200+scan_step+10:
            left_domain = 0
        else:
            left_domain = xfit1[0]

        if xfit1[-1] > ny - 1100-scan_step-10:
            right_domain = ny-1
        else:
            right_domain = xfit1[-1]
        dm1 = (left_domain, right_domain)

        poly1 = Chebyshev.fit(xfit1, yfit1, domain=dm1, deg=4)

        if xfit2[0] < 1200+scan_step+10:
            left_domain = 0
        else:
            left_domain = xfit2[0]

        if xfit2[-1] > ny - 1100-scan_step-10:
            right_domain = ny-1
        else:
            right_domain = xfit2[-1]
        dm2 = (left_domain, right_domain)

        poly2 = Chebyshev.fit(xfit2, yfit2, domain=dm2, deg=4)

        aperture_loc1 = ApertureLocation(direct='y', shape=(ny, nx))
        aperture_loc1.set_position(poly1)
        aperture_set1[aperture] = aperture_loc1

        aperture_loc2 = ApertureLocation(direct='y', shape=(ny, nx))
        aperture_loc2.set_position(poly2)
        aperture_set2[aperture] = aperture_loc2

        '''
        fig6 = plt.figure()
        ax61 = fig6.add_subplot(211)
        ax62 = fig6.add_subplot(212)
        ax61.plot(xfit1, yfit1, 'o', ms=3, alpha=0.6)
        ax61.plot(xfit2, yfit2, 'o', ms=3, alpha=0.6)
        newx1 = poly1(ally)
        newx2 = poly2(ally)
        ax61.plot(ally, newx1, '-', lw=0.5)
        ax61.plot(ally, newx2, '--', lw=0.5)
        ax62.plot(xfit1, yfit1-poly1(xfit1), 'o', ms=3, alpha=0.6)
        ax62.plot(xfit2, yfit2-poly2(xfit2), 'o', ms=3, alpha=0.6)
        plt.show()
        plt.close(fig6)
        '''
        
    for aper, aperloc1 in sorted(aperture_set1.items()):
        dm1 = aperloc1.position.domain
        ypos1 = np.linspace(dm1[0], dm1[-1], 200)
        xpos1 = aperloc1.position(ypos1)
        ax4.plot(xpos1, ypos1, color='C{}'.format(aper%10), ls='-', lw=0.5)

        aperloc2 = aperture_set2[aper]
        dm2 = aperloc2.position.domain
        ypos2 = np.linspace(dm2[0], dm2[-1], 200)
        xpos2 = aperloc2.position(ypos2)
        ax4.plot(xpos2, ypos2, color='C{}'.format(aper%10), ls='--', lw=0.5)
    plt.show()

    return aperture_set1, aperture_set2

