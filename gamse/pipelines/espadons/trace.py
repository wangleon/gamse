import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import matplotlib.pyplot as plt

from ...echelle.trace import ApertureSt, ApertureLocation
from ...utils.regression import get_clip_mean, iterative_polyfit
from .common import norm_profile

def find_order_locations(section, y, aligned_allx=None, plot=False):

    nx = section.size
    allx = np.arange(0, nx)

    min_width = 20 # minimum width of order

    xnodes = [100,1200]
    wnodes = [30, 65]
    snodes = [5, 33]
    
    c1 = np.polyfit(xnodes, wnodes, deg=1)
    c2 = np.polyfit(xnodes, snodes, deg=1)

    def get_winlen(x):
        if x < 0:
            return 0
        else:
            return np.polyval(c1, x)

    def get_gaplen(x):
        if x < 0:
            return 0
        else:
            return np.polyval(c2, x)

    '''
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(np.arange(nx), np.polyval(c1, np.arange(nx)))
    ax.plot(np.arange(nx), np.polyval(c2, np.arange(nx)))
    plt.show()
    '''

    winmask = np.ones_like(section, dtype=np.bool)
    #for i1 in np.arange(0, nx-np.polyval(c1, nx), 1):
    for i1 in np.arange(0, nx, 1):

        if aligned_allx is None:
            winlen = get_winlen(i1)
            gaplen = get_gaplen(i1)
            if winlen <= 0 or gaplen <= 0:
                winmask[i1] = False
                continue
            percent = gaplen/winlen*100
        else:
            ii1 = aligned_allx[i1]
            winlen = get_winlen(ii1)
            gaplen = get_gaplen(ii1)
            if winlen <= 0 or gaplen <= 0:
                winmask[i1] = False
                continue
            percent = gaplen/winlen*100

        #if y==504:
        #    print(i1, ii1, winlen, gaplen, percent)

        i1 = int(i1)
        i2 = i1 + int(winlen)
        if i2 >= nx-1:
            break
        v = np.percentile(section[i1:i2], percent)
        idx = np.nonzero(section[i1:i2]>v)[0]
        winmask[idx+i1] = False
    
    bkgmask = winmask.copy()
    maxiter = 10
    for ite in range(maxiter):
        c = np.polyfit(allx[bkgmask], np.log(section[bkgmask]), deg=13)
        newy = np.polyval(c, allx)
        resy = np.log(section) - newy
        std = resy[bkgmask].std()
        newbkgmask = resy < 3*std
        if newbkgmask.sum() == bkgmask.sum():
            break
        bkgmask = newbkgmask
    
    aper_mask = section > np.exp(newy+3*std)
    aper_idx = np.nonzero(aper_mask)[0]
    
    gap_mask = ~aper_mask
    gap_idx = np.nonzero(gap_mask)[0]

    # determine the order edges

    order_index_lst = []
    for group in np.split(aper_idx, np.where(np.diff(aper_idx)>=3)[0]+1):
        i1 = group[0]
        i2 = group[-1]+1
        if i2 - i1 < min_width:
            continue

        order_index_lst.append((i1, i2))

    order_pos_lst = np.array([(i1+i2)/2 for i1, i2 in order_index_lst])
    order_wid_lst = np.array([i2-i1 for i1, i2 in order_index_lst])
    order_sep_lst = derivative(order_pos_lst)

    # calculate mean order width
    width_mean, width_std, wid_mask = get_clip_mean(order_wid_lst,
                                high=3, low=10, maxiter=5)
    width_med = np.median(order_wid_lst[wid_mask])
    #width_mean = np.mean(order_wid_lst)
    #width_med  = np.median(order_wid_lst)
    #width_std  = np.std(order_wid_lst)
    sep_mask = np.ones_like(order_sep_lst, dtype=np.bool)
    for idx in np.nonzero(~wid_mask)[0]:
        sep_mask[idx] = False
        sep_mask[max(idx-1, 0)] = False
        sep_mask[min(idx+1, sep_mask.size-1)] = False

    result = iterative_polyfit(order_pos_lst, order_sep_lst, deg=3,
                mask=sep_mask)
    coeff_sep, order_sep_fit_lst, _, order_sep_mask, _= result

    '''
    # plot the order widths and separations
    figw = plt.figure(dpi=150)
    axw = figw.gca()
    axw.plot(order_pos_lst, order_wid_lst,
                'o', color='none', mec='C0')
    axw.plot(order_pos_lst[wid_mask], order_wid_lst[wid_mask],
                'o', color='C0', alpha=0.7, label='Width')
    axw.axhline(y=width_mean, ls='-', color='C1', lw=0.5,
                label='Median of Width')
    clip = 3
    axw.axhline(y=width_mean+clip*width_std, ls='--', color='C1', lw=0.5)
    axw.axhline(y=width_mean-clip*width_std, ls='--', color='C1', lw=0.5)
    axw.plot(order_pos_lst, order_sep_lst, 'o', color='none', mec='C3')
    axw.plot(order_pos_lst, order_sep_fit_lst, '-', color='C3')
    axw.plot(order_pos_lst[order_sep_mask], order_sep_lst[order_sep_mask],
                'o', color='C3', label='Separation', alpha=0.7)
    axw.legend(loc='upper left')
    axw.set_xlim(0, section.size-1)
    axw.set_ylim(0,)
    axw.grid(True, ls='--', lw=0.5)
    axw.set_axisbelow(True)
    figw.suptitle('Order Widths and Separations')
    figw.savefig('order_width_{:04d}.png'.format(y))
    plt.close(figw)
    '''

    m = order_wid_lst > width_mean + 3*width_std
    multi_order_lst = np.nonzero(m)[0]

    new_order_lst = []

    for multi_order in multi_order_lst:
        i1, i2 = order_index_lst[multi_order]
        local_sep = np.polyval(coeff_sep, (i1+i2)/2)
        multi = (i2-i1)/local_sep
        print(('Col {:4d} - Multi order: {:2d} {:4d}-{:4d}. width={:4d}  '
             'Multi={:4.2f}').format(y, multi_order, i1, i2, i2-i1, multi)
             )
        # plot the multi-peak orders
        #fig = plt.figure()
        #ax = fig.gca()
        #ax.plot(np.arange(i1, i2+1), np.log(section[i1:i2+1]), color='C3')
        #ax.plot(allx[winmask], np.log(section[winmask]), 'o', ms=5, color='none',
        #            mec='C0', mew=1)
        #_y1, _y2 = ax.get_ylim()

        multi = int(round(multi))
        prev_i1 = i1
        for j in range(1, multi):
            #e.g., mutli=4, j = 1,2,3
            pos = i1+(i2-i1)/multi*j
            pos1, pos2 = pos-local_sep/4, pos+local_sep/4
            #ax.axvline(x=pos, ls='-', color='C0')
            #ax.fill_betweenx([_y1, _y2], pos1, pos2,
            #        color='C0', alpha=0.2, lw=0)
            pos1 = int(round(pos1))
            pos2 = int(round(pos2))
            min_idx = section[pos1:pos2].argmin()+pos1
            new_order_lst.append((prev_i1, min_idx))
            prev_i1 = min_idx+1

            #ax.plot(min_idx, np.log(section[min_idx]), 'o', color='C3')
        new_order_lst.append((prev_i1, i2))

        #ax.set_xlim(i1, i2)
        #ax.set_ylim(_y1, _y2)
        #fig.savefig('multi_{:04d}_{:02d}.png'.format(y, multi_order))
        #plt.close(fig)

    # if has new splitted orders from multi-orders, append them into the order
    # list and replace the old order_index_lst
    if len(new_order_lst)>0:
        new_order_index_lst = []
        for i1, i2 in order_index_lst:
            has_new_order = False
            for ii1, ii2, in new_order_lst:
                if ii1>=i1 and ii2<=i2:
                    new_order_index_lst.append((ii1, ii2))
                    has_new_order = True

            if not has_new_order:
                new_order_index_lst.append((i1, i2))
        order_index_lst = new_order_index_lst

    # get order gap list
    order_gap_lst = []
    tmplst = [(0,0)] + order_index_lst + [(nx,nx)]
    for prev_ord, this_ord, next_ord in zip(tmplst, tmplst[1:], tmplst[2:]):
        gapleft  = this_ord[0] - prev_ord[1]
        gapright = next_ord[0] - this_ord[1]
        order_gap_lst.append((gapleft, gapright))

    # initialize the mean profiles
    all_xnodes = np.array([])
    all_ynodes = np.array([])

    if plot:
        fig2 = plt.figure(dpi=150, figsize=(12, 6))
        for i in range(4):
            ax = fig2.add_axes([0.05, 0.08+(3-i)*0.22, 0.55, 0.18])
        ax25 = fig2.add_axes([0.65, 0.08+2*0.22, 0.32, 0.22+0.18])
        ax26 = fig2.add_axes([0.65, 0.08,        0.32, 0.22+0.18])
    
        for i in range(4):
            ax = fig2.get_axes()[i]
            i1, i2 = nx//4*i, nx//4*(i+1)
            ax.plot(allx[winmask], section[winmask],
                        'o', ms=3, color='none', mec='C0', mew=0.6)
            ax.plot(allx[bkgmask], section[bkgmask],
                        'o', ms=3, color='C0')
            ax.plot(allx[i1:i2], section[i1:i2],
                        ls='-', color='k', lw=0.5)
            ax.plot(allx[i1:i2], np.exp(newy[i1:i2]),
                        ls='-', color='C1', lw=0.5)
            ax.plot(allx[i1:i2], np.exp(newy[i1:i2]+3*std),
                        ls='--', color='C1', lw=0.5)
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(tck.MultipleLocator(100))
            ax.grid(True, ls='--', lw=0.5)
            ax.set_axisbelow(True)


    # calculte mean order profile
    order_param_lst = []
    for (i1, i2), (g1, g2) in zip(order_index_lst, order_gap_lst):
        ii1 = i1 - min(g1//2, 2)
        ii2 = i2 + min(g2//2, 2)
        xnodes = np.arange(ii1, ii2)
        ynodes = section[ii1:ii2]
    
        newx, newy, param = norm_profile(xnodes, ynodes)
        order_param_lst.append(param[0:3])

        if plot:
            ax26.scatter(newx, newy, alpha=0.5, s=4)
    
        all_xnodes = np.append(all_xnodes, newx)
        all_ynodes = np.append(all_ynodes, newy)

    if plot:
        # plot order range
        for i in range(4):
            ax = fig2.get_axes()[i]
            for (i1, i2), param in zip(order_index_lst, order_param_lst):
                i3, i4, i5 = param
                #i3, i4, i5: gap, first central peak, second central peak
                if nx//4*i < i1 < nx//4*(i+1) or nx//4*i < i2 < nx//4*(i+1):
                    #ax.fill_betweenx([2, 2**16], i1, i2,
                    #            color='C0', alpha=0.2, lw=0)
                    ax.fill_betweenx([2, 2**16], i1, i3,
                                color='C0', alpha=0.2, lw=0)
                    ax.fill_betweenx([2, 2**16], i3, i2,
                                color='C3', alpha=0.2, lw=0)
                if nx//4*i < i4 < nx//4*(i+1):
                    vi4 = section[int(round(i4))]
                    ax.vlines(i4, vi4*1.2, vi4*3, color='C0', lw=0.5)
                if nx//4*i < i5 < nx//4*(i+1):
                    vi5 = section[int(round(i5))]
                    ax.vlines(i5, vi5*1.2, vi5*3, color='C3', lw=0.5)
            ax.set_ylim(2, 2**16)

    if plot:
        order_pos_lst = np.array([(i1+i2)/2 for i1, i2 in order_index_lst])
        order_wid_lst = np.array([i2-i1 for i1, i2 in order_index_lst])
        order_sep_lst = derivative(order_pos_lst)
        width_mean, width_std, wid_mask = get_clip_mean(order_wid_lst,
                                    high=3, low=10, maxiter=5)
        result = iterative_polyfit(order_pos_lst, order_sep_lst, deg=3)
        coeff_sep, order_sep_fit_lst, _, order_sep_mask, _= result
        ax25.plot(order_pos_lst, order_wid_lst,
                    'o', color='none', ms=5, mec='C0')
        ax25.plot(order_pos_lst[wid_mask], order_wid_lst[wid_mask],
                    'o', color='C0', ms=5, alpha=0.7, label='Order Width')
        ax25.plot(order_pos_lst, order_sep_lst,
                    'o', color='none', ms=5, mec='C3')
        ax25.plot(order_pos_lst, order_sep_fit_lst,
                    '-', color='C3')
        ax25.plot(order_pos_lst[order_sep_mask], order_sep_lst[order_sep_mask],
                    'o', color='C3', ms=5, alpha=0.7, label='Order Separation')
        ax25.axhline(y=width_mean, ls='-', color='C0', lw=0.7,
                    label='Mean Width = {:.2f}'.format(width_mean))
        ax25.legend(loc='upper left')
        ax25.set_xlim(0, section.size-1)
        ax25.set_ylim(0,)
        ax25.grid(True, ls='--', lw=0.5)
        ax25.set_axisbelow(True)
    
        p1, p2 = -15, 15
        xlst, ylst = get_mean_profile(all_xnodes, all_ynodes, p1, p2, 0.5)
        f = intp.InterpolatedUnivariateSpline(xlst, ylst, k=3, ext=3)
        newx = np.arange(p1, p2+1e-5, 0.1)
        newy = f(newx)
        ax26.plot(newx, newy, ls='-', color='k', lw=0.7)
        ax26.fill_betweenx([-0.3, 1.3], p1, 0, color='C0', alpha=0.2, lw=0)
        ax26.fill_betweenx([-0.3, 1.3], 0, p2, color='C3', alpha=0.2, lw=0)
        ax26.axhline(0, ls='--', lw=0.5, color='k')
        ax26.text(p1/2, -0.1, 'A')
        ax26.text(p2/2, -0.1, 'B')
        ax26.set_xlim(p1-1, p2+1)
        ax26.grid(True, ls='--', lw=0.5)
        ax26.set_axisbelow(True)
        ax26.set_ylim(-0.3, 1.3)
        
        for i in range(4):
            ax = fig2.get_axes()[i]
            ax.set_xlim(nx//4*i, nx//4*(i+1))
        fig2.savefig('section_{:04d}.png'.format(y))
        
        plt.close(fig2)

    result_lst = [(i1, i2, v1, v2, v3) for (i1, i2), (v1, v2, v3) in zip(
                                            order_index_lst, order_param_lst)]
    return result_lst


def find_apertures(data):
    ny, nx = data.shape
    allx = np.arange(nx)

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
            mask = np.ones_like(flux0, dtype=np.bool)
        return res_lst[mask]

    align_deg = 2
    scan_step = 100
    
    x0 = ny//2
    x_lst = {-1:[], 1:[]}
    param_lst = {-1:[], 1:[]}
    x1 = x0
    direction = -1
    icol = 0

    all_order_param_lst = {}
    all_aligned_x_lst = {}
    
    fig0 = plt.figure(figsize=(12,8), dpi=200)
    fig1 = plt.figure(figsize=(12,8), dpi=200)
    ax0 = fig0.add_axes([0.1, 0.1, 0.85, 0.85])
    ax1 = fig1.add_axes([0.1, 0.1, 0.85, 0.85])
    while(True):
        #flux1 = np.mean(logdata[x1-2:x1+3, :], axis=0)
        flux1 = np.mean(data[x1-2:x1+3, :], axis=0)
    
        negmask = flux1<0
        if negmask.sum()>0:
            print('Negative values:', allx[negmask])
            f = intp.InterpolatedUnivariateSpline(
                    allx[~negmask], flux1[~negmask], k=1)
            flux1 = f(allx)
        logflux1 = np.log(flux1)
    
        if icol == 0:
            logflux1_center = logflux1
            ax0.plot(np.arange(nx), (logflux1-1)*100+x1, color='C0', lw=0.6)
            ax1.plot(np.arange(nx), (logflux1-1)*100+x1, color='C0', lw=0.6)
    
            all_order_param_lst[x1] = find_order_locations(flux1, x1)
            all_aligned_x_lst[x1] = allx
    
        else:
    
            p0 = [0.0 for i in range(align_deg+1)]
            p0[-3] = 1.0
            #p0 = [0.0 for i in range(deg+2)]
            #p0[-4] = 1.0
            interfunc = intp.InterpolatedUnivariateSpline(
                        np.arange(logflux1.size), logflux1, k=3, ext=3)
            mask = np.ones_like(logflux0, dtype=np.bool)
            clipping = 5.
            maxiter = 10
            for i in range(maxiter):
                param, _ = opt.leastsq(resfunc, p0, args=(interfunc, logflux0, mask))
                res_lst = resfunc(param, interfunc, logflux0)
                std = res_lst.std()
                mask1 = res_lst <  clipping*std
                mask2 = res_lst > -clipping*std
                new_mask = mask1*mask2
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask
                p0 = param
    
            '''
            figalg = plt.figure(dpi=200)
            axa1 = figalg.add_subplot(211)
            axa2 = figalg.add_subplot(212)
            axa1.plot(logflux0, lw=0.5)
            axa1.plot(logflux1, lw=0.5)
            axa1.plot(fitfunc(param, interfunc, logflux0.size), lw=0.5)
            axa2.plot(resfunc(param, interfunc, logflux0), lw=0.5)
            axa1.set_ylim(1, 10)
            figalg.savefig('fit_{:04d}.png'.format(x1))
            plt.close(figalg)
            '''
    
            param_lst[direction].append(param[0:-1])
            #param_lst[direction].append(param[0:-2])
    
            aligned_allx = allx.copy()
            for param in param_lst[direction][::-1]:
                aligned_allx = backward(aligned_allx, param)
    
            ax0.plot(allx, (logflux1-1)*100+x1,
                        color='k', alpha=0.2, lw=0.6)
            ax1.plot(aligned_allx, (logflux1-1)*100+x1,
                        color='k', alpha=0.2, lw=0.6)
    
            all_order_param_lst[x1] = find_order_locations(flux1, x1, aligned_allx)
            all_aligned_x_lst[x1] = aligned_allx
    
        x1 += direction*scan_step
        if x1 <= 10:
            # turn to the other direction
            direction = +1
            x1 = x0 + direction*scan_step
            x_lst[direction].append(x1)
            logflux0 = logflux1_center
            icol += 1
            continue
        elif x1 >= ny - 20:
            # scan ends
            break
        else:
            x_lst[direction].append(x1)
            logflux0 = logflux1
            icol += 1
            continue

    aligned_bound_lst = []
    all_aligned_order_param_lst = {}
    for x1, order_param_lst in sorted(all_order_param_lst.items()):
        aligned_x = all_aligned_x_lst[x1]
    
        aligned_bound_lst.append((math.floor(aligned_x[0]),
                                  math.ceil(aligned_x[-1])))
    
        f = intp.InterpolatedUnivariateSpline(allx, aligned_x, k=3)
        
        # find aligned order param
        aligned_order_param_lst = [(f(i1), f(i2), f(v1), f(v2), f(v3))
                                    for i1, i2, v1, v2, v3 in order_param_lst]
        all_aligned_order_param_lst[x1] = aligned_order_param_lst

    aligned_peakAB_lst = []
    aligned_peakA_lst = []
    aligned_peakB_lst = []
    for x1, aligned_order_param_lst in sorted(all_aligned_order_param_lst.items()):
        for _, _, newv1, newv2, newv3 in aligned_order_param_lst:
            aligned_peakAB_lst.append(newv1)
            aligned_peakA_lst.append(newv2)
            aligned_peakB_lst.append(newv3)

    minx = min(aligned_bound_lst, key=lambda item:item[0])[0]
    maxx = max(aligned_bound_lst, key=lambda item:item[1])[1]
    bins = np.arange(minx, maxx+1, 1)
    histAB, _ = np.histogram(aligned_peakAB_lst, bins=bins)
    histA,  _ = np.histogram(aligned_peakA_lst, bins=bins)
    histB,  _ = np.histogram(aligned_peakB_lst, bins=bins)
    binx = bins[0:-1] + np.diff(bins)/2

    # find allsize_lst, which is the number of columns scanned in each
    # cross-disp pixels
    allsize_lst = np.zeros(maxx-minx)
    for (x1, x2) in aligned_bound_lst:
        xlst = np.ones(x2-x1)
        # add zeros in the beginning
        xlst = np.insert(xlst,0,[0]*(x1-minx))
        # add zeros in the end
        xlst = np.append(xlst, [0]*(maxx-x2))
        allsize_lst += xlst
    # normalize the histogram
    norm_histAB = histAB/allsize_lst
    norm_histA  = histA/allsize_lst
    norm_histB  = histB/allsize_lst

    fig5 = plt.figure(dpi=200)
    ax51 = fig5.add_subplot(211)
    ax52 = fig5.add_subplot(212)
    ax51.fill_between(binx, histAB, color='C1', step='mid', alpha=0.6)
    ax51.fill_between(binx, histA,  color='C0', step='mid', alpha=0.6)
    ax51.fill_between(binx, histB,  color='C3', step='mid', alpha=0.6)
    ax51.step(binx, allsize_lst)
    
    ax52.fill_between(binx, norm_histAB, color='C1', step='mid', alpha=0.6)
    ax52.fill_between(binx, norm_histA,  color='C0', step='mid', alpha=0.6)
    ax52.fill_between(binx, norm_histB,  color='C3', step='mid', alpha=0.6)

    y1, y2 = ax52.get_ylim()

    # get group list
    idx = np.where(norm_histAB>1e-5)[0]
    groupAB_lst = np.split(idx, np.where(np.diff(idx)>2)[0]+1)
    centAB_lst = [(binx[group]*norm_histAB[group]).sum()/(norm_histAB[group].sum())
                    for group in groupAB_lst]
    cumnAB_lst = [norm_histAB[group].sum() for group in groupAB_lst]
    
    idx = np.where(norm_histA>1e-5)[0]
    groupA_lst = np.split(idx, np.where(np.diff(idx)>2)[0]+1)
    centA_lst = [(binx[group]*norm_histA[group]).sum()/(norm_histA[group].sum())
                    for group in groupA_lst]
    cumnA_lst = [norm_histA[group].sum() for group in groupA_lst]
    
    idx = np.where(norm_histB>1e-5)[0]
    groupB_lst = np.split(idx, np.where(np.diff(idx)>2)[0]+1)
    centB_lst = [(binx[group]*norm_histB[group]).sum()/(norm_histB[group].sum())
                    for group in groupB_lst]
    cumnB_lst = [norm_histB[group].sum() for group in groupB_lst]

    x1_lst = [x0]
    for direction in [1, -1]:
        for x1 in x_lst[direction]:
            x1_lst.append(x1)

    order_AB_lst = {}
    order_A_lst = {}
    order_B_lst = {}
    iorder = 0
    for group, cent, cumn, groupA, centA, cumnA, groupB, centB, cumnB in zip(
            groupAB_lst, centAB_lst, cumnAB_lst,
            groupA_lst,  centA_lst,  cumnA_lst,
            groupB_lst,  centB_lst,  cumnB_lst,
            ):
        if cumn < 0.3:
            continue

        xlst, yABlst, yAlst, yBlst = [], [], [], []

        for x1 in x1_lst:
            order_param_lst         = all_order_param_lst[x1]
            aligned_order_param_lst = all_aligned_order_param_lst[x1]
    
            for (_, _, v1, v2, v3), (_, _, newv1, newv2, newv3) in zip(
                    order_param_lst, aligned_order_param_lst):
                if binx[group[0]]-1 < newv1 < binx[group[-1]]+1:
                    xlst.append(x1)
                    yABlst.append(v1)
                    yAlst.append(v2)
                    yBlst.append(v3)
                    break
        xlst   = np.array(xlst)
        yABlst = np.array(yABlst)
        yAlst  = np.array(yAlst)
        yBlst  = np.array(yBlst)

        # resort
        idx = xlst.argsort()
        xlst   = xlst[idx]
        yABlst = yABlst[idx]
        yAlst  = yAlst[idx]
        yBlst  = yBlst[idx]
        order_AB_lst[iorder] = (xlst, yABlst)
        order_A_lst[iorder] = (xlst, yAlst)
        order_B_lst[iorder] = (xlst, yBlst)
        iorder += 1

    for group in groupAB_lst:
        i1, i2 = group[0], group[-1]
        ax52.fill_betweenx([y1,y2], binx[i1], binx[i2], color='C3', alpha=0.1)
    ax52.set_ylim(y1, y2)
    ax51.set_xlim(minx, maxx)
    ax52.set_xlim(minx, maxx)

    aperture_set = ApertureSet(shape=(ny, nx))
    aperture_set_A = ApertureSet(shape=(ny, nx))
    aperture_set_B = ApertureSet(shape=(ny, nx))

    figall = plt.figure(dpi=200, figsize=(15, 7))
    axall = figall.gca()
    for iorder in sorted(order_AB_lst.keys()):
        xlst_AB, ylst_AB = order_AB_lst[iorder]
        xlst_A, ylst_A = order_A_lst[iorder]
        xlst_B, ylst_B = order_B_lst[iorder]

    #for iorder, (xlst, ylst) in sorted(order_AB_lst.items()):

        fitmask = np.ones_like(xlst_AB, dtype=np.bool)
        maxiter = 10
        for nite in range(maxiter):
            poly = Chebyshev.fit(xlst_AB[fitmask], ylst_AB[fitmask], deg=4)
            yres = ylst_AB - poly(xlst_AB)
            std = yres[fitmask].std()
            new_fitmask = (yres > -3*std) * (yres < 3*std)
            if new_fitmask.sum() == fitmask.sum():
                break
            fitmask = new_fitmask
    
        aperture_loc = ApertureLocation(direct='y', shape=(ny, nx))
        aperture_loc.set_position(poly)
        aperture_set[iorder] = aperture_loc
    
        # for plotting
        newx = np.arange(0, ny)
        newy = poly(newx)
        m = (newy >= 0) * (newy < nx)
    
        axall.scatter(xlst_AB, ylst_AB, s=15, color='none', edgecolor='C0')
        axall.scatter(xlst_AB[fitmask], ylst_AB[fitmask], s=15, color='C0')
        axall.plot(newx[m], newy[m], '-', color='C0', lw=0.7)

        figm = plt.figure(dpi=150)
        axm1 = figm.add_subplot(211)
        axm2 = figm.add_subplot(212)
        axm1.scatter(xlst_AB, ylst_AB, s=10, color='none', edgecolor='C0')
        axm1.scatter(xlst_AB[fitmask], ylst_AB[fitmask], s=10, color='C0')
        axm1.plot(newx[m], newy[m], '-', lw=0.7)
        axm2.scatter(xlst_AB, ylst_AB-poly(xlst_AB),
                    s=10, color='none', edgecolor='C0')
        axm2.scatter(xlst_AB[fitmask], ylst_AB[fitmask]-poly(xlst_AB[fitmask]),
                    s=10, color='C0')
        figm.savefig('order_fit_{:03d}.png'.format(iorder))
        plt.close(figm)

        ###################################
        fitmask = np.ones_like(xlst_A, dtype=np.bool)
        maxiter = 10
        for nite in range(maxiter):
            poly = Chebyshev.fit(xlst_A[fitmask], ylst_A[fitmask], deg=4)
            yres = ylst_A - poly(xlst_A)
            std = yres[fitmask].std()
            new_fitmask = (yres > -3*std) * (yres < 3*std)
            if new_fitmask.sum() == fitmask.sum():
                break
            fitmask = new_fitmask
    
        aperture_loc = ApertureLocation(direct='y', shape=(ny, nx))
        aperture_loc.set_position(poly)
        aperture_set_A[iorder] = aperture_loc
    
        # for plotting
        newx = np.arange(0, ny)
        newy = poly(newx)
        m = (newy >= 0) * (newy < nx)
    
        axall.scatter(xlst_A, ylst_A, s=15, color='none', edgecolor='C0')
        axall.scatter(xlst_A[fitmask], ylst_A[fitmask], s=15, color='C0')
        axall.plot(newx[m], newy[m], '-', color='C0', lw=0.7)


        ###################################
        fitmask = np.ones_like(xlst_B, dtype=np.bool)
        maxiter = 10
        for nite in range(maxiter):
            poly = Chebyshev.fit(xlst_B[fitmask], ylst_B[fitmask], deg=4)
            yres = ylst_B - poly(xlst_B)
            std = yres[fitmask].std()
            new_fitmask = (yres > -3*std) * (yres < 3*std)
            if new_fitmask.sum() == fitmask.sum():
                break
            fitmask = new_fitmask
    
        aperture_loc = ApertureLocation(direct='y', shape=(ny, nx))
        aperture_loc.set_position(poly)
        aperture_set_B[iorder] = aperture_loc
    
        # for plotting
        newx = np.arange(0, ny)
        newy = poly(newx)
        m = (newy >= 0) * (newy < nx)
    
        axall.scatter(xlst_B, ylst_B, s=15, color='none', edgecolor='C0')
        axall.scatter(xlst_B[fitmask], ylst_B[fitmask], s=15, color='C0')
        axall.plot(newx[m], newy[m], '-', color='C0', lw=0.7)

    axall.grid(True, ls='--', lw=0.5)
    axall.set_axisbelow(True)
    axall.set_xlim(0, ny-1)
    axall.set_ylim(0, nx-1)
    axall.set_aspect(1)
    figall.savefig('order_all.png')
    plt.close(figall)

    return aperture_set, aperture_set_A, aperture_set_B

