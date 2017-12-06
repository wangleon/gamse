import logging

logger = logging.getLogger(__name__)

import os
import time
import math
import numpy as np
from numpy.polynomial import Chebyshev
import astropy.io.fits   as fits
import scipy.interpolate as intp
import scipy.signal      as sg
import scipy.optimize    as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ApertureLocation(object):
    '''
    Location of an echelle order.

    Attributes:
        direct (int): 0 if along Y axis; 1 if along X axis
        


    '''
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__update_attr()

    def set_nodes(self, key, xdata, ydata):
        '''
        set nodes for center, lower, or upper lines
        xdata:
            [x1, x2, x3, ...]
        ydata:
            [y1, y2, y3, ...]
        '''
        # filter the None values in (xdata, ydata)
        xydata = [(x,y) for x, y in zip(xdata, ydata)
                  if x is not None and y is not None]
        xnodes, ynodes = zip(*xydata)

        # sort the nodes according to x coordinates
        xnodes, ynodes = np.array(xnodes), np.array(ynodes)
        if self.direct == 0:
            # order is along y axis
            xnodes = xnodes[ynodes.argsort()]
            ynodes = np.sort(ynodes)
        elif self.direct == 1:
            # order is along x axis
            ynodes = ynodes[xnodes.argsort()]
            xnodes = np.sort(xnodes)

        xydata = [(x, y) for x, y in zip(xnodes, ynodes)]

        setattr(self, 'nodes_%s'%key, xydata)


    def fit_nodes(self, key, degree, clipping, maxiter):
        '''
        Fit the polynomial iteratively with sigma-clipping method and get the
        coefficients.
        '''

        h, w = self.shape
        nodes = getattr(self, 'nodes_%s'%key)
        xnodes, ynodes = zip(*nodes)
        
        # normalize to [0, 1)
        xfit = np.array(xnodes, dtype=np.float32)/w
        yfit = np.array(ynodes, dtype=np.float32)/h

        if self.direct == 0:
            # order is along y axis
            xfit, yfit = yfit, xfit
        elif self.direct == 1:
            # order is along x axis
            xfit, yfit = xfit, yfit

        # initialize mask
        mask = np.ones_like(xfit, dtype=np.bool)
        for niter in range(maxiter):
            # determine the appropriate degree of polynomial
            npoints = mask.sum()
            if npoints <= 1:
                # if there's only one point in this order, then skip
                continue
            else:
                deg = min(npoints-1, degree)

            coeff = np.polyfit(xfit[mask], yfit[mask], deg=deg)
            res = yfit - np.polyval(coeff, xfit)
            mean = res[mask].mean()
            std  = res[mask].std(ddof=1)
            m1 = res < mean + clipping*std
            m2 = res > mean - clipping*std
            new_mask = m1*m2
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask

        # put the coefficients into the order location instance
        setattr(self, 'coeff_%s'%key, coeff)

        info = {
                'xnodes':    xfit*w,
                'ynodes':    yfit*h,
                'residuals': res,
                'sigma':     std,
                'mask':      mask,
                }

        return info

    def set_position(self, poly):
        setattr(self, 'position', poly)

    def __str__(self):

        string  = '%8s = %d'%('direct', self.direct)+os.linesep
        string += '%8s = %s'%('shape',  str(self.shape))+os.linesep

        strlst = ['%+15.10e'%c for c in self.position.coef]
        string += 'position = [%s]'%(', '.join(strlst))+os.linesep

        # find nodes
        for key in ['lower', 'center', 'upper']:
            key1 = 'nodes_%s'%key
            if hasattr(self, key1):
                nodes = getattr(self, key1)
                strlst = ['(%g, %g)'%(x,y) for x, y in nodes]
                string += '%6s = [%s]%s'%(key1, ', '.join(strlst), os.linesep)

        # find coefficients
        for key in ['lower', 'center', 'upper']:
            key1 = 'coeff_%s'%key
            if hasattr(self, key1):
                coeff = getattr(self, key1)
                strlst = ['%+15.10e'%c for c in coeff]
                string += '%6s = [%s]%s'%(key1, ', '.join(strlst), os.linesep)

        for key in ['nsat','mean','median','max']:
            if hasattr(self, key):
                value = getattr(self, key)
                string += '%8s = %g'%(key, value)+os.linesep

        return string

    def __update_attr(self):
        '''Update attributes'''

        # udpate direction attribute
        if hasattr(self, 'direct'):
            if self.direct in ['x','X']:
                self.direct = 1
            elif self.direct in ['y','Y']:
                self.direct = 0

class ApertureSet(object):
    '''
    ApertureSet is a group of ApertureLocation instances.

    Attributes:
        dict (dict): Dict containing aperture numbers and :class:`ApertureLocation` instances
        current (dict): 
    '''
    def __init__(self, *args, **kwargs):
        self.dict = {}
        self.curret = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        return key in self.dict

    def items(self):
        return self.dict.items()

    def __str__(self):
        string = ''
        for aperture, aperture_loc in sorted(self.dict.items()):
            string += 'APERTURE LOCATION %d%s'%(aperture, os.linesep)
            string += str(aperture_loc)
        return string

    def save_txt(self, filename):
        '''
        Save the order set into an ascii file.
        
        Args:
            filename (str): Path to the output ascii file
        '''
        outfile = open(filename, 'w')
        outfile.write(str(self))
        outfile.close()

def find_apertures(filename, **kwargs):
    '''
    Find the positions of apertures on a CCD image.

    Args:
        filename (str): Path to the input image FITS file.
        mask_file (str): Path to the mask FITS file.
        scan_step (int): Steps of pixels used to scan along the main dispersion
            direction.
        minimum (float): Minimum value to filter the input image.
        seperation (int): Estimated order seperations (in pixel) along the
            cross-dispersion.
        direction (int):
        filling (float): Fraction of detected pixels to total step of scanning.
        degree (int): Degree of polynomials to fit aperture locations.
        display (bool): If True, display a figure on the screen.
        fig_file (str): Path to the output figure.
        result_file (str): Path to the output ascii file.

    Returns:
        list: A list containing the coefficients of the locations.

    '''
    from ..utils.onedarray import get_local_minima

    if os.path.exists(filename):
        data, head = fits.getdata(filename, header=True)
    else:
        logger.error('File: %s does not exist'%filename)
        raise ValueError

    mask_file    = kwargs.pop('mask_file', None)
    minimum      = kwargs.pop('minimum', 1e-3)
    scan_step    = kwargs.pop('scan_step', 50)
    seperation   = kwargs.pop('seperation', 20)
    filling      = kwargs.pop('filling', 0.3)
    display      = kwargs.pop('display', True)
    degree       = kwargs.pop('degree', 3)
    fig_file     = kwargs.pop('fig_file', None)
    result_file  = kwargs.pop('result_file', None)

    # initialize the mask
    if mask_file is not None and os.path.exists(mask_file):
        mask_data = fits.getdata(mask_file)
        logger.info('Read mask from existing file: %s'%mask_file)
    else:
        mask_data = np.zeros_like(data, dtype=np.int16)
        logger.info('Initialize mask from data array')

    # find saturation pixels
    sat_mask = (mask_data&4 == 4)

    h, w = data.shape

    # filter the negative pixels and replace them with the minimum value
    mask = data > 0
    substitute = data[mask].min()
    logdata = np.log10(np.maximum(data,minimum))

    # initialize the color list
    colors = 'rgbcmyk'

    # create a background image
    fig = plt.figure(figsize=(20,10),dpi=150)
    ax1 = fig.add_axes([0.05,0.06,0.43,0.86])
    ax2 = fig.add_axes([0.54,0.06,0.43,0.86])
    ax1.imshow(logdata,cmap='gray',interpolation='none')
    # create a colormap for saturation mask
    sat_cmap = mcolors.LinearSegmentedColormap.from_list('TransRed',
               [(1,0,0,0), (1,0,0,0.8)], N=2)
    ax1.imshow(sat_mask, interpolation='none',cmap=sat_cmap)
    ax1.set_xlim(0,w-1)
    ax1.set_ylim(h-1,0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # define a scroll function
    def on_scroll(event):
        if event.inaxes == ax1:
            x1, x2 = ax1.get_xlim()
            y1, y2 = ax1.get_ylim()
            x1 = event.xdata - (1-event.step*0.1)*(event.xdata - x1)
            x2 = event.xdata + (1-event.step*0.1)*(x2 - event.xdata)
            y1 = event.ydata - (1-event.step*0.1)*(event.ydata - y1)
            y2 = event.ydata + (1-event.step*0.1)*(y2 - event.ydata)
            ax1.set_xlim(x1, x2)
            ax1.set_ylim(y1, y2)
            fig.canvas.draw()
    fig.suptitle('Trace for %s'%os.path.basename(filename))
    fig.canvas.mpl_connect('scroll_event',on_scroll)
    fig.canvas.draw()
    if display:
        plt.show(block=False)

    x0 = int(w/2.)
    x_lst = {-1:[], 1:[]}
    x1 = x0
    direction = -1
    density = 10
    icol = 0
    peak_lst = []
    csec_i1 = -int(h/2.)
    csec_i2 = h + int(h/2.)
    csec_lst    = np.zeros(csec_i2 - csec_i1)
    csec_nlst   = np.zeros(csec_i2 - csec_i1)
    csec_maxlst = np.zeros(csec_i2 - csec_i1)
    slope_lst = {-1:[], 1:[]}
    shift_lst = {-1:[], 1:[]}
    nodes_lst = {}

    def fitfunc(p, interfunc, n):
        slope, shift, zoom = p
        return interfunc(np.arange(n)*slope+shift) + zoom
    def resfunc(p, interfunc, flux0, mask=None):
        res_lst = flux0 - fitfunc(p, interfunc, flux0.size)
        if mask is None:
            mask = np.ones_like(flux0, dtype=np.bool)
        return res_lst[mask]
    def find_shift(flux0, flux1):
        p0 = [1.0, 0.0, 0.0]
        interfunc = intp.InterpolatedUnivariateSpline(
                    np.arange(flux1.size), flux1, k=3)
        mask = np.ones_like(flux0, dtype=np.bool)
        clipping = 5.
        for i in range(10):
            p, _ = opt.leastsq(resfunc, p0, args=(interfunc, flux0, mask))
            res_lst = resfunc(p, interfunc, flux0)
            mean = res_lst.mean()
            std  = res_lst.std()
            new_mask = (res_lst < mean + clipping*std)*(res_lst > mean - clipping*std)
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask
        return p, mask

    while(True):
        nodes_lst[x1] = []
        flux1 = logdata[:,x1]
        linflux1 = data[:,x1]
        flux1 = sg.savgol_filter(flux1, window_length=5, polyorder=2)
        if icol == 0:
            # will be used when turning the direction
            flux1_center = flux1

        # find peaks with Y precision of 1./density pixels
        f = intp.InterpolatedUnivariateSpline(np.arange(flux1.size), flux1, k=3)
        ynew = np.linspace(0, flux1.size-1, (flux1.size-1)*density+1)
        flux2 = f(ynew)
        imax, fmax = get_local_minima(-flux2, window=2*seperation*density+1)
        ymax = ynew[imax]
        fmax = -fmax

        if icol == 0:
            for y,f in zip(ymax,fmax):
                peak_lst.append((y,f))
                nodes_lst[x1].append(y)
            i1 = 0 - csec_i1
            i2 = 0 - csec_i1 + h
            csec_lst[i1:i2] += linflux1
            csec_nlst[i1:i2] += 1
            csec_maxlst[i1:i2] = np.maximum(csec_maxlst[i1:i2],linflux1)
        else:
            # aperture alignment of each selected column, described by
            # (slope, shift)
            (slope, shift, zoom), mask = find_shift(flux0, flux1)
            slope_lst[direction].append(slope)
            shift_lst[direction].append(shift)
            for y, f in zip(ymax, fmax):
                ystep = y
                for slope, shift in zip(slope_lst[direction][::-1],
                                        shift_lst[direction][::-1]):
                    ystep = (ystep - shift)/slope
                peak_lst.append((ystep,f))
                nodes_lst[x1].append(y)

            # find ysta & yend, the start and point pixel after aperture
            # aligment
            ysta, yend = 0., h-1.
            for slope, shift in zip(slope_lst[direction][::-1],
                                    shift_lst[direction][::-1]):
                ysta = (ysta - shift)/slope
                yend = (yend - shift)/slope
            # interplote the new csection, from ysta to yend
            ynew = np.linspace(ysta, yend, h)
            interfunc = intp.InterpolatedUnivariateSpline(ynew, linflux1, k=3)
            # find the starting and ending indices for the new csection
            ysta_int = int(round(ysta))
            yend_int = int(round(yend))
            fnew = interfunc(np.arange(ysta_int, yend_int+1))
            i1 = ysta_int - csec_i1
            i2 = yend_int + 1 - csec_i1
            # add the new csection into csec_lst
            csec_lst[i1:i2] += fnew
            csec_nlst[i1:i2] += 1
            csec_maxlst[i1:i2] = np.maximum(csec_maxlst[i1:i2],fnew)

        nodes_lst[x1] = np.array(nodes_lst[x1])

        x1 += direction*scan_step
        if x1<20:
            # turn to the other direction
            direction = +1
            x1 = x0 + direction*scan_step
            x_lst[direction].append(x1)
            flux0 = flux1_center
            icol += 1
            continue
        elif x1 > w-20:
            break
        else:
            x_lst[direction].append(x1)
            flux0 = flux1
            icol += 1
            continue

    # filter the consecutive zero elements at the beginning and the end
    i_nonzero = np.nonzero(csec_nlst)[0]
    istart, iend = i_nonzero[0], i_nonzero[-1]
    csec_ylst = np.arange(csec_lst.size) + csec_i1

    # set the zero elements to 1, preparing for the division
    csec_nlst = np.maximum(csec_nlst, 1)
    csec_lst /= csec_nlst
    # convolve csec_lst
    smcore = np.hanning(seperation*2+1)
    smcore /= smcore.sum()
    csec_conv_lst = np.convolve(csec_lst, smcore, mode='same')


    # find aperture positions
    peaky, _ = get_local_minima(-csec_lst[istart:iend],
                                window=2*seperation+1)
    peaky += istart

    ax2.plot(csec_ylst[istart:iend], csec_lst[istart:iend], 'g-')
    ax2.set_yscale('log')
    ax2.set_xlim(csec_ylst[istart], csec_ylst[iend])

    #for x1,y_lst in nodes_lst.items():
    #    ax1.scatter(np.repeat(x1, y_lst.size), y_lst, c='b', s=5, lw=0)

    # parse peaks
    # cutx, cuty, cutn are the stacked peak list
    peak_lst = np.array(peak_lst)
    peak_ylst = peak_lst[:,0]
    peak_flst = peak_lst[:,1]
    peak_yintlst = np.int32(np.round(peak_ylst))
    cutf = np.zeros_like(csec_lst)
    cutn = np.zeros_like(csec_lst)
    for y,f in zip(peak_yintlst,peak_flst):
        cutn[y-csec_i1] += 1
        cutf[y-csec_i1] += f
    # remove those element equal to one
    onemask = cutn == 1
    cutf[onemask] = 0
    cutn[onemask] = 0
    cuty = np.arange(cutn.size) + csec_i1

    ax2.plot(cuty[istart:iend], cutn[istart:iend],'r-',alpha=1.)
    f1, f2 = ax2.get_ylim()

    # find central positions along Y axis for all apertures
    mid_lst = []
    for y in peaky:
        f = csec_lst[y]
        i1,i2 = y,y
        while(cutn[i1]>0):
            i1 -= 1
        while(cutn[i2]>0):
            i2 += 1
        n = cutn[max(i1,y-2):min(i2,y+2)].sum()
        if n > csec_nlst[y]*filling:
            ax2.plot([csec_ylst[y],csec_ylst[y]],
                     [f*(f2/f1)**0.01, f*(f2/f1)**0.03], 'k-', alpha=1.0)
            mid_lst.append(csec_ylst[y])

    aperture_set = ApertureSet(shape=(h,w))

    # generate a 2-D mesh grid
    yy, xx = np.mgrid[:h:,:w:]

    for aperture, mid in enumerate(mid_lst):
        xfit, yfit = [x0], [mid]
        for direction in [-1,1]:
            ystep = mid
            for ix, (slope, shift) in enumerate(zip(slope_lst[direction],
                                                    shift_lst[direction])):
                ystep = ystep*slope + shift
                x1 = x_lst[direction][ix]
                y_lst = nodes_lst[x1]
                # use (slope, shift) as nodes for polynomial
                xfit.append(x1)
                yfit.append(ystep)
                # or alternatively, use points as nodes for polynomial
                #diff = np.abs(y_lst - ystep)
                #dmin = diff.min()
                #imin = diff.argmin()
                #if dmin < 2:
                #    xfit.append(x1)
                #    yfit.append(y_lst[imin])

        # resort xfit and yfit
        xfit, yfit = np.array(xfit), np.array(yfit)
        argsort = xfit.argsort()
        xfit, yfit = xfit[argsort], yfit[argsort]
        #ax1.plot(xfit, yfit, 'ro-',lw=0.5,alpha=0.6)

        # fit chebyshev polynomial
        poly = Chebyshev.fit(xfit, yfit, domain=[0, w-1], deg=3)

        # generate a curve using for plot
        newx, newy = poly.linspace()
        ax1.plot(newx, newy, 'g-',lw=0.5, alpha=0.6)

        # initialize aperture position instance
        aperture_loc = ApertureLocation(direct='x', shape=(h,w))
        aperture_loc.set_position(poly)

        # generate a curve using for find saturation pixels
        center_line = aperture_loc.position(np.arange(w))

        # find approximate lower and upper boundaries of this order
        lower_bound = center_line - 3
        upper_bound = center_line + 3

        aperture_mask = (yy > lower_bound)*(yy < upper_bound)
        # find saturation mask
        aperture_sat = (np.int16(sat_mask)*aperture_mask).sum(axis=0)
        # find 1d mask of saturation pixels for this aperture
        sat_mask_1d = aperture_sat > 0
        # find how many saturated pixels in this aperture
        nsat = sat_mask_1d.sum()
        aperture_loc.nsat = nsat

        # get peak flux for this flat
        peak_flux = (data*aperture_mask).max(axis=0)

        aperture_loc.mean   = peak_flux.mean()
        aperture_loc.median = np.median(peak_flux)
        aperture_loc.max    = peak_flux.max()

        aperture_set[aperture] = aperture_loc

    # write the order set into an ascii file
    if result_file is not None:
        aperture_set.save_txt(result_file)

    fig.canvas.draw()
    fig.savefig(fig_file)
    plt.close(fig)

    return aperture_set

def load_aperture_set(filename):
    '''
    Reads an ApertureSet instance from an ascii file.
    
    Parameters
    ----------
    filename : string
        name of an ascii file
        
    Returns
    -------
    aperture_list : ApertureSet instance
        order set
    '''

    aperture_set = ApertureSet()

    magic = 'APERTURE LOCATION'

    # read input file
    infile = open(filename)
    order = None
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#!;':
            continue
        elif len(row)>len(magic) and row[0:len(magic)]==magic:
            aperture = int(row[len(magic):])
            aperture_set[aperture] = ApertureLocation()
        elif aperture is not None and '=' in row:
            g = row.split('=')
            key   = g[0].strip()
            value = g[1].strip()
            if key == 'position':
                value = eval(value)
                aperture_loc = aperture_set[aperture]
                n = aperture_loc.shape[aperture_loc.direct]
                poly = Chebyshev(coef=value, domain=[0, n-1])
                aperture_loc.set_position(poly)
            else:
                setattr(aperture_set[aperture],key,eval(value))
    infile.close()

    return aperture_set

def select_ref_tracefile(aperture_set_lst):
    '''
    Select the reference flat image.

    Parameters
    ----------
    order_set_lst : 
        Order set list

    Notes
    ------
    The reference flat shall be the brightest non-saturated flat image.
    If all flats are saturated, choose the image with the least saturated
    orders.
    '''

    tracename_lst = sorted(order_set_lst.keys())
    # flat_name_lst: flat_1, flat_2, ... flat_N
    # n_sat_lst:          0,      0, ...      5
    # median_lst:      1000,   2000, ...   9000
    n_sat_lst  = []
    median_lst = []
    for tracename, order_set in sorted(order_set_lst.iteritems()):

        n_sat_orders = 0
        median_count_lst = []

        # loop every order
        for order, order_loc in sorted(order_set.dict.iteritems()):
            if order_loc.nsat > 0:
                n_sat_orders += 1
            median_count_lst.append(order_loc.median)
        # find median of all orders for this flat
        median_median = np.median(median_count_lst)

        n_sat_lst.append(n_sat_orders)
        median_lst.append(median_median)

    # write to log file
    message = 'flatname,   n_sat_orders,   median_count:'+os.linesep
    message += os.linesep.join(['  %-12s %3d %9.2f'%(tracename,n_sat,median)
                    for tracename, n_sat, median
                    in zip(tracename_lst, n_sat_lst, median_lst)])
    logger.info(message)

    # choose the reference flat image
    if min(n_sat_lst)==0:
        # if there's at least one flat that is not saturated
        max_median = -1
        iname = None
        for i, tracename in enumerate(tracename_lst):
            if n_sat_lst[i] == 0 and median_lst[i] > max_median:
                max_median = median_lst[i]
                iname = i
    else:
        # find the flat with the least saturated orders
        min_sat = 999
        iname = None
        for i, tracename in enumerate(tracename_lst):
            if n_sat_lst[i] < min_sat:
                min_sat = n_sat_lst[i]
                iname = i

    # write to log
    logger.info(
            '"%s" is selected as the reference trace file'%tracename_lst[iname])

    return tracename_lst[iname]

def align_orders(order_set_lst, ref_tracename):
    '''
    Align echelle orders
    '''
    new_order_set_lst = {}

    tracename_lst = sorted(order_set_lst.keys())

    if ref_tracename not in tracename_lst:
        logger.error("Can't find referenced trace file '%s'"%ref_tracename)

    # generate the reference Y for each order in reference flat
    ref_order_lst = {}
    ref_order_set = order_set_lst[ref_tracename]
    # add reference orderset to new_order_set_lst
    new_order_set_lst[ref_tracename] = order_set_lst[ref_tracename]

    for order, order_loc in sorted(ref_order_set.dict.iteritems()):
        h, w = order_loc.shape
        anchor_x = np.linspace(0, w-1, 5, dtype=np.float32)[1:-1]
        ref_order_lst[order] = order_loc.get_polyval('center', anchor_x)

    # determine the offset between the flat image and the referenced flat
    for tracename, order_set in sorted(order_set_lst.iteritems()):
        # skip the reference trace file
        if tracename == ref_tracename:
            continue
        offset_lst = {}
        for order, order_loc in sorted(order_set.dict.iteritems()):
            h, w  = order_loc.shape
            anchor_x = np.linspace(0, w-1, 5, dtype=np.float32)[1:-1]
            y = order_loc.get_polyval('center', anchor_x)
            for order_ref, y_ref in sorted(ref_order_lst.iteritems()):
                diff = (np.abs(y-y_ref)).mean()
                if diff < 1:
                    offset = order - order_ref
                    offset_lst[order] = offset
                    break
            # if did not find, leave this order blanck

        # write short offsets to running log
        message_lst = []
        prev_order, prev_offset = None, None
        for order, offset in sorted(offset_lst.iteritems()):
            # o1 is the staring order of this offset
            if prev_order is None or prev_offset is None:
                o1 = order
            elif order != prev_order + 1 or offset != prev_offset:
                message_lst.append((o1,prev_order,prev_offset))
                o1 = order
            prev_order = order
            prev_offset = offset_lst[order]
        message_lst.append((o1, prev_order, prev_offset))

        string = ','.join(['%d-%d: %d'%(o1,o2,offset)
                            for o1, o2, offset in message_lst])
        logger.info('Order offsets for %s = %s'%(tracename, string))

        # determine the final offset
        maxorder = -1
        maxoffset = None
        for o1, o2, offset in message_lst:
            noffset = o2 - o1 + 1
            if noffset > maxorder:
                maxorder = noffset
                maxoffset = offset
        logger.info('Found offset for %s = %s'%(tracename, maxoffset))

        # now align all order with the final offset
        new_order_set = OrderSet()
        for order, order_loc in sorted(order_set.dict.iteritems()):
            new_order_set[order - maxoffset] = order_loc
        new_order_set_lst[tracename] = new_order_set

    # now all orders are aligned

    return new_order_set_lst