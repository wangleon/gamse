import re
import os
import math
import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage.filters import median_filter
import scipy.interpolate as intp
import scipy.signal as sg
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm     as cmap
import matplotlib.ticker as tck
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..echelle.trace      import ApertureSet
from ..echelle.imageproc  import savitzky_golay_2d
from ..utils.onedarray    import get_local_minima, get_edge_bin
from ..utils.regression   import get_clip_mean
from ..utils.regression2d import polyfit2d, polyval2d
from .imageproc           import table_to_array, array_to_table

def find_background2(data, mask, channels, apertureset_lst,
        method='poly', scale='linear', scan_step=200,
        xorder=2, yorder=2, maxiter=5, upper_clip=3, lower_clip=3,
        extend=True, display=True, fig_file=None, reg_file=None):
    """Subtract the background for an input FITS image.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Mask of input data image.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of :class:`~edrs.echelle.trace.ApertureSet`
            at different channels.
        method (str): Method of finding background light.
        scale (str): Scale of the 2-D polynomial fitting. If 'log', fit the
            polynomial in the logrithm scale.
        scan_step (int): Steps of scan in pixels.
        xorder (int): Order of 2D polynomial along the main dispersion
            direction (only applicable if **method** = "poly").
        yorder (int): Order of 2D polynomial along the cross-dispersion
            direction (only applicable if **method** = "poly").
        maxiter (int): Maximum number of iteration of 2D polynomial fitting
            (only applicable if **method** = "poly").
        upper_clip (float): Upper sigma clipping threshold (only applicable if
            **method** = "poly").
        lower_clip (float): Lower sigma clipping threshold (only applicable if
            **method** = "poly").
        extend (bool): Extend the grid to the whole CCD image if *True*.
        display (bool): Display figures on the screen if *True*.
        fig_file (str): Name of the output figure. No image file generated if
            *None*.
        reg_file (string): Name of the output DS9 region file. No file generated
            if *None*.

    Returns:
        :class:`numpy.ndarray`: Image of background light.
    """
    
    plot = (display or fig_file is not None)

    plot_paper_fig = False

    h, w = data.shape

    meddata = median_filter(data, size=(3,3), mode='reflect')

    xnodes, ynodes, znodes = [], [], []

    # find the minimum and maximum aperture number
    min_aper = min([min(apertureset_lst[ch].keys()) for ch in channels])
    max_aper = max([max(apertureset_lst[ch].keys()) for ch in channels])

    # generate the horizontal scan list
    x_lst = np.arange(0, w-1, scan_step)
    # add the last column to the list
    if x_lst[-1] != w-1:
        x_lst = np.append(x_lst, w-1)

    # find intra-order pixels
    _message_lst = ['Column, N (between), N (extend), N (removed), N (total)']
    for x in x_lst:
        xsection = meddata[:,x]
        inter_aper = []
        prev_newy = None
        # loop for every aperture
        for aper in range(min_aper, max_aper+1):
            # for a new aperture, initialize the count of channel
            count_channel = 0
            for ich, channel in enumerate(channels):
                # check every channel in this frame
                if aper in apertureset_lst[channel]:
                    count_channel += 1
                    this_newy = apertureset_lst[channel][aper].position(x)
                    if count_channel == 1 and prev_newy is not None:
                        # this channel is the first channel in this aperture and
                        # there is a previous y
                        mid_newy = (prev_newy + this_newy)//2
                        i1 = min(h-1, max(0, int(prev_newy)))
                        i2 = min(h-1, max(0, int(this_newy)))
                        #if len(inter_aper)==0 or \
                        #    abs(mid_newy - inter_aper[-1])>scan_step*0.7:
                        #    if i2-i1>0:
                        if i2-i1>0:
                                mid_newy = i1 + xsection[i1:i2].argmin()
                                inter_aper.append(mid_newy)
                    prev_newy = this_newy

        inter_aper = np.array(inter_aper)

        # count how many nodes found between detected orders
        n_nodes_inter = inter_aper.size

        # if extend = True, expand the grid with polynomial fitting to
        # cover the whole CCD area
        n_nodes_extend = 0
        if extend:
            if x==2304:
                _fig = plt.figure(dpi=150)
                _ax = _fig.gca()
                for _x in inter_aper:
                    _ax.axvline(x=_x,color='g', ls='--',lw=0.5, alpha=0.6)
                _ax.plot(data[:, x],'b-',lw=0.5)
                _fig2 = plt.figure(dpi=150)
                _ax2 = _fig2.gca()
            print(inter_aper)

            coeff = np.polyfit(np.arange(inter_aper.size), inter_aper, deg=3)
            if x== 2304:
                _ax2.plot(np.arange(inter_aper.size), inter_aper,'go', alpha=0.6)
                _newx = np.arange(0, inter_aper.size, 0.1)
                _ax2.plot(_newx, np.polyval(coeff, _newx),'g-')
            # find the points after the end of inter_aper
            ii = inter_aper.size-1
            new_y = inter_aper[-1]
            while(new_y<h-1):
                ii += 1
                new_y = int(np.polyval(coeff,ii))
                inter_aper = np.append(inter_aper,new_y)
                n_nodes_extend += 1
            # find the points before the beginning of order_mid
            ii = 0
            new_y = inter_aper[0]
            while(new_y>0):
                ii -= 1
                new_y = int(np.polyval(coeff,ii))
                inter_aper = np.insert(inter_aper,0,new_y)
                n_nodes_extend += 1
            if x==2304:
                #for _x in np.polyval(coeff, np.arange(0,25)):
                #    _ax.axvline(x=_x, color='r',ls='--',lw=0.5)
                #_newx = np.arange(0, 25)
                #_ax2.plot(_newx, np.polyval(coeff, _newx), 'ro', alpha=0.6)
                plt.show()

        # remove those points with y<0 or y>h-1
        m1 = inter_aper > 0
        m2 = inter_aper < h-1
        inter_aper = inter_aper[np.nonzero(m1*m2)[0]]

        # filter those masked pixels
        m = mask[inter_aper, x]==0
        inter_aper = inter_aper[m]

        # remove backward points
        tmp = np.insert(inter_aper,0,0.)
        m = np.diff(tmp)>0
        inter_aper = inter_aper[np.nonzero(m)[0]]

        # count how many nodes removed
        n_nodes_removed = (n_nodes_inter + n_nodes_extend) - inter_aper.size

        # pack infos into message list
        _message_lst.append('| %6d | %6d | %6d | %6d | %6d |'%(
            x, n_nodes_inter, n_nodes_extend, n_nodes_removed, inter_aper.size))

        # pack all nodes
        for y in inter_aper:
            xnodes.append(x)
            ynodes.append(y)
            znodes.append(meddata[y,x])

        # extrapolate
        #if extrapolate:
        if False:
            _y0, _y1 = inter_aper[0], inter_aper[1]
            newy = _y0 - (_y1 - _y0)
            newz = meddata[_y0, x] - (meddata[_y1, x] - meddata[_y0, x])
            xnodes.append(x)
            ynodes.append(newy)
            znodes.append(newz)

            _y1, _y2 = inter_aper[-2], inter_aper[-1]
            newy = _y2 + (_y2 - _y1)
            newz = meddata[_y2, x] + (meddata[_y2, x] - meddata[_y1, x])
            xnodes.append(x)
            ynodes.append(newy)
            znodes.append(newz)


    # convert to numpy array
    xnodes = np.array(xnodes)
    ynodes = np.array(ynodes)
    znodes = np.array(znodes)

    # write to running log
    _message_lst.append('Total: %4d'%xnodes.size)
    logger.info((os.linesep+'  ').join(_message_lst))

    # if scale='log', filter the negative values
    if scale=='log':
        pmask = znodes > 0
        znodes[~pmask] = znodes[pmask].min()
        znodes = np.log10(znodes)

    if plot:
        # initialize figures
        fig = plt.figure(figsize=(10,10), dpi=150)
        ax11 = fig.add_axes([0.07, 0.54, 0.39,  0.39])
        ax12 = fig.add_axes([0.52, 0.54, 0.39,  0.39])
        ax13 = fig.add_axes([0.94, 0.54, 0.015, 0.39])
        ax21 = fig.add_axes([0.07, 0.07, 0.39,  0.39], projection='3d')
        ax22 = fig.add_axes([0.52, 0.07, 0.39,  0.39], projection='3d')

        fig.suptitle('Background')
        ax11.imshow(data, cmap='gray')

        # plot nodes
        for ax in [ax11, ax12]:
            ax.set_xlim(0,w-1)
            ax.set_ylim(h-1,0)
            ax.set_xlabel('X (pixel)', fontsize=10)
            ax.set_ylabel('Y (pixel)', fontsize=10)
        for ax in [ax21, ax22]:
            ax.set_xlim(0,w-1)
            ax.set_ylim(0,h-1)
            ax.set_xlabel('X (pixel)', fontsize=10)
            ax.set_ylabel('Y (pixel)', fontsize=10)
        for ax in [ax11, ax12]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(9)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(9)
        for ax in [ax21, ax22]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(9)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(9)
            for tick in ax.zaxis.get_major_ticks():
                tick.label1.set_fontsize(9)

        if display:
            plt.show(block=False)

        # plot the figure used in paper
        if plot_paper_fig:
            figp1 = plt.figure(figsize=(6,6), dpi=150)
            axp1 = figp1.add_axes([0.00, 0.05, 1.00, 0.95], projection='3d')
            figp2 = plt.figure(figsize=(6.5,6), dpi=150)
            axp2 = figp2.add_axes([0.12, 0.1, 0.84, 0.86])

    if method=='poly':
        background_data, fitmask = fit_background(data.shape,
                xnodes, ynodes, znodes, xorder=xorder, yorder=yorder,
                maxiter=maxiter, upper_clip=upper_clip, lower_clip=lower_clip)
    elif method=='interp':
        background_data, fitmask = interpolate_background(data.shape,
                xnodes, ynodes, znodes)
    else:
        print('Unknown method: %s'%method)

    m = (ynodes >= 0)*(ynodes <= h-1)
    xnodes = xnodes[m]
    ynodes = ynodes[m]
    znodes = znodes[m]
    fitmask = fitmask[m]

    if scale=='log':
        background_data = np.power(10, background_data)

    # save nodes to DS9 region file
    if reg_file is not None:
        outfile = open(reg_file, 'w')
        outfile.write('# Region file format: DS9 version 4.1'+os.linesep)
        outfile.write('global color=green dashlist=8 3 width=1 ')
        outfile.write('font="helvetica 10 normal roman" select=1 highlite=1 ')
        outfile.write('dash=0 fixed=0 edit=1 move=1 delete=1 include=1 ')
        outfile.write('source=1'+os.linesep)
        outfile.write('physical'+os.linesep)
        for x, y in zip(xnodes, ynodes):
            text = ('point(%4d %4d) # point=circle'%(x+1, y+1))
            outfile.write(text+os.linesep)
        outfile.close()

    # write nodes to running log
    message = ['Background Nodes:', ' x,    y,    value,  mask']
    for x,y,z,m in zip(xnodes, ynodes, znodes, fitmask):
        message.append('| %4d | %4d | %+10.8e | %1d |'%(x,y,z,m))
    logger.info((os.linesep+' '*4).join(message))

    residual = znodes - background_data[ynodes, xnodes]

    if plot:
        # prepare for plotting the fitted surface with a loose grid
        yy, xx = np.meshgrid(np.linspace(0,h-1,32), np.linspace(0,w-1,32))
        yy = np.int16(np.round(yy))
        xx = np.int16(np.round(xx))
        zz = background_data[yy, xx]

        # plot 2d fitting in a 3-D axis in fig2
        # plot the linear fitting
        ax21.set_title('Background fitting (%s Z)'%scale, fontsize=10)
        ax22.set_title('residuals (%s Z)'%scale, fontsize=10)
        ax21.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
                          linewidth=0, antialiased=True, alpha=0.5)
        ax21.scatter(xnodes[fitmask], ynodes[fitmask], znodes[fitmask],
                    color='C0', linewidth=0)
        ax22.scatter(xnodes[fitmask], ynodes[fitmask], residual[fitmask],
                    color='C0', linewidth=0)
        if (~fitmask).sum()>0:
            ax21.scatter(xnodes[~fitmask], ynodes[~fitmask], znodes[~fitmask],
                        color='none', edgecolor='C0', linewidth=1)
            ax22.scatter(xnodes[~fitmask], ynodes[~fitmask], residual[~fitmask],
                        color='none', edgecolor='C0', linewidth=1)

        # plot the logrithm fitting in another fig
        #if scale=='log':
        #    ax23.plot_surface(xx, yy, log_zz, rstride=1, cstride=1, cmap='jet',
        #                        linewidth=0, antialiased=True, alpha=0.5)
        #    ax23.scatter(xnodes[fitmask], ynodes[fitmask], zfit[fitmask],         linewidth=0)
        #    ax24.scatter(xnodes[fitmask], ynodes[fitmask], log_residual[fitmask], linewidth=0)

        for ax in [ax21, ax22]:
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))

        if display: fig.canvas.draw()

        # plot figure for paper
        if plot_paper_fig:
            axp1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
                                linewidth=0, antialiased=True, alpha=0.5)
            axp1.scatter(xnodes[fitmask], ynodes[fitmask], znodes[fitmask], linewidth=0)
            axp1.xaxis.set_major_locator(tck.MultipleLocator(500))
            axp1.xaxis.set_minor_locator(tck.MultipleLocator(100))
            axp1.yaxis.set_major_locator(tck.MultipleLocator(500))
            axp1.yaxis.set_minor_locator(tck.MultipleLocator(100))
            axp1.set_xlim(0, w-1)
            axp1.set_ylim(0, h-1)
            axp1.set_xlabel('X')
            axp1.set_ylabel('Y')
            axp1.set_zlabel('Count')

    if plot:
        # plot the accepted nodes in subfig 1
        ax11.scatter(xnodes[fitmask], ynodes[fitmask],
                    c='r', s=6, linewidth=0, alpha=0.8)
        # plot the rejected nodes
        if (~fitmask).sum()>0:
            ax11.scatter(xnodes[~fitmask], ynodes[~fitmask],
                    c='none', s=6, edgecolor='r', linewidth=0.5)

        # plot subfig 2
        cnorm = colors.Normalize(vmin = background_data.min(),
                                 vmax = background_data.max())
        scalarmap = cmap.ScalarMappable(norm=cnorm, cmap=cmap.jet)
        # plot the background light
        image = ax12.imshow(background_data, cmap=scalarmap.get_cmap())
        # plot the accepted nodes
        ax12.scatter(xnodes[fitmask], ynodes[fitmask],
                    c='k', s=6, linewidth=0.5)
        # plot the rejected nodes
        if (~fitmask).sum()>0:
            ax12.scatter(xnodes[~fitmask], ynodes[~fitmask],
                        c='none', s=6, edgecolor='k', linewidth=0.5)

        # set colorbar
        plt.colorbar(image, cax=ax13)
        # set font size of colorbar
        for tick in ax13.get_yaxis().get_major_ticks():
            tick.label2.set_fontsize(9)

        if display: fig.canvas.draw()

        # plot for figure in paper
        if plot_paper_fig:
            pmask = data>0
            logdata = np.zeros_like(data)-1
            logdata[pmask] = np.log(data[pmask])
            axp2.imshow(logdata, cmap='gray')
            axp2.scatter(xnodes, ynodes, c='b', s=8, linewidth=0, alpha=0.8)
            cs = axp2.contour(background_data, linewidth=1, cmap='jet')
            axp2.clabel(cs, inline=1, fontsize=11, fmt='%d', use_clabeltext=True)
            axp2.set_xlim(0, w-1)
            axp2.set_ylim(h-1, 0)
            axp2.set_xlabel('X')
            axp2.set_ylabel('Y')
            figp1.savefig('fig_background1.png')
            figp2.savefig('fig_background2.png')
            figp1.savefig('fig_background1.pdf')
            figp2.savefig('fig_background2.pdf')
            plt.close(figp1)
            plt.close(figp2)

    if fig_file is not None:
        fig.savefig(fig_file)
    plt.close(fig)

    return background_data


def fit_background(shape, xnodes, ynodes, znodes, xorder=2, yorder=2,
    maxiter=5, upper_clip=3, lower_clip=3):
    """Find the background light by fitting a 2D polynomial.

    Args:
        shape (tuple): Shape of image.
        xnodes (:class:`numpy.ndarray`): List of X coordinates of the nodes.
        ynodes (:class:`numpy.ndarray`): List of Y coordinates of the nodes.
        znodes (:class:`numpy.ndarray`): List of pixel values of the nodes.
        xorder (int): Order of 2D polynomial along the main dispersion
            direction.
        yorder (int): Order of 2D polynomial along the cross-dispersion
            direction.
        maxiter (int): Maximum number of iteration of 2D polynomial fitting.
        upper_clip (float): Upper sigma clipping threshold.
        lower_clip (float): Lower sigma clipping threshold.

    Returns:
        tuple: A tuple containing:

            * **background_data** (:class:`numpy.ndarray`) – Array of background
              light.
            * **mask** (:class:`numpy.ndarray`) – Mask of used nodes in the
              fitting.

    See also:
        :func:`interpolate_background`
    """

    h, w = shape
    # normalize to 0 ~ 1 for x and y nodes
    xfit = np.float64(xnodes)/w
    yfit = np.float64(ynodes)/h
    zfit = znodes

    # fit the 2-d polynomial
    _messages = [
        'Polynomial Background Fitting Xorder=%d, Yorder=%d:'%(xorder, yorder)
        ]
    mask = np.ones_like(zfit, dtype=np.bool)

    for niter in range(maxiter):
        coeff = polyfit2d(xfit[mask], yfit[mask], zfit[mask],
                          xorder=xorder, yorder=yorder)
        values = polyval2d(xfit, yfit, coeff)
        residuals = zfit - values
        sigma = residuals[mask].std(dtype=np.float64)
        m1 = residuals < upper_clip*sigma
        m2 = residuals > -lower_clip*sigma
        new_mask = m1*m2

        # write info to running log
        _message = 'Iter. %d: std=%10.6f, N=%4d, N(new)=%4d'%(
            niter, sigma, mask.sum(), new_mask.sum())
        _messages.append(_message)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    logger.debug((os.linesep+' '*4).join(_messages))

    yy, xx = np.mgrid[:h:, :w:]
    background_data = polyval2d(xx/w, yy/h, coeff)
    return background_data, mask

def interpolate_background(shape, xnodes, ynodes, znodes):
    """Find the background light by interpolating 2D cubic splines.

    Args:
        shape (tuple): Shape of image.
        xnodes (:class:`numpy.ndarray`): List of X coordinates of the nodes.
        ynodes (:class:`numpy.ndarray`): List of Y coordinates of the nodes.
        znodes (:class:`numpy.ndarray`): List of pixel values of the nodes.

    Returns:
        tuple: A tuple containing:

            * **background_data** (:class:`numpy.ndarray`) – Array of background
              light.
            * **mask** (:class:`numpy.ndarray`) – Mask of used nodes in the
              fitting.

    See also:
        :func:`fit_background`
    """
    h, w = shape
    yy, xx = np.mgrid[:h:, :w:]
    background_data = intp.griddata((xnodes, ynodes), znodes, (xx, yy),
            rescale=True, method='cubic')
    mask = np.ones_like(znodes, dtype=np.bool)

    # fix non values
    notnan_mask = ~np.isnan(background_data)
    for j in np.arange(w):
        array = background_data[:, j]
        m = notnan_mask[:, j]
        notnan_index = np.nonzero(m)[0]
        i1 = notnan_index[0]
        if i1 > 0:
            background_data[0:i1, j] = array[i1]
        i2 = notnan_index[-1]
        if i2 < h-1:
            background_data[i2+1:, j] = array[i2]

    return background_data, mask

def find_background(data, mask, aperturesets, ncols, distance,
        yorder=7, ymaxiter=5, yupper_clip=3, ylower_clip=3,
        fig_stray=None, fig_section=None):
    """Subtract the background for an input FITS image.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Mask of input data image.
        aperturesets (:class:`~gamse.echelle.trace.ApertureSet` or dict):
            A :class:`~gamse.echelle.trace.ApertureSet` instance, or a dict of
            :class:`~gamse.echelle.trace.ApertureSet` at different channels.
        yorder (int): Order of polynomial along the cross-dispersion
            direction.
        fig_stray (str): Name of the figure showing stray light. No file
            generated if *None*.
        fig_section (str): Name of the figure showing cross-sections. No file
            generated if *None*.

    Returns:
        :class:`numpy.ndarray`: Image of background light. It has the same shape
            and datatype as the arg **data**.
    """

    h, w = data.shape
    cols = np.int32(np.round(np.linspace(1, w-2, ncols)))

    # prepare for cross-section figure
    plot_section = (fig_section is not None)
    if plot_section:
        plot_cols = [cols[np.abs(cols - h*t).argmin()]
                        for t in np.linspace(0, 1, 5)]
        fig1 = plt.figure(figsize=(18,12), dpi=150)
        tick_size  = 13
        label_size = 14

    ally = np.arange(h)

    # prepare interpolation grid
    grid = []

    # parse apertureset_lst
    if isinstance(aperturesets, ApertureSet):
        # put input aperture in a dict
        apertureset_lst = {'A': aperturesets}
    elif isinstance(aperturesets, dict):
        apertureset_lst = aperturesets
    else:
        print('Unknown aperturesets:',aperturesets)
        exit()

    for x in cols:
        xsection = np.median(data[:,x-1:x+2], axis=1)
        intermask = np.ones(h, dtype=np.bool)
        allimin, _ = get_local_minima(xsection)
        allmin = np.array([v in allimin for v in np.arange(h)])

        if plot_section and x in plot_cols:
            i = plot_cols.index(x)
            ax1 = fig1.add_axes([0.05, (4-i)*0.19+0.05, 0.93, 0.18])
            #ax2 = ax1.twinx()
            ax1.plot(xsection, ls='-' ,color='C0', lw=0.5, alpha=0.2)

        for ichannel, (channel, apertureset) in enumerate(sorted(apertureset_lst.items())):
            # predict post- and pre-aperutres.
            # post- and pre-apertures are virtual apertures that are not
            # identified by the order detection function, probabaly because they
            # are too weak
            y_lst, aper_lst = [], []
            apercen_lst = []
            for aper, aperloc in apertureset.items():
                y = aperloc.position(x)
                apercen_lst.append(y)
                y_lst.append(y)
                aper_lst.append(aper)
            y_lst = np.array(y_lst)
            aper_lst = np.array(aper_lst)
            apercen_lst = np.array(apercen_lst)
            coeff = np.polyfit(aper_lst, y_lst, deg=3)
     
            # find post apertures
            aper = aper_lst[-1]
            post_aper_lst, post_apercen_lst = [], []
            while(True):
                aper += 1
                y = np.polyval(coeff, aper)
                if 0 < y < h-1:
                    post_aper_lst.append(aper)
                    post_apercen_lst.append(y)
                else:
                    break
            post_aper_lst = np.array(post_aper_lst)
            post_apercen_lst = np.array(post_apercen_lst)
     
            # find pre apertures
            aper = aper_lst[0]
            pre_aper_lst, pre_apercen_lst = [], []
            while(True):
                aper -= 1
                y = np.polyval(coeff, aper)
                if 0 < y < h-1:
                    pre_aper_lst.append(aper)
                    pre_apercen_lst.append(y)
                else:
                    break
            pre_aper_lst = np.array(pre_aper_lst)
            pre_apercen_lst = np.array(pre_apercen_lst)

            '''
            # plot aper_lst, pre-aperture list, and post-aperture list
            if plot_section and x in plot_cols:
                _color = 'C%d'%ichannel
                ax2.plot(y_lst, aper_lst, 'o',
                        color=_color, ms=3, alpha=0.5)
                ax2.plot(post_apercen_lst, post_aper_lst, '^',
                        color=_color, ms=3, alpha=0.5)
                ax2.plot(pre_apercen_lst, pre_aper_lst, 'v',
                        color=_color, ms=3, alpha=0.5)
                _newx = np.arange(aper_lst[0], aper_lst[-1], 0.1)
                ax2.plot(np.polyval(coeff, _newx), _newx, '-',
                        color=_color, lw=1, alpha=0.5)
            '''

            for y in np.concatenate((apercen_lst, post_apercen_lst, pre_apercen_lst)):
                mask = np.abs(ally - y) > distance
                intermask *= mask
                if plot_section and x in plot_cols:
                    # plot order center using vertical lines
                    ax1.axvline(x=y, color='C%d'%ichannel,
                                ls='--', lw=0.5, alpha=0.3)

        if plot_section and x in plot_cols:
            _yplot = np.copy(xsection)
            _yplot[~intermask] = np.NaN
            ax1.plot(_yplot, '-', color='C0', lw=0.7, alpha=0.5)

        notnanindex = np.nonzero(intermask)[0]
        group_lst = np.split(notnanindex, np.where(np.diff(notnanindex)!=1)[0]+1)
        fitx_lst, fity_lst, fityerr_lst = [], [], []
        for group in group_lst:
            ydata = xsection[group]
            local_min = allmin[group]
            idx = np.nonzero(local_min)[0]
            if idx.size == 0:
                continue
            if idx.size == 1:
                meanx = ydata.argmin() + group[0]
                mean, std = ydata.min(),0
            else:
                i1, i2 = idx[0], idx[-1]+1
                mean, std, m = get_clip_mean(ydata[i1:i2], high=2, low=3, maxiter=10)
                meanx = np.arange(i1, i2)[m].mean() + group[0]
                if mean > 0 and std/math.sqrt(mean) > 2 and (i2-i1) >= 5:
                    # remove the two largest points
                    m = ydata[i1:i2].argsort().argsort() < i2-i1-2
                    mean, std, m = get_clip_mean(ydata[i1:i2], mask=m, high=2, low=3, maxiter=10)
                    meanx = np.arange(i1,i2)[m].mean() + group[0]
                ii1 = i1 + group[0]
                ii2 = i2 + group[0]

                if plot_section and x in plot_cols:
                    ax1.plot(np.arange(ii1, ii2), xsection[ii1:ii2], ls='-',
                            color='C3', lw=0.8, alpha=0.8)
                    if m.sum() < m.size:
                        ax1.plot(np.arange(ii1, ii2)[~m], xsection[ii1:ii2][~m],
                            'o', color='gray', ms=3, lw=1, alpha=0.5)
    
            fitx_lst.append(meanx)
            fity_lst.append(mean)
            fityerr_lst.append(std)
            #print('%4d %4d %10.6f %10.6f %10.6f'%(
            #    group[0], group[-1], mean, std, std/math.sqrt(abs(mean))))
    
        fitx_lst    = np.array(fitx_lst)
        fity_lst    = np.array(fity_lst)
        fityerr_lst = np.array(fityerr_lst)

        maxiter = 5
        mask = fity_lst > 0
        for ite in range(maxiter):
            coeff = np.polyfit(fitx_lst[mask]/h, np.log(fity_lst[mask]), deg=yorder)
            yres = np.log(fity_lst) - np.polyval(coeff, fitx_lst/h)
            std = yres[mask].std()
            allflux = np.polyval(coeff, ally/h)
            allflux = np.exp(allflux)
            
            new_mask = (yres < 2.*std)*(yres > -5.*std)
            if new_mask.sum()==mask.sum():
                break
            mask = new_mask
        grid.append(allflux)

        if plot_section and x in plot_cols:
            # plot fitx and fity with errorbars
            ax1.errorbar(fitx_lst[~mask], fity_lst[~mask], yerr=fityerr_lst[~mask],
                        fmt='o', mfc='w', mec='C2', ms=3, mew=1,
                        ecolor='C2', elinewidth=0.8, alpha=0.8)
            ax1.errorbar(fitx_lst[mask], fity_lst[mask], yerr=fityerr_lst[mask],
                        fmt='o', mfc='C2', mec='C2', ms=3, mew=1,
                        ecolor='C2', elinewidth=0.8, alpha=0.8)
            ax1.plot(ally, allflux, '-', color='C2', lw=0.7)

            _ymin, _ymax = fity_lst[mask].min(), fity_lst[mask].max()
            _y1, _y2 = 1.2*_ymin-0.2*_ymax, 1.2*_ymax-0.2*_ymin
            ax1.set_ylim(_y1, _y2)
            ax1.set_xlim(0, h-1)
            ax1.text(0.03*h, 0.8*_y1+0.2*_y2, 'x=%d'%x,
                    fontsize=label_size, alpha=0.8)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)
            for tick in ax1.yaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)
            #for tick in ax2.yaxis.get_major_ticks():
            #    tick.label2.set_fontsize(tick_size)
            #    tick.label2.set_color('C0')
            #for tickline in ax2.yaxis.get_ticklines():
            #    tickline.set_color('C0')
            if i < 4:
                ax1.set_xticklabels([])
            else:
                ax1.set_xlabel('Y', fontsize=label_size)
            ax1.set_ylabel('Flux', fontsize=label_size)
            #ax2.set_ylabel('Aperture Number', fontsize=label_size, color='C0')

    if plot_section:
        fig1.savefig(fig_section)
        plt.close(fig1)

    grid = np.array(grid)
    stray = np.zeros_like(data, dtype=data.dtype)

    for y in np.arange(h):
        f = intp.InterpolatedUnivariateSpline(cols, grid[:,y], k=3)
        stray[y,:] = f(np.arange(w))

    return stray

def simple_debackground(data, mask, xnodes, smooth=20, maxiter=10, deg=3):
    """
    """

    h, w = data.shape
    allx = np.arange(h)

    if smooth is not None:
        core = np.hanning(smooth)
        core = core/core.sum()

    # prepare interpolation grid
    grid = []
    for x in xnodes:
        section   = data[:, x]
        sect_mask = mask[:, x]>0

        if sect_mask.sum() > 0:
            f = intp.InterpolatedUnivariateSpline(allx[~sect_mask], section[~sect_mask], k=3, ext=3)
            section = f(allx)
        
        if smooth is not None:
            section_new = np.convolve(section, core, mode='same')
        else:
            section_new = section

        allimin, allmin = get_local_minima(section_new)
        # remove the first and last local minima
        m = (allimin>0) * (allimin<h-1)
        allimin = allimin[m]
        allmin  = allmin[m]

        #allimin, allmin = get_local_minima(section)
        #mask = np.ones_like(allmin, dtype=np.bool)
        fitmask = (allmin > 0)*(sect_mask[allimin]==0)
        for i in range(maxiter):
            coeff = np.polyfit(allimin[fitmask]/h, np.log(allmin[fitmask]), deg=deg)
            #res_lst = allmin - np.exp(np.polyval(coeff, allimin/h))
            res_lst = np.log(allmin) - np.polyval(coeff, allimin/h)
            std = res_lst[fitmask].std()
            mask1 = res_lst < 3*std
            mask2 = res_lst > -3*std
            new_fitmask = mask1*mask2
            if new_fitmask.sum() == fitmask.sum():
                break
            else:
                fitmask = fitmask*new_fitmask

        logbkg = np.polyval(coeff, allx/h)
        linbkg = np.exp(logbkg)
        ######################## plot #####################
        #figname = 'bkg-b-%04d.png'%x
        figname = None
        if figname is not None:
            fig = plt.figure(dpi=150, figsize=(10,8))
            ax = fig.gca()
            ax.plot(allx, linbkg, color='C0')
            #ax.plot(allx, linbkg+std, color='C0', ls='--')
            #ax.plot(allx, linbkg-std, color='C0', ls='--')
            ax.plot(allx, section, color='C1', lw=0.5)
            if smooth is not None:
                ax.plot(allx, section_new, color='C2', lw=0.5)
            ax.scatter(allimin, allmin, c='C3', s=10)
            ax.set_yscale('log')
            plt.savefig(figname)
            plt.close(fig)
        ###################################################

        logbkg = np.polyval(coeff, allx/h)
        grid.append(logbkg)

    # interpolate the whole image
    grid = np.array(grid)
    stray = np.zeros_like(data, dtype=data.dtype)
    for y in np.arange(h):
        f = intp.InterpolatedUnivariateSpline(xnodes, grid[:,y], k=3)
        stray[y, :] = f(np.arange(w))

    pmask = data>0
    corrected_data = np.zeros_like(data)
    corrected_data[pmask] = np.log(data[pmask]) - stray[pmask]
    return np.exp(corrected_data)


def get_single_background(data, apertureset):
    #apertureset = load_aperture_set('../midproc/trace_A.trc')
    h, w = data.shape

    bkg_image = np.zeros_like(data, dtype=np.float32)
    allrows = np.arange(h)
    plot_x = []
    for x in np.arange(w):
        if x in plot_x:
            plot = True
        else:
            plot = False
        mask_rows = np.zeros_like(allrows, dtype=np.bool)
        for aper, aperloc in sorted(apertureset.items()):
            ycen = aperloc.position(x)
            if plot:
                ax01.axvline(x=ycen, color='C0', ls='--', lw=0.5, alpha=0.4)
                ax02.axvline(x=ycen, color='C0', ls='--', lw=0.5, alpha=0.4)
    
            imask = np.abs(allrows - ycen)<7
            mask_rows += imask
        if plot:
            ax01.plot(allrows, data[:, x], color='C0', alpha=0.3, lw=0.7)
        x_lst, y_lst = [], []
        for (y1, y2) in get_edge_bin(~mask_rows):
            if plot:
                ax01.plot(allrows[y1:y2], data[y1:y2,x], color='C0', alpha=1, lw=0.7)
                ax02.plot(allrows[y1:y2], data[y1:y2,x], color='C0', alpha=1, lw=0.7)
            if y2-y1>1:
                yflux = data[y1:y2, x]
                xlist = np.arange(y1, y2)
                _m = xlist == y1 + np.argmax(yflux)
                mean = yflux[~_m].mean()
                std  = yflux[~_m].std()
                if yflux.max() < mean + 3.*std:
                    meany = yflux.mean()
                    meanx = (y1+y2-1)/2
                else:
                    meanx = xlist[~_m].mean()
                    meany = mean
            else:
                meany = data[y1,x]
                meanx = y1
            x_lst.append(meanx)
            y_lst.append(meany)
        x_lst = np.array(x_lst)
        y_lst = np.array(y_lst)
        y_lst = np.maximum(y_lst, 0)
        y_lst = sg.medfilt(y_lst, 3)
        f = intp.InterpolatedUnivariateSpline(x_lst, y_lst, k=3, ext=3)
        bkg = f(allrows)
        bkg_image[:, x] = bkg
        if plot:
            ax01.plot(x_lst, y_lst, 'o', color='C3', ms=3)
            ax02.plot(x_lst, y_lst, 'o', color='C3', ms=3)
            ax01.plot(allrows, bkg, ls='-', color='C3', lw=0.7, alpha=1)
            ax02.plot(allrows, bkg, ls='-', color='C3', lw=0.7, alpha=1)
            _y1, _y2 = ax02.get_ylim()
            ax02.plot(allrows, data[:, x], color='C0', alpha=0.3, lw=0.7)
            ax02.set_ylim(_y1, _y2)
    
    bkg_image = median_filter(bkg_image, size=(9,1), mode='nearest')
    #fits.writeto('bkg_{}.fits'.format(fileid), bkg_image, overwrite=True)
    bkg_image = savitzky_golay_2d(bkg_image, window_length=(21, 101), order=3, mode='nearest')
    #fits.writeto('bkg_{}_sm.fits'.format(fileid), bkg_image, overwrite=True)
    return bkg_image



def get_xdisp_profile(data, apertureset):
    """Get brightness profile along the cross-dispersion direction.

    Args:
        data (numpy.ndarray):
        apertureset ():

    Returns:
        tuple: A tuple containing:

            * list of aperture numbers
            * list of aperture positions
            * list of aperture britness
    """
    # get order brightness profile
    ny, nx = data.shape
    yy, xx = np.mgrid[:ny:, :nx:]
    
    aper_num_lst, aper_brt_lst, aper_pos_lst = [], [], []
    x_lst = np.arange(nx)
    for aper, aperloc in sorted(apertureset.items()):
        ycen_lst = aperloc.position(x_lst)
        m1 = yy > ycen_lst - 1
        m2 = yy < ycen_lst + 2
        mask_image = m1*m2
        maxflux_lst = (data*mask_image).max(axis=0)
        # maxflux is a spectrum but with maximum values in each pixel
        brightness = np.percentile(maxflux_lst, 99)
        aper_num_lst.append(aper)
        aper_brt_lst.append(brightness)
        aper_pos_lst.append(aperloc.position(nx//2))
    aper_num_lst = np.array(aper_num_lst)
    aper_brt_lst = np.array(aper_brt_lst)
    aper_pos_lst = np.array(aper_pos_lst)
    
    return aper_num_lst, aper_pos_lst, aper_brt_lst

def find_profile_scale(input_profile, ref_profile):
    """Find the scaling factor of two brightness profiles.

    """
    fitfunc = lambda s: ref_profile*s
    errfunc = lambda s: input_profile - fitfunc(s)
    s0 = np.median(input_profile)/np.median(ref_profile)
    fitres = opt.least_squares(errfunc, s0)
    s = fitres.x[0]
    return s

class BackgroundLight(object):
    def __init__(self, info=None, header=None, data=None, aper_num_lst=None,
            aper_ord_lst=None, aper_pos_lst=None, aper_brt_lst=None,
            aper_wav_lst=None):
        """
        """
        self.info   = info
        self.header = header
        self.data   = data
        self.aper_num_lst = aper_num_lst
        self.aper_ord_lst = aper_ord_lst
        self.aper_pos_lst = aper_pos_lst
        self.aper_brt_lst = aper_brt_lst
        self.aper_wav_lst = aper_wav_lst

    def get_wavelength(self, aperture=None, order=None):
        """Get wavelength of a specific aperture or order.

        Args:
            aperture (int): Aperture number.
            order (int): Order number.

        Returns:
            *float*: wavelength of the specific aperture or order.
        """
        if aperture is not None and order is None:
            # aperture is given and order is NOT given
            for i, aper in enumerate(self.aper_num_lst):
                if aper == aperture:
                    return self.aper_wav_lst[i]
            print('Error: Aperture {} does not exist'.format(aperture))
            raise ValueError
        elif order is not None and aperture is None:
            # order is given and aperture is NOT given
            for i, o in enumerate(self.aper_ord_lst):
                if o == order:
                    return self.aper_wav_lst[i]
            print('Error: Order {} does not exist'.format(order))
            raise ValueError
        else:
            raise ValueError

    def get_brightness(self, aperture=None, order=None):
        """Get brightness of a specific aperture or order.

        Args:
            aperture (int): Aperture number.
            order (int): Order number.

        Returns:
            *float*: brigtness of the specific aperture or order.
        """
        if aperture is not None and order is None:
            # aperture is given and order is NOT given
            for i, aper in enumerate(self.aper_num_lst):
                if aper == aperture:
                    return self.aper_brt_lst[i]
            print('Error: Aperture {} does not exist'.format(aperture))
            raise ValueError
        elif order is not None and aperture is None:
            # order is given and aperture is NOT given
            for i, o in enumerate(self.aper_ord_lst):
                if o == order:
                    return self.aper_brt_lst[i]
            print('Error: Order {} does not exist'.format(order))
            raise ValueError
        else:
            raise ValueError

    def get_position(self, aperture=None, order=None):
        """Get position of a specific aperture or order.

        Args:
            aperture (int): Aperture number.
            order (int): Order number.

        Returns:
            *float*: position of the specific aperture or order.
        """
        if aperture is not None and order is None:
            # aperture is given and order is NOT given
            for i, aper in enumerate(self.aper_num_lst):
                if aper == aperture:
                    return self.aper_pos_lst[i]
            print('Error: Aperture {} does not exist'.format(aperture))
            raise ValueError
        elif order is not None and aperture is None:
            # order is given and aperture is NOT given
            for i, o in enumerate(self.aper_ord_lst):
                if o == order:
                    return self.aper_pos_lst[i]
            print('Error: Order {} does not exist'.format(order))
            raise ValueError
        else:
            raise ValueError

    def savefits(self, filename):
        """Save this object to FITS file.

        Args:
            filename (str):

        """
        prefix = 'HIERARCH GAMSE BACKGROUNDLIGHT '
        self.header.append((prefix + 'FILEID',   self.info['fileid']))
        self.header.append((prefix + 'FIBER',    self.info['fiber']))
        self.header.append((prefix + 'OBJECT',   self.info['object']))
        #self.header.append((prefix + 'OBJTYPE',  self.info['objtype']))
        self.header.append((prefix + 'EXPTIME',  self.info['exptime']))
        self.header.append((prefix + 'DATE-OBS', self.info['date-obs']))

        for aper, order, pos, brt, wav in zip(self.aper_num_lst,
                                              self.aper_ord_lst,
                                              self.aper_pos_lst,
                                              self.aper_brt_lst,
                                              self.aper_wav_lst,
                                             ):

            prefix2 = prefix + 'APERTURE {} '.format(aper)

            self.header.append((prefix2 + 'ORDER',      order))
            self.header.append((prefix2 + 'POSITION',   pos))
            self.header.append((prefix2 + 'BRIGHTNESS', brt))
            self.header.append((prefix2 + 'WAVELENGTH', wav))

        fits.writeto(filename, self.data, self.header, overwrite=True)

    @staticmethod
    def read(filename):
        data, head = fits.getdata(filename, header=True)
        prefix = 'GAMSE BACKGROUNDLIGHT '
        info = {'fileid':   head[prefix + 'FILEID'],
                'fiber':    head[prefix + 'FIBER'],
                'object':   head[prefix + 'OBJECT'],
                #'objtype':  head[prefix + 'OBJTYPE'],
                'exptime':  head[prefix + 'EXPTIME'],
                'date-obs': head[prefix + 'DATE-OBS'],
                }

        aper_num_lst = []
        aper_ord_lst = []
        aper_pos_lst = []
        aper_brt_lst = []
        aper_wav_lst = []
        for key, value in head.items():
            m = re.match('^GAMSE BACKGROUNDLIGHT APERTURE (\d+) ORDER', key)
            if m:
                aper = int(m.group(1))
                aper_num_lst.append(aper)
                aper_ord_lst.append(value)
                continue
            m = re.match('^GAMSE BACKGROUNDLIGHT APERTURE (\d+) POSITION', key)
            if m:
                aper_pos_lst.append(value)
                continue
            m = re.match('^GAMSE BACKGROUNDLIGHT APERTURE (\d+) BRIGHTNESS', key)
            if m:
                aper_brt_lst.append(value)
                continue
            m = re.match('^GAMSE BACKGROUNDLIGHT APERTURE (\d+) WAVELENGTH', key)
            if m:
                aper_wav_lst.append(value)
                continue

        bkg_obj = BackgroundLight(
                    info         = info,
                    header       = head,
                    data         = data,
                    aper_num_lst = aper_num_lst,
                    aper_ord_lst = aper_ord_lst,
                    aper_pos_lst = aper_pos_lst,
                    aper_brt_lst = aper_brt_lst,
                    aper_wav_lst = aper_wav_lst,
                )
        return bkg_obj

    def find_xdisp_shift(self, bkg_obj):
        """Find the relative shift between this and the input background light
        object.

        Args:
            bkg_obj ():

        Returns:
            *float*: Relative shift in pixel along the cross-dispersion
                    direction.
        """
        
        common_ord_lst = [order for order in self.aper_ord_lst
                                if order in bkg_obj.aper_ord_lst]
        pixel_shift_lst = [self.get_position(order=o)
                            - bkg_obj.get_position(order=o)
                                for o in common_ord_lst]
        pixel_shift_lst = np.array(pixel_shift_lst)
        return np.median(pixel_shift_lst)

    def find_brightness_scale(self, bkg_obj):
        """Find the scale factor of the brightness between this and the input
        background light object.

        Args:
            bkg_obj ():

        Returns:
            *float*: Scale factor of brightness.
        """
        common_ord_lst = [order for order in self.aper_ord_lst
                                if order in bkg_obj.aper_ord_lst]

        brt_lst1 = [self.get_brightness(order=o) for o in common_ord_lst]
        brt_lst2 = [bkg_obj.get_brightness(order=o) for o in common_ord_lst]

        fitfunc = lambda s: brt_lst2*s
        errfunc = lambda s: brt_lst1 - fitfunc(s)
        s0 = np.median(brt_lst1)/np.median(brt_lst2)
        fitres = opt.least_squares(errfunc, s0)
        s = fitres.x[0]
        return s

class BackgroundFigureCommon(Figure):
    """Figure to plot the background correction.
    """
    def __init__(self, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.canvas = FigureCanvasAgg(self)

def find_best_background(background_lst, background, fiber, objname, time,
        objtype):

    if objname == 'comb':
        candidate_lst = []
        shift_lst = []
        scale_lst = []
        for bkg_obj in background_lst:
            if bkg_obj.info['object'] == 'comb' \
                and  bkg_obj.info['fiber'] == fiber:
                shift = background.find_xdisp_shift(bkg_obj)
                scale = background.find_brightness_scale(bkg_obj)
                shift_lst.append(shift)
                scale_lst.append(scale)
                candidate_lst.append(bkg_obj)

        if len(candidate_lst)>0:
            index= np.array(scale_lst).argmin()
            return candidate_lst[index]

        for bkg_obj in background_lst:
            if bkg_obj.info['object'] == 'comb':
                shift = background.find_xdisp_shift(bkg_obj)
                scale = background.find_brightness_scale(bkg_obj)
                shift_lst.append(shift)
                scale_lst.append(scale)
                candidate_lst.append(bkg_obj)

        if len(candidate_lst)>0:
            index= np.array(scale_lst).argmin()
            return candidate_lst[index]

    elif objtype == 'star':
        candidate_lst = []
        scale_lst = []

        # check the same star
        for bkg_obj in background_lst:
            if bkg_obj.info['object'] != objname \
                or bkg_obj.info['fiber'] != fiber:
                continue
            scale = background.find_brightness_scale(bkg_obj)
            scale_lst.append(scale)
            candidate_lst.append(bkg_obj)

        if len(candidate_lst)>0:
            index = np.array(scale_lst).argmin()
            return candidate_lst[index]

        for bkg_obj in background_lst:
            if bkg_obj.info['fiber'] != fiber:
                continue
            scale = background.find_brightness_scale(bkg_obj)
            scale_lst.append(scale)
            candidate_lst.append(bkg_obj)

        if len(candidate_lst)>0:
            index = np.array(scale_lst).argmin()
            return candidate_lst[index]

def select_background_from_database(path, **args):
    # find the index file
    shape     = args.pop('shape')
    fiber     = args.pop('fiber')
    direction = args.pop('direction')
    objtype   = args.pop('objtype',  None)
    obj       = args.pop('obj',      None)
    spectype  = args.pop('spectype', None)

    logger.info('objtype={}, obj={}, spectype={}'.format(objtype, obj, spectype))

    filename = os.path.join(path, 'index.dat')
    table = Table.read(filename, format='ascii.fixed_width_two_line')
    # first round

    mask = table['objtype']==objtype
    table = table[mask]
    logger.info('mask={}'.format(mask))

    if obj == 'comb':
        mask = table['object']=='comb'
        table = table[mask]
        m1 = table['shape']==str(shape)[1:-1]
        m2 = table['fiber']==fiber
        m3 = table['direction']==direction
        score = np.int32(m1) + np.int32(m2) + np.int32(m3)
        logger.debug('score={}'.format(score))
        mask = score == score.max()
        logger.debug('mask={}'.format(mask))
        table = table[mask]
        row = table[0]
        logger.debug('selected {} (obj={}, fiber={})'.format(
                      row['fileid'], row['object'], row['fiber']))
    elif objtype == 'star':
        mask = []
        for row in table:
            if row['object'].lower()==obj:
                mask.append(True)
            else:
                mask.append(False)
        if sum(mask)>0:
            table = table[mask]
            row = table[0]
        else:
            row = table[0]

        logger.debug('selected {} (obj={}, fiber={})'.format(
                      row['fileid'], row['object'], row['fiber']))
    else:
        pass

    selected_fileid = row['fileid']
    filename = os.path.join(path, 'bkg.{}.fits'.format(selected_fileid))
    return BackgroundLight.read(filename)
