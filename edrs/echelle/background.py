import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage.filters import median_filter
import scipy.interpolate as intp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm     as cmap
import matplotlib.ticker as tck

from ..utils.regression import polyfit2d, polyval2d
from .imageproc         import table_to_array, array_to_table

def find_background(data, mask, channels, apertureset_lst,
        method='poly', scale='linear', scan_step=200,
        xorder=2, yorder=2, maxiter=5, upper_clip=3, lower_clip=3,
        extend=True, display=True, fig_file=None, reg_file=None):

    '''Subtract the background for an input FITS image.

    Args:
        data (:class:`numpy.ndarray`): Input data image.
        mask (:class:`numpy.ndarray`): Mask of input data image.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of :class:`~edrs.echelle.trace.ApertureSet`
            at different channels.
        method (string): Method of finding background light.
        scale (string): Scale of the 2-D polynomial fitting. If 'log', fit the
            polynomial in the logrithm scale.
        scan_step (integer): Steps of scan in pixels.
        xorder (integer): Order of 2D polynomial along the main dispersion
            direction (only applicable when **method** = "poly").
        yorder (integer): Order of 2D polynomial along the cross-dispersion
            direction (only applicable when **method** = "poly").
        maxiter (integer): Maximum number of iteration of 2D polynomial fitting
            (only applicable when **method** = "poly").
        upper_clip (float): Upper sigma clipping threshold (only applicable when
            **method** = "poly").
        lower_clip (float): Lower sigma clipping threshold (only applicable when
            **method** = "poly").
        extend (bool): Extend the grid to the whole CCD image if *True*.
        display (bool): Display figures on the screen if *True*.
        fig_file (string): Name of the output figure. No image file generated if
            *None*.
        reg_file (string): Name of the output DS9 region file. No file generated
            if *None*.

    Returns:
        :class:`numpy.ndarray`: Array of background light.
    '''
    
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
                        if len(inter_aper)==0 or \
                            abs(mid_newy - inter_aper[-1])>scan_step*0.7:
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
            coeff = np.polyfit(np.arange(inter_aper.size), inter_aper, deg=3)
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
    '''
    Find the background light by fitting a 2D polynomial.

    Args:
        shape (tuple): Shape of image.
        xnodes (:class:`numpy.ndarray`): List of X coordinates of the nodes.
        ynodes (:class:`numpy.ndarray`): List of Y coordinates of the nodes.
        znodes (:class:`numpy.ndarray`): List of pixel values of the nodes.
        xorder (integer): Order of 2D polynomial along the main dispersion
            direction.
        yorder (integer): Order of 2D polynomial along the cross-dispersion
            direction.
        maxiter (integer): Maximum number of iteration of 2D polynomial fitting.
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
    '''

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
    '''
    Find the background light by interpolating 2D cubic splines.

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
    '''
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
