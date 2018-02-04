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
from ..ccdproc import save_fits, table_to_array, array_to_table

def correct_background(infilename, mskfilename, outfilename, scafilename,
        channels, apertureset_lst, scale='linear', block_mask=4, scan_step=200,
        xorder=2, yorder=2, maxiter=5, upper_clip=3., lower_clip=3.,
        extend=True, display=True, fig_file=None, reg_file=None):

    '''Subtract the background for an input FITS image.

    Args:
        infilename (string): Name of the input file.
        outfilename (string): Name of the output file.
        scafilename (string): Name of the scatter light file.
        channels (list): List of channels as strings.
        apertureset_lst (dict): Dict of ApertureSet at different channels.
        scale (string): Scale of the image. Either 'linear' or 'log'.
        block_mask (integer): Block value in the mask file.
        scan_step (integer): Steps of scan in pixels.
        xorder (integer): Order of 2D polynomial along *x*-axis
            (main dispersion direction)
        yorder (integer): Order of 2D polynomial along *y*-axis
            (cross-dispersion direction)
        maxiter (integer): Maximum number of iteration of 2D polynomial fitting.
        upper_clip (float): Upper sigma clipping threshold.
        lower_clip (float): Lower sigma clipping threshold.
        extend (bool): Extend the grid to the whole image if *True*.
        display (bool): Display figures on the screen if *True*.
        fig_file (string): Name of the output figure.
        reg_file (string): Name of the output DS9 region file.

    Returns:
        No returns.
    '''
    
    plot = (display or fig_file is not None)

    data, head = fits.getdata(infilename, header=True)

    h, w = data.shape

    # read data mask
    mask_table = fits.getdata(mskfilename)
    if mask_table.size==0:
        mask = np.zeros_like(data, dtype=np.int16)
    else:
        mask = table_to_array(mask_table, data.shape)
    data_mask = (np.int16(mask) & block_mask > 0)

    meddata = median_filter(data, size=(3,3), mode='reflect')

    xnodes, ynodes, znodes = [], [], []

    # find the minimum and maximum aperture number
    min_aper = min([min(apertureset_lst[ch].dict.keys()) for ch in channels])
    max_aper = max([max(apertureset_lst[ch].dict.keys()) for ch in channels])

    for x in np.arange(1, w, scan_step):
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
                        mid_newy = int((prev_newy + this_newy)/2.)
                        i1 = max(0,  int(prev_newy))
                        i2 = min(h-1,int(this_newy))
                        if len(inter_aper)==0 or \
                            abs(mid_newy - inter_aper[-1])>scan_step*0.6:
                            mid_newy = i1 + xsection[i1:i2].argmin()
                            inter_aper.append(mid_newy)
                    prev_newy = this_newy

        inter_aper = np.array(inter_aper)

        # if extend = True, expand the grid with polynomial fitting to
        # cover the whole CCD area
        if extend:
            coeff = np.polyfit(np.arange(inter_aper.size), inter_aper, deg=3)
            # find the points after the end of inter_aper
            ii = inter_aper.size-1
            new_y = inter_aper[-1]
            while(new_y<h-1):
                ii += 1
                new_y = int(np.polyval(coeff,ii))
                inter_aper = np.append(inter_aper,new_y)
            # find the points before the beginning of order_mid
            ii = 0
            new_y = inter_aper[0]
            while(new_y>0):
                ii -= 1
                new_y = int(np.polyval(coeff,ii))
                inter_aper = np.insert(inter_aper,0,new_y)

        # remove those points with y<0 or y>h-1
        m1 = inter_aper > 0
        m2 = inter_aper < h-1
        inter_aper = inter_aper[np.nonzero(m1*m2)[0]]

        # remove backward points
        tmp = np.insert(inter_aper,0,0.)
        mask = np.diff(tmp)>0
        inter_aper = inter_aper[np.nonzero(mask)[0]]

        for y in inter_aper:
            # avoid including masked pixels in fitting
            if not data_mask[y,x]:
                xnodes.append(x)
                ynodes.append(y)
                znodes.append(meddata[y,x])

    # convert to numpy array
    xnodes = np.array(xnodes)
    ynodes = np.array(ynodes)
    znodes = np.array(znodes)
    
    # if scale='log', filter the negative values
    if scale=='log':
        mask = znodes > 0
        xnodes = xnodes[mask]
        ynodes = ynodes[mask]
        znodes = znodes[mask]


    if plot:
        # initialize figures
        fig = plt.figure(figsize=(12,12), dpi=150)
        ax11 = fig.add_axes([0.07, 0.54, 0.39,  0.39])
        ax12 = fig.add_axes([0.52, 0.54, 0.39,  0.39])
        ax13 = fig.add_axes([0.94, 0.54, 0.015, 0.39])
        ax21 = fig.add_axes([0.07, 0.07, 0.39,  0.39], projection='3d')
        ax22 = fig.add_axes([0.52, 0.07, 0.39,  0.39], projection='3d')

        fig.suptitle('Background of %s'%os.path.basename(infilename))
        ax11.imshow(data, cmap='gray')

        # plot nodes
        ax11.scatter(xnodes, ynodes, c='r', s=8, linewidth=0, alpha=0.8)
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

        if display: plt.show(block=False)

    # normalize to 0 ~ 1 for x and y nodes
    xfit = np.float64(xnodes)/w
    yfit = np.float64(ynodes)/h
    if scale=='log':
        # calculate logarithmic Z
        zfit = np.log(znodes)
    elif scale=='linear':
        zfit = znodes

    # fit the 2-d polynomial
    fitmask = np.ones_like(zfit, dtype=np.bool)
    for niter in range(maxiter):
        coeff = polyfit2d(xfit[fitmask], yfit[fitmask], zfit[fitmask],
                          xorder=xorder, yorder=yorder)
        fitvalues = polyval2d(xfit, yfit, coeff)
        residual = zfit - fitvalues
        mean  = residual[fitmask].mean(dtype=np.float64)
        sigma = residual[fitmask].std(dtype=np.float64)
        m1 = residual < mean + upper_clip*sigma
        m2 = residual > mean - lower_clip*sigma
        new_fitmask = m1*m2
        if new_fitmask.sum() == fitmask.sum():
            break
        fitmask = new_fitmask

    # write nodes to running log
    message = ['Background Nodes for "%s":'%infilename,
                ' x,    y,    value,  mask']
    for x,y,z,m in zip(xnodes, ynodes, znodes, fitmask):
        message.append('%4d %4d %+10.8e %2d'%(x,y,z,m))
    logger.info((os.linesep+' '*4).join(message))


    if scale=='log':
        log_residual = residual
        residual = znodes - np.exp(fitvalues)

    if plot:
        # prepare for plotting the fitted surface with a loose grid
        xx, yy = np.meshgrid(np.linspace(0,w-1,32), np.linspace(0,h-1,32))
        zz = polyval2d(xx/w, yy/h, coeff)
        if scale=='log':
            log_zz = zz
            zz = np.exp(log_zz)

        # plot 2d fitting in a 3-D axis in fig2
        # plot the linear fitting
        ax21.set_title('Background fitting (%s Z)'%scale, fontsize=10)
        ax22.set_title('residuals (%s Z)'%scale, fontsize=10)
        ax21.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
                          linewidth=0, antialiased=True, alpha=0.5)
        ax21.scatter(xnodes[fitmask], ynodes[fitmask], znodes[fitmask],   linewidth=0)
        ax22.scatter(xnodes[fitmask], ynodes[fitmask], residual[fitmask], linewidth=0)

        # plot the logrithm fitting in another fig
        #if scale=='log':
        #    ax23.plot_surface(xx, yy, log_zz, rstride=1, cstride=1, cmap='jet',
        #                        linewidth=0, antialiased=True, alpha=0.5)
        #    ax23.scatter(xnodes[fitmask], ynodes[fitmask], zfit[fitmask],         linewidth=0)
        #    ax24.scatter(xnodes[fitmask], ynodes[fitmask], log_residual[fitmask], linewidth=0)

        for ax in [ax21, ax22]:
            #ax.set_xlim(0, w-1)
            #ax.set_ylim(0, h-1)
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
            #ax.set_xlabel('X (pixel)', fontsize=10)
            #ax.set_ylabel('Y (pixel)', fontsize=10)

        if display: fig.canvas.draw()

    # calculate the background
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = np.float64(xx)/w
    yy = np.float64(yy)/h
    if scale=='log':
        background_data = np.exp(polyval2d(xx, yy, coeff))
    elif scale=='linear':
        background_data = polyval2d(xx, yy, coeff)

    if plot:
        # plot the background light
        cnorm = colors.Normalize(vmin = background_data.min(),
                                 vmax = background_data.max())
        scalarmap = cmap.ScalarMappable(norm=cnorm, cmap=cmap.jet)
        image = ax12.imshow(background_data, cmap=scalarmap.get_cmap())
        ax12.scatter(xnodes, ynodes, c=znodes, s=8, lw=0.5, cmap=scalarmap.get_cmap())
        # set colorbar
        plt.colorbar(image, cax=ax13)
        # set font size of colorbar
        for tick in ax13.get_yaxis().get_major_ticks():
            tick.label2.set_fontsize(9)

        if display: fig.canvas.draw()

    # correct background
    corrected_data = data - background_data

    save_fits(outfilename, corrected_data,  head) 
    save_fits(scafilename, background_data, head) 


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


    if fig_file is not None:
        fig.savefig(fig_file)
        #for ii in xrange(0,360,15):
        #    for ax in fig2.get_axes():
        #        ax.view_init(elev=20.,azim=ii)
        #    fig2.savefig('bkg-overview-3d-%s-%03d.png'%(
        #              os.path.basename(infilename),ii))

    plt.close(fig)

