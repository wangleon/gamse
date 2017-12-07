import os
import numpy as np
import astropy.io.fits as fits

from ..ccdproc import save_fits

def correct_background(infilename, mskfilename, outfilename, scafilename,
        order_lst, scale='linear', block_mask=4, scan_step=200,
        xorder=2, yorder=2, maxiter=5, upper_clipping=3., lower_clipping=3.,
        expand_grid = True,
        fig1 = None, fig2 = None, report_img_path = None):
    '''Subtract the background for an input FITS image.

    Args:
        infilename (str): Name of the input file.
        outfilename (str): Name of the output file.
        scafilename (str): Name of the scatter light file.
        order_lst (list): Positions of each order.
        scan_step (int): Steps of scan in pixels.
        fig1 (:class:`matplotlib.figure`): Figure to display.
        fig2 (:class:`matplotlib.figure`): Figure to display.
        report_img_path (str): Path to the report directory.

    Returns:
        No returns.
    '''

    from scipy.ndimage.filters import median_filter
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm     as cmap
    import matplotlib.ticker as tck

    from ..utils.regression import polyfit2d, polyval2d

    data, head = fits.getdata(infilename,header=True)
    # get data mask
    mdata = fits.getdata(mskfilename)
    data_mask = (np.int16(mdata) & block_mask > 0)

    meddata = median_filter(data, size=(3,3), mode='reflect')

    ax11 = fig1.get_axes()[0]
    ax12 = fig1.get_axes()[1]
    ax13 = fig1.get_axes()[2]
    for ax in fig1.get_axes():
        ax.cla()
    title1 = fig1.suptitle('')
    title1.set_text('Background of %s'%os.path.basename(infilename))
    ax11.imshow(data,cmap='gray')

    xnodes, ynodes, znodes = [], [], []
    h, w = data.shape
    for x in np.arange(1, w, scan_step):
        order_cen = []
        for location in order_lst:
            xdata = location['x']
            ydata = location['y']
            if x in xdata:
                i = np.where(xdata == x)[0][0]
                order_cen.append(ydata[i])

        order_cen = np.array(order_cen)
        #ax.plot(np.zeros_like(order_cen)+x, order_cen,'ro')
        order_mid = ((order_cen + np.roll(order_cen,1))/2.)[1:]

        order_mid = np.int16(order_mid[::4])

        # if expand_grid = True, expand the grid with polynomial fitting to
        # cover the whole CCD area
        if expand_grid:
            coeff = np.polyfit(np.arange(order_mid.size), order_mid, deg=3)
            # find the points after the end of order_mid
            ii = order_mid.size-1
            new_y = order_mid[-1]
            while(new_y<h-1):
                ii += 1
                new_y = int(np.polyval(coeff,ii))
                order_mid = np.append(order_mid,new_y)
            # find the points before the beginning of order_mid
            ii = 0
            new_y = order_mid[0]
            while(new_y>0):
                ii -= 1
                new_y = int(np.polyval(coeff,ii))
                order_mid = np.insert(order_mid,0,new_y)

        # remove those points with y<0 or y>h-1
        m1 = order_mid > 0
        m2 = order_mid < h-1
        order_mid = order_mid[np.nonzero(m1*m2)[0]]

        # remove backward points
        tmp = np.insert(order_mid,0,0.)
        mask = np.diff(tmp)>0
        order_mid = order_mid[np.nonzero(mask)[0]]


        for y in order_mid:
            # avoid including masked pixels in fitting
            if not data_mask[y,x]:
                xnodes.append(x)
                ynodes.append(y)
                znodes.append(meddata[y,x])

    # convert to numpy array
    xnodes = np.array(xnodes)
    ynodes = np.array(ynodes)
    znodes = np.array(znodes)
    
    # if scale='log', then filter the negative values
    if scale=='log':
        mask = znodes > 0
        xnodes = xnodes[mask]
        ynodes = ynodes[mask]
        znodes = znodes[mask]

    nodefile = open(outfilename[0:-5]+'_nodes.txt','w')
    for x,y,z in zip(xnodes, ynodes, znodes):
        nodefile.write('%4d %4d %+10.8e%s'%(x,y,z,os.linesep))
    nodefile.close()


    # plot nodes
    ax11.scatter(xnodes, ynodes, c='r', s=10, linewidth=0)

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
        mean  = residual[fitmask].mean(dtype='float64')
        sigma = residual[fitmask].std(dtype='float64')
        m1 = residual < mean + upper_clipping*sigma
        m2 = residual > mean - lower_clipping*sigma
        new_fitmask = m1*m2
        if new_fitmask.sum() == fitmask.sum():
            break
        fitmask = new_fitmask

    if scale=='log':
        log_residual = residual
        residual = znodes - np.exp(fitvalues)

    # prepare for plotting the fitted surface with a loose grid
    xx, yy = np.meshgrid(np.linspace(0,w-1,32), np.linspace(0,h-1,32))
    zz = polyval2d(xx/w, yy/h, coeff)
    if scale=='log':
        log_zz = zz
        zz = np.exp(log_zz)

    # plot 2d fitting in a 3-D axis in fig2
    ms = 3
    alpha=0.5
    fontsize=9
    for ax in fig2.get_axes():
        ax.cla()
    title2 = fig2.suptitle('')
    title2.set_text('3D Background of %s'%os.path.basename(infilename))
    # plot the linear fitting
    ax21 = fig2.get_axes()[0]
    ax22 = fig2.get_axes()[1]
    ax21.set_title('Background fitting (linear Z)',fontsize=fontsize)
    ax22.set_title('residuals (linear Z)',         fontsize=fontsize)
    ax21.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
                      linewidth=0, antialiased=True, alpha=alpha)
    ax21.scatter(xnodes[fitmask], ynodes[fitmask], znodes[fitmask],   linewidth=0)
    ax22.scatter(xnodes[fitmask], ynodes[fitmask], residual[fitmask], linewidth=0)
    if scale=='log':
        # plot the logrithm fitting
        ax23 = fig2.get_axes()[2]
        ax24 = fig2.get_axes()[3]
        ax23.set_title('Background fitting (log Z)',   fontsize=fontsize)
        ax24.set_title('residuals (log Z)',            fontsize=fontsize)
        ax23.plot_surface(xx, yy, log_zz, rstride=1, cstride=1, cmap='jet',
                          linewidth=0, antialiased=True, alpha=alpha)
        ax23.scatter(xnodes[fitmask], ynodes[fitmask], zfit[fitmask],         linewidth=0)
        ax24.scatter(xnodes[fitmask], ynodes[fitmask], log_residual[fitmask], linewidth=0)

    for ax in fig2.get_axes():
        ax.set_xlim(0,w-1)
        ax.set_ylim(0,h-1)
        ax.xaxis.set_major_locator(tck.MultipleLocator(500))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.yaxis.set_major_locator(tck.MultipleLocator(500))
        ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
    fig2.canvas.draw()

    # calculate the background
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = np.float64(xx)/w
    yy = np.float64(yy)/h
    if scale=='log':
        background_data = np.exp(polyval2d(xx, yy, coeff))
    elif scale=='linear':
        background_data = polyval2d(xx, yy, coeff)

    # plot the background light in fig1
    cnorm = colors.Normalize(vmin = background_data.min(),
                             vmax = background_data.max())
    scalarmap = cmap.ScalarMappable(norm=cnorm, cmap=cmap.jet)
    image = ax12.imshow(background_data, cmap=scalarmap.get_cmap())
    ax12.scatter(xnodes, ynodes, c=znodes, s=10, cmap=scalarmap.get_cmap())
    plt.colorbar(image,cax=ax13)
    for ax in fig1.get_axes()[0:2]:
        ax.set_xlim(0,w-1)
        ax.set_ylim(h-1,0)
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
    fig1.canvas.draw()

    # correct background
    corrected_data = data - background_data

    save_fits(outfilename, corrected_data,  head) 
    save_fits(scafilename, background_data, head) 

    if report_img_path != None:
        fig1.savefig(os.path.join(report_img_path,
                'bkg-overview-%s.png'%os.path.basename(infilename)
            ))
        fig2.savefig(os.path.join(report_img_path,
                'bkg-overview-3d-%s.png'%os.path.basename(infilename)
            ))

        #for ii in xrange(0,360,15):
        #    for ax in fig2.get_axes():
        #        ax.view_init(elev=20.,azim=ii)
        #    fig2.savefig('bkg-overview-3d-%s-%03d.png'%(
        #              os.path.basename(infilename),ii))
