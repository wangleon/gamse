import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ...utils.onedarray import iterative_savgol_filter

from ...echelle.trace import TraceFigureCommon, AlignFigureCommon
from ...echelle.background import BackgroundFigureCommon

def print_wrapper(string, item):
    return string

def correct_overscan(data, head):
    """Correct overscan.

    Args:
        data ():
        head ():

    Returns:
        

    """
    scidata1 = data[:, 0:2048]
    scidata2 = data[:, 2048:4096]
    ovrdata1 = data[:, 4096:4096+32]
    ovrdata2 = data[:, 4096+32:4096+64]
    ovrmean1 = ovrdata1.mean(axis=1)
    ovrmean2 = ovrdata2.mean(axis=1)


    ovrsmooth1, _, _, _ = iterative_savgol_filter(ovrmean1, winlen=351,
                               order=3, upper_clip=3.0)
    ovrsmooth2, _, _, _ = iterative_savgol_filter(ovrmean2, winlen=351,
                               order=3, upper_clip=3.0)

    scidata = np.zeros((data.shape[0], 4096), dtype=np.float32)
    scidata[:, 0:2048]    = scidata1 - ovrsmooth1.reshape(-1, 1)
    scidata[:, 2048:4096] = scidata2 - ovrsmooth2.reshape(-1, 1)

    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(ovrmean1, lw=0.5, alpha=0.8)
    ax.plot(ovrmean2, lw=0.5, alpha=0.8)
    ax.plot(ovrsmooth1, lw=0.5, alpha=0.8)
    ax.plot(ovrsmooth2, lw=0.5, alpha=0.8)
    plt.show()
    exit()
    '''
    return scidata

def get_mask(data, head):
    ny, nx = data.shape
    if nx == 4160:
        nx = 4096 # last 64 columns are overscan region

    bad_mask = np.zeros((ny, nx), dtype=int)

    mask = np.zeros((ny, nx), dtype=np.int16)

    if (ny, nx) == (4096, 4096):
        bad_mask[1720, 2314:2315] = 1
    mask = bad_mask*2
    return mask

class BackgroundFigure(BackgroundFigureCommon):
    """Figure to plot the background correction.
    """
    def __init__(self, data=None, background=None, dpi=300, figsize=(12, 5.5),
           title=None, figname=None, zscale=('log', 'linear'), contour=True):
        BackgroundFigureCommon.__init__(self, figsize=figsize, dpi=dpi)
        width = 0.36
        height = width*figsize[0]/figsize[1]
        self.ax1  = self.add_axes([0.06, 0.1, width, height])
        self.ax2  = self.add_axes([0.55, 0.1, width, height])
        self.ax1c = self.add_axes([0.06+width+0.01, 0.1, 0.015, height])
        self.ax2c = self.add_axes([0.55+width+0.01, 0.1, 0.015, height])

        if data is not None and background is not None:
            self.plot_background(data, background,
                            zscale=zscale, contour=contour)
        if title is not None:
            self.suptitle(title)
        if figname is not None:
            self.savefig(figname)

    def plot_background(self, data, background, scale=(5, 99),
            zscale=('log', 'linear'), contour=True):
        """Plot the image data with background and the subtracted background
        light.

        Args:
            data (:class:`numpy.ndarray`): Image data to be background
                subtracted.
            background (:class:`numpy.ndarray`): Background light as a 2D array.
        """
        # find the minimum and maximum value of plotting

        if zscale[0] == 'linear':
            vmin = np.percentile(data, scale[0])
            vmax = np.percentile(data, scale[1])
            cax1 = self.ax1.imshow(data, cmap='gray', vmin=vmin, vmax=vmax,
                        origin='lower')
            # set colorbar
            cbar1 = self.colorbar(cax1, cax=self.ax1c)
        elif zscale[0] == 'log':
            m = data <= 0
            plotdata1 = np.zeros_like(data, dtype=np.float32)
            plotdata1[m] = 0.1
            plotdata1[~m] = np.log10(data[~m])
            vmin = np.percentile(plotdata1[~m], scale[0])
            vmax = np.percentile(plotdata1[~m], scale[1])
            cax1 = self.ax1.imshow(plotdata1, cmap='gray', vmin=vmin, vmax=vmax,
                        origin='lower')
            # set colorbar
            tick_lst = np.arange(int(np.ceil(vmin)), int(np.ceil(vmax)))
            ticklabel_lst = ['$10^{}$'.format(i) for i in tick_lst]
            cbar1 = self.colorbar(cax1, cax=self.ax1c, ticks=tick_lst)
            cbar1.ax.set_yticklabels(ticklabel_lst)
        else:
            print('Unknown zscale:', zscale)


        if zscale[1] == 'linear':
            vmin = background.min()
            vmax = background.max()
            cax2 = self.ax2.imshow(background, cmap='viridis',
                    vmin=vmin, vmax=vmax, origin='lower')
            # set colorbar
            cbar2 = self.colorbar(cax2, cax=self.ax2c)
        elif zscale[1] == 'log':
            m = background <= 0
            plotdata2 = np.zeros_like(background, dtype=np.float32)
            plotdata2[m] = 0.1
            plotdata2[~m] = np.log10(background[~m])
            vmin = max(0.1, background[~m].min())
            vmax = plotdata2[~m].max()
            cax2 = self.ax2.imshow(plotdata2, cmap='viridis',
                    vmin=vmin, vmax=vmax, origin='lower')
            # plot contour in background panel
            if contour:
                cs = self.ax2.contour(plotdata2, colors='r', linewidths=0.5)
                self.ax2.clabel(cs, inline=1, fontsize=7, use_clabeltext=True)
            # set colorbar
            tick_lst = np.arange(int(np.ceil(vmin)), int(np.ceil(vmax)))
            ticklabel_lst = ['$10^{}$'.format(i) for i in tick_lst]
            cbar2 = self.colorbar(cax2, cax=self.ax2c, ticks=tick_lst)
            cbar2.ax.set_yticklabels(ticklabel_lst)
        else:
            print('Unknown zscale:', zscale)

        # set labels and ticks
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X (pixel)')
            ax.set_ylabel('Y (pixel)')
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self, datashape, figsize=(12,6)):
        TraceFigureCommon.__init__(self, figsize=figsize, dpi=150)
        axh = 0.86
        axw = axh/figsize[0]*figsize[1]/datashape[0]*datashape[1]
        x1 = 0.06
        self.ax1 = self.add_axes([x1,0.07,axw,axh])

        hgap = 0.05
        x2 = x1 + axw + hgap
        self.ax2 = self.add_axes([x2, 0.50, 0.95-x2, 0.40])
        self.ax3 = self.add_axes([x2, 0.10, 0.95-x2, 0.40])
        self.ax4 = self.ax3.twinx()

class AlignFigure(AlignFigureCommon):
    """Figure to plot the order alignment.
    """
    def __init__(self):
        AlignFigureCommon.__init__(self, figsize=(12,6), dpi=150)
        self.ax1 = self.add_axes([0.08, 0.1, 0.4, 0.8])
        self.ax2 = self.add_axes([0.55, 0.1, 0.4, 0.8])

