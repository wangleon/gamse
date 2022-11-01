import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ...utils.onedarray import iterative_savgol_filter
from ...echelle.trace import TraceFigureCommon, AlignFigureCommon

def print_wrapper(string, item):
    imgtype = item['imgtype']
    objname = item['object'].strip().lower()

    if imgtype=='cal' and objname=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'
    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'
    elif imgtype=='cal' and objname=='thar':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

def correct_overscan(data, head):
    """Correct overscan.

    Args:
        data ():
        head ():

    Returns:
        

    """
    scidata = data[:, 0:4096]
    ovrdata = data[:, 4096:4096+32]
    ovrmean = ovrdata.mean(axis=1)

    ovrsmooth, _, _, _ = iterative_savgol_filter(ovrmean, winlen=351,
                               order=3, upper_clip=3.0)

    corrdata = np.zeros((data.shape[0], 4096), dtype=np.float32)
    corrdata[:, 0:4096] = scidata - ovrsmooth.reshape(-1, 1)

    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(ovrmean, lw=0.5, alpha=0.8)
    ax.plot(ovrsmooth, lw=0.5, alpha=0.8)
    fig.savefig('{}.png'.format(head['DATE-OBS']))
    plt.close(fig)
    '''

    return corrdata

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
