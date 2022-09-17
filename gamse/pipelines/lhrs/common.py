import numpy as np
from ...utils.onedarray import iterative_savgol_filter

from ...echelle.trace import TraceFigureCommon, AlignFigureCommon

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

