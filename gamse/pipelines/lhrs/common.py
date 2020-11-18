import numpy as np

from ...echelle.trace import TraceFigureCommon

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
    ovrmean1 = ovrdata1[100:,:].mean()
    ovrmean2 = ovrdata2[100:,:].mean()

    scidata = np.zeros((data.shape[0], 4096), dtype=np.float32)
    scidata[:, 0:2048]    = scidata1 - ovrmean1
    scidata[:, 2048:4096] = scidata2 - ovrmean2

    '''
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(ovrdata1.mean(axis=1))
    ax.axhline(y=ovrmean1)
    ax.plot(ovrdata2.mean(axis=1))
    ax.axhline(y=ovrmean2)
    plt.show()
    '''
    return scidata

def get_mask():
    pass

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

