import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ...utils.onedarray import iterative_savgol_filter
from ...echelle.trace import TraceFigureCommon, AlignFigureCommon
from ...echelle.flat import ProfileNormalizerCommon
from ...echelle.background import BackgroundFigureCommon
from ...echelle.wlcalib import get_calib_from_header, get_wavelength
from ...utils.download import get_file
from ...utils.onedarray import iterative_savgol_filter, get_edge_bin

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

def get_mask(data, header):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        header (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.

    The shape of output mask is determined by the keywords in the input FITS
    header. The numbers of columns and rows are given by::
     
        N (columns) = head['NAXIS1'] - head['COVER']

        N (rows)    = head['NAXIS2'] - head['ROVER']

    where *head* is the input FITS header. 

    """
    saturation_adu = 65535
    # find the saturation mask
    mask_sat = (data[:, 0:4096] >= saturation_adu)

    mask = np.int16(mask_sat)*4

    return mask

def select_calib_from_database(index_file, dateobs):
    """Select wavelength calibration file in database.

    Args:
        index_file (str): Index file of saved calibration files.
        dateobs (str): .

    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
              spectra.
            * **calib** (dict): Previous calibration results.
    """
    
    calibtable = Table.read(index_file, format='ascii.fixed_width_two_line')

    input_date = dateutil.parser.parse(dateobs)

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable[mask]['obsdate']]
    irow = np.abs(timediff).argmin()
    row = calibtable[mask][irow]
    fileid = row['fileid']  # selected fileid
    md5    = row['md5']

    message = 'Select {} from database index as ThAr reference'.format(fileid)
    logger.info(message)

    filepath = os.path.join('instruments/yhrs',
                'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    calib = get_calib_from_header(head)

    return spec, calib

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

class SpatialProfileFigure(Figure):
    """Figure to plot the cross-dispersion profiles.

    """
    def __init__(self,
            nrow = 3,
            ncol = 3,
            figsize = (12,8),
            dpi = 200,
            ):

        # create figure
        Figure.__init__(self, figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self)

        # add axes
        _w = 0.27
        _h = 0.26
        for irow in range(nrow):
            for icol in range(ncol):
                _x = 0.08 + icol*0.31
                _y = 0.06 + (nrow-1-irow)*0.30

                ax = self.add_axes([_x, _y, _w, _h])

    def close(self):
        plt.close(self)

class ProfileNormalizer(ProfileNormalizerCommon):
    def __init__(self, xdata, ydata, mask):
        self.xdata = xdata
        self.ydata = ydata
        self.mask  = mask

        sat_mask = (mask&4 > 0)
        bad_mask = (mask&2 > 0)

        # iterative fitting using fitfunc
        A0 = ydata.max() - ydata.min()
        c0 = (xdata[0] + xdata[-1])/2
        b0 = ydata.min()
        p0 = [A0, c0, 5.0, 4.0, b0]
        lower_bounds = [-np.inf, xdata[0],  0.5,    0.5,    -np.inf]
        upper_bounds = [np.inf,  xdata[-1], np.inf, 100., ydata.max()]
        _m = (~sat_mask)*(~bad_mask)

        for i in range(10):
            opt_result = opt.least_squares(self.errfunc, p0,
                        args=(xdata[_m], ydata[_m]),
                        bounds=(lower_bounds, upper_bounds))
            p1 = opt_result['x']
            residuals = self.errfunc(p1, xdata, ydata)
            std = residuals[_m].std(ddof=1)
            _new_m = (np.abs(residuals) < 3*std)*_m
            if _m.sum() == _new_m.sum():
                break
            _m = _new_m
            p0 = p1
    
        A, c, alpha, beta, bkg = p1
        self.x = xdata - c
        self.y = (ydata - bkg)/A
        self.m = _m
        
        self.param = p1
        self.std = std

    def is_succ(self):
        A, center, alpha, beta, bkg = self.param
        std = self.std

        if A>0 and A/std>10 and alpha<10 and beta<10 and \
            (bkg>0 or (bkg<0 and abs(bkg)<A/10)):
            return True
        else:
            return False

    def fitfunc(self, param, x):
        """Use Generalized Gaussian.
        """
        A, center, alpha, beta, bkg = param
        return A*np.exp(-np.power(np.abs(x-center)/alpha, beta)) + bkg


def norm_profile(xdata, ydata, mask):
    # define the fitting and error functions
    def gaussian_gen_bkg(A, center, alpha, beta, bkg, x):
        return A*np.exp(-np.power(np.abs(x-center)/alpha, beta)) + bkg
    def fitfunc(p, x):
        return gaussian_gen_bkg(p[0], p[1], p[2], p[3], p[4], x)
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # iterative fitting using gaussian + bkg function
    A0 = ydata.max()-ydata.min()
    c0 = (xdata[0]+xdata[-1])/2
    b0 = ydata.min()
    p0 = [A0, c0, 5.0, 4.0, b0]
    lower_bounds = [-np.inf, xdata[0],  0.5,    0.5,    -np.inf]
    upper_bounds = [np.inf,  xdata[-1], np.inf, np.inf, ydata.max()]
    _m = (~sat_mask)*(~bad_mask)

    for i in range(10):
        opt_result = opt.least_squares(errfunc, p0,
                    args=(xdata[_m], ydata[_m], fitfunc),
                    bounds=(lower_bounds, upper_bounds))
        p1 = opt_result['x']
        residuals = errfunc(p1, xdata, ydata, fitfunc)
        std = residuals[_m].std(ddof=1)
        _new_m = (np.abs(residuals) < 3*std)*_m
        if _m.sum() == _new_m.sum():
            break
        _m = _new_m
        p0 = p1

    A, c, alpha, beta, bkg = p1
    newx = xdata - c
    newy = ydata - bkg

    param = (A, c, alpha, beta, bkg, std)

    if A < 1e-3:
        return None
    return newx, newy/A, param
