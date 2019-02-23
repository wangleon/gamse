import math
import itertools
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def polyfit2d(x, y, z, xorder=3, yorder=3, linear=False):
    """Two-dimensional polynomial fit.

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        z (:class:`numpy.ndarray`): Input Z array.
        xorder (int): X order.
        yorder (int): Y order.
        linear (bool): Return linear solution if `True`.
    Returns:
        :class:`numpy.ndarray`: Coefficient array.

    Examples:

        .. code-block:: python
    
           import numpy as np
           numdata = 100
           x = np.random.random(numdata)
           y = np.random.random(numdata)
           z = 6*y**2+8*y-x-9*x*y+10*x*y**2+7+np.random.random(numdata)
           m = polyfit2d(x, y, z, xorder=1, yorder=3)
           # evaluate it on a grid
           nx, ny = 20, 20
           xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                                np.linspace(y.min(), y.max(), ny))
           zz = polyval2d(xx, yy, m)
    
           fig1 = plt.figure(figsize=(10,5))
           ax1 = fig1.add_subplot(121,projection='3d')
           ax2 = fig1.add_subplot(122,projection='3d')
           ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
               linewidth=0, antialiased=True, alpha=0.3)
           ax1.set_xlabel('X (pixel)')
           ax1.set_ylabel('Y (pixel)')
           ax1.scatter(x, y, z, linewidth=0)
           ax2.scatter(x, y, z-polyval2d(x,y,m),linewidth=0)
           plt.show()

        if `linear = True`, the fitting only consider linear solutions such as

        .. math::

            z = a(x-x_0)^2 + b(y-y_0)^2 + c
    
        the returned coefficients are organized as an *m* x *n* array, where *m*
        is the order along the y-axis, and *n* is the order along the x-axis::
    
            1   + x     + x^2     + ... + x^n     +
            y   + xy    + x^2*y   + ... + x^n*y   +
            y^2 + x*y^2 + x^2*y^2 + ... + x^n*y^2 +
            ... + ...   + ...     + ... + ...     +
            y^m + x*y^m + x^2*y^m + ... + x^n*y^m

    """
    ncols = (xorder + 1)*(yorder + 1)
    G = np.zeros((x.size, ncols))
    ji = itertools.product(range(yorder+1), range(xorder+1))
    for k, (j,i) in enumerate(ji):
        G[:,k] = x**i * y**j
        if linear & (i != 0) & (j != 0):
            G[:,k] = 0
    coeff, residuals, _, _ = np.linalg.lstsq(G, z)
    coeff = coeff.reshape(yorder+1, xorder+1)
    return coeff

def polyval2d(x, y, m):
    """Get values for the 2-D polynomial values

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        m (:class:`numpy.ndarray`): Coefficients of the 2-D polynomial.
    Returns:
        z (:class:`numpy.ndarray`): Values of the 2-D polynomial.
    """
    yorder = m.shape[0] - 1
    xorder = m.shape[1] - 1
    z = np.zeros_like(x)
    for j,i in itertools.product(range(yorder+1), range(xorder+1)):
        z += m[j,i] * x**i * y**j
    return z

def gaussian_gen_rot2d(A, center, alpha, beta, theta, xx, yy):
    cx, cy = center
    ax, ay = alpha
    bx, by = beta
    sint, cost = math.sin(theta), math.cos(theta)
    xxnew = cost*(xx-cx) - sint*(yy-cy) + cx
    yynew = sint*(xx-cx) + cost*(yy-cy) + cy
    return A*np.exp(-(np.power(np.abs(xxnew-cx)/ax, bx) +
                      np.power(np.abs(yynew-cy)/ay, by)))

class Fit2DModel(object):
    def __init__(self, data, x0, y0):
        self.data = data
        self.x0, self.y0 = x0, y0
        self.ny, self.nx = data.shape
        self.yy, self.xx = np.mgrid[0:self.ny, 0:self.nx]

    def errfunc(self, param):
        residual = self.data - self.fitfunc(param)
        return residual.flatten()

    def plot(self, figname=None, show=False):
        sample = 0.1
        fitimg = self.fitfunc(self.param)
        fitimg2 = self.fitfunc(self.param, sample=sample)
        resimg = self.data - fitimg

        center_x = self.p['center_x']
        center_y = self.p['center_y']
        alpha_x  = self.p['alpha_x']
        alpha_y  = self.p['alpha_y']
        beta_x   = self.p['beta_x']
        beta_y   = self.p['beta_y']
        theta    = self.p['theta']

        fig = plt.figure(figsize=(12,6), dpi=150)
        ax1  = fig.add_axes([0.05, 0.1, 0.25, 0.8])
        ax2  = fig.add_axes([0.32, 0.1, 0.25, 0.8])
        axc1 = fig.add_axes([0.59, 0.1, 0.015, 0.8])
        ax3  = fig.add_axes([0.66, 0.1, 0.25, 0.8])
        axc2 = fig.add_axes([0.93, 0.1, 0.015, 0.8])

        cax1 = ax1.imshow(self.data, vmin=self.data.min(), vmax=self.data.max(),
                            interpolation='none', aspect=1, origin='bottom')
        cax2 = ax2.imshow(fitimg2, vmin=self.data.min(), vmax=self.data.max(),
                            interpolation='none', aspect=1, origin='bottom')
        cax3 = ax3.imshow(resimg, vmin=resimg.min(), vmax=resimg.max(),
                            interpolation='none', aspect=1, origin='bottom')

        ax1.plot(center_x, center_y, 'kx')

        ax2.axhline(y=center_y/sample, color='k', ls='--', lw=0.7)
        ax2.axvline(x=center_x/sample, color='k', ls='--', lw=0.7)

        # plot rotation angle
        if True:
            x1 = center_x - center_y*math.tan(theta)
            y1 = 0
            x2 = center_x + (self.ny-1 - center_y)*math.tan(theta)
            y2 = self.ny-1
            ax2.plot([x1/sample,x2/sample],[y1/sample,y2/sample],'k-',lw=0.7)
            x1 = 0
            y1 = center_y + center_x*math.tan(theta)
            x2 = self.nx-1
            y2 = center_y - (self.nx-1 - center_x)*math.tan(theta)
            ax2.plot([x1/sample,x2/sample],[y1/sample,y2/sample],'k-',lw=0.7)

        if True:
            for ratio in [0.25, 0.5, 0.75]:
                x_lst, y_lst = [], []
                c = -math.log(ratio)
                for alpha in np.deg2rad(np.linspace(0, 360, 360)):
                    A1 = math.pow(abs(math.cos(alpha))/alpha_x, beta_x)
                    A2 = math.pow(abs(math.sin(alpha))/alpha_y, beta_y)
                    r = math.pow(c/(A1+A2), 2./(beta_x+beta_y))
                    func = lambda r: A1*r**beta_x + A2*r**beta_y - c
                    func1 = lambda r: A1*beta_x*r**(beta_x-1) + A2*beta_y*r**(beta_y-1)
                    while(True):
                        if abs(func(r)) < 1e-4:
                            break
                        dr = func(r)/func1(r)
                        r -= dr
                    x = r*math.cos(alpha) + center_x
                    y = r*math.sin(alpha) + center_y
                    x_lst.append(x)
                    y_lst.append(y)
                x_lst = np.array(x_lst)
                y_lst = np.array(y_lst)
                sint, cost = math.sin(theta), math.cos(theta)
                x_lst_new =  cost*(x_lst - center_x) + sint*(y_lst - center_y) + center_x
                y_lst_new = -sint*(x_lst - center_x) + cost*(y_lst - center_y) + center_y
                ax2.plot(x_lst_new/sample, y_lst_new/sample, 'k-', lw=0.7)

        else:
            cx = self.param[1]
            cy = self.param[2]
            sx = self.param[3]
            sy = self.param[4]
            for ratio in (0.25, 0.5, 0.75):
                x_lst, y_lst = [], []
                for alpha in np.linspace(0,360,360):
                    beta = alpha/180.*math.pi + self.rotate
                    r2 = -math.log(ratio)/(math.cos(beta)**2/2/sx**2 + math.sin(beta)**2/2/sy**2)
                    r = math.sqrt(r2)
                    x = r*math.cos(alpha/180.*math.pi) + cx
                    y = r*math.sin(alpha/180.*math.pi) + cy
                    x_lst.append(x)
                    y_lst.append(y)
                x_lst = np.array(x_lst)
                y_lst = np.array(y_lst)
                sint, cost = math.sin(self.rotate), math.cos(self.rotate)
                x_lst_new = cost*(x_lst-cx) - sint*(y_lst-cy) + cx
                y_lst_new = sint*(x_lst-cx) + cost*(y_lst-cy) + cy
                ax2.plot(x_lst_new*sample, y_lst_new*sample, 'k-')

        for ax in [ax1, ax3]:
            ax.set_xlim(-0.5, self.nx-0.5)
            ax.set_ylim(-0.5, self.ny-0.5)
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
        #ax2.set_xlim(-0.5, self.nx*sample-0.5)
        #ax2.set_ylim(-0.5, self.ny*sample-0.5)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        cbar1 = fig.colorbar(cax1, cax=axc1)
        cbar2 = fig.colorbar(cax3, cax=axc2)

        redchi2 = (resimg**2).sum()/(resimg.size-1-len(self.param))
        self.redchi2 = redchi2
        fwhm_x = 2*alpha_x*math.pow(math.log(2), 1/beta_x)
        fwhm_y = 2*alpha_y*math.pow(math.log(2), 1/beta_y)
        str_lst = [u'\u03b1 (x, y) = (%4.2f, %4.2f)'%(alpha_x, alpha_y),
                   u'FWHM (x, y) = (%4.2f, %4.2f)'%(fwhm_x, fwhm_y),
                   u'\u03b2 (x, y) = (%4.2f, %4.2f)'%(beta_x, beta_y),
                   u'\u03b8 = %+5.2f'%(theta/math.pi*180),
                   u'red\u03c7\xb2 = %5.3e'%redchi2,
                   ]
        ax3.text(0,1,'\n'.join(str_lst), family='Times New Roman',fontsize=10)
        fig.suptitle(self.__class__.__name__)
        if figname is not None:
            fig.savefig(figname)
        if show:
            plt.show()
        plt.close(fig)


class Gaussian2D(Fit2DModel):
    def fitfunc(self, param, sample=1):
        A, cx, cy, ax, ay, bkg = param
        if sample == 1:
            yy, xx = self.yy, self.xx
        else:
            yy, xx = np.mgrid[0:self.ny-1+1e-6:sample, 0:self.nx-1+1e-6:sample]
        return gaussian_gen_rot2d(A, (cx, cy), (ax, ay), (2, 2), 0, xx, yy) + bkg

    def fit(self):
        vmin, vmax = self.data.min(), self.data.max()
        param0 = [vmax-vmin, self.nx/2, self.ny/2, 4., 4., vmin]
        res = opt.least_squares(self.errfunc, param0)
        self.param = res['x']
        self.p = {
                'A':        self.param[0],
                'center_x': self.param[1],
                'center_y': self.param[2],
                'alpha_x':  self.param[3],
                'alpha_y':  self.param[4],
                'beta_x':   2,
                'beta_y':   2,
                'theta':    0,
                'bkg':      self.param[5],
                }
        #fig.savefig('%s_%04d_%04d.png'%(self.__class__.__name__, self.xc0, self.yc0))

class Gaussian2DRot(Fit2DModel):
    def fitfunc(self, param, sample=1):
        A, cx, cy, ax, ay, theta, bkg = param
        if sample == 1:
            yy, xx = self.yy, self.xx
        else:
            yy, xx = np.mgrid[0:self.ny-1+1e-6:sample, 0:self.nx-1+1e-6:sample]
        return gaussian_gen_rot2d(A, (cx, cy), (ax, ay), (2, 2), theta, xx, yy) + bkg

    def fit(self):
        vmin, vmax = self.data.min(), self.data.max()
        bounds = ((0,          np.inf),    # for A
                  (0,          self.nx),   # for center_x
                  (0,          self.ny),   # for center_y
                  (0,          self.nx),   # for alpha_x
                  (0,          self.ny),   # for alpha_y
                  (-math.pi/2, math.pi/2), # for theta
                  (0,          vmax),      # for background
                )
        param0 = [vmax-vmin, self.nx/2., self.ny/2., 4., 4., 0., vmin]
        res = opt.least_squares(self.errfunc, param0, bounds=list(zip(*bounds)))
        self.param = res['x']
        self.p = {
                'A':        self.param[0],
                'center_x': self.param[1],
                'center_y': self.param[2],
                'alpha_x':  self.param[3],
                'alpha_y':  self.param[4],
                'beta_x':   2,
                'beta_y':   2,
                'theta':    self.param[5],
                'bkg':      self.param[6],
                }

class GaussianGen2DRot(Fit2DModel):
    def fitfunc(self, param, sample=1):
        A, cx, cy, ax, ay, bx, by, theta, bkg = param
        if sample == 1:
            yy, xx = self.yy, self.xx
        else:
            yy, xx = np.mgrid[0:self.ny-1+1e-6:sample, 0:self.nx-1+1e-6:sample]
        return gaussian_gen_rot2d(A, (cx, cy), (ax, ay), (bx, by), theta, xx, yy) + bkg
    
    def fit(self):
        vmin, vmax = self.data.min(), self.data.max()
        bounds  = ((0,          np.inf),
                   (0,          self.nx),
                   (0,          self.ny),
                   (0,          np.inf),
                   (0,          np.inf),
                   (0,          5),
                   (0,          5),
                   (-math.pi/2, math.pi/2), # for theta
                   (0,          vmax),
                   )
        param0 = [vmax-vmin, self.nx/2., self.ny/2., 4., 4., 2., 2., 0., vmin]
        res = opt.least_squares(self.errfunc, param0, bounds=list(zip(*bounds)))
        self.param = res['x']
        self.p = {
                'A':        self.param[0],
                'center_x': self.param[1],
                'center_y': self.param[2],
                'alpha_x':  self.param[3],
                'alpha_y':  self.param[4],
                'beta_x':   self.param[5],
                'beta_y':   self.param[6],
                'theta':    self.param[7],
                'bkg':      self.param[8],
                }

