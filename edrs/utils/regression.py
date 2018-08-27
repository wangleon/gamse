import math
import itertools
import numpy as np

def get_clip_mean(x, err=None, high=3, low=3, maxiter=5):
    '''Get the mean value of an input array using the sigma-clipping method

    Args:
        x (:class:`numpy.ndarray`): The input array.
        err (:class:`numpy.ndarray`): Errors of the input array.
        high (float): Upper rejection threshold.
        low (float): Loweer rejection threshold.
        maxiter (integer): Maximum number of iterations.
    Returns:
        tuple: A tuple containing:

            * **mean** (*float*) – Mean value after the sigma-clipping.
            * **std** (*float*) – Standard deviation after the sigma-clipping.
            * **mask** (:class:`numpy.ndarray`) – Mask of accepted values in the
              input array.
    '''
    x = np.array(x)
    mask = np.zeros_like(x)<1

    niter = 0
    while(True):

        niter += 1
        if err is None:
            mean = x[mask].mean()
            std  = x[mask].std()
        else:
            mean = (x/err*mask).sum()/((1./err*mask).sum())
            std = math.sqrt(((x - mean)**2/err*mask).sum()/((1./err*mask).sum()))

        if maxiter==0 or niter>maxiter:
            # return without new mask
            break

        # calculate new mask
        m1 = x < mean + high*std
        m2 = x > mean - low*std
        new_mask = np.logical_and(m1,m2)

        if mask.sum()==new_mask.sum():
            break
        else:
            mask = new_mask

    return mean, std, new_mask

def linear_fit(x,y,yerr=None,maxiter=10,high=3.0,low=3.0,full=False):
    '''Fit the input arrays using a linear function.

    Args:
        x (:class:`numpy.ndarray`): The input X values.
        y (:class:`numpy.ndarray`): The input Y values.
        yerr (:class:`numpy.ndarray`): Errors of the Y array.
        maxiter (integer): Maximum number of interations.
        high (float): Upper rejection threshold.
        low (float): Lower rejection threshold.
        full (bool): `True` if return all information.

    Returns:
        tuple: A tuple containing:

            * **p** (:class:`numpy.ndarray`) – Parameters of the fitting.
            * **std** (*float*) – Standard deviation of the fitting.
            * **mask** (:class:`numpy.ndarray`) – Mask of the accepted values in
              the input array.
            * **func** (*func*) – Function.
            * **r2** (*float*) – *R*:sup:`2` value.
            * **p_std** (*tuple*) – Standar deviations of the parameters.
    '''
    x = np.array(x)
    y = np.array(y)
    if yerr != None:
        yerr = np.array(yerr)
    mask = np.ones_like(x)>0
    niter = 0
    func = lambda p, x: p[0] + p[1]*np.array(x)
    while(True):
        niter += 1
        if yerr == None:
            xm = x[mask].mean()
            ym = y[mask].mean()
            a = ((x[mask]-xm)*(y[mask]-ym)).sum()/((x[mask]-xm)**2).sum()
            b = ym - a*xm
            p = np.array([b,a])
            vy = func(p,x)
            std = (y[mask]-vy[mask]).std()
        else:
            w = 1./yerr[mask]**2
            w /= w.sum()
            b = (x[mask]*y[mask]*w).sum()
            c = (x[mask]*w).sum()
            d = (y[mask]*w).sum()
            e = (x[mask]**2*w).sum()
            f = e-c**2
            p = np.array([(e*d-c*b)/f, (b-c*d)/f])
            vy = func(p,x)
            std = math.sqrt(((y[mask]-vy[mask])**2*w).sum())

        if maxiter==0 or niter>maxiter:
            # return without new mask
            break

        # calculate new mask
        m1 = y > vy - low*std
        m2 = y < vy + high*std
        new_mask = np.logical_and(m1, m2)

        if mask.sum()==new_mask.sum() or new_mask.sum()<=2:
            break
        else:
            mask = new_mask
    if full:
        xm = x[mask].mean()
        ym = y[mask].mean()
        n  = mask.sum()
        lxx = (x[mask]**2).sum() - n*xm**2
        lyy = (y[mask]**2).sum() - n*ym**2
        lxy = (x[mask]*y[mask]).sum() - n*xm*ym
        r2 = lxy**2/lxx/lyy
        s = math.sqrt((lyy-lxy**2/lxx)/(n-2))
        sa = s/math.sqrt(lxx)
        sb = s*math.sqrt(1./n + xm**2/lxx)
        p_std = np.array([sb,sa])
        return p, std, mask, func, r2, p_std
    else:
        return p, std, mask, func

def polyfit2d(x, y, z, xorder=3, yorder=3, linear=False):
    '''Two-dimensional polynomial fit.

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        z (:class:`numpy.ndarray`): Input Z array.
        xorder (integer): X order.
        yorder (integer): Y order.
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

    '''
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
    '''Get values for the 2-D polynomial values

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        m (:class:`numpy.ndarray`): Coefficients of the 2-D polynomial.
    Returns:
        z (:class:`numpy.ndarray`): Values of the 2-D polynomial.
    '''
    yorder = m.shape[0] - 1
    xorder = m.shape[1] - 1
    z = np.zeros_like(x)
    for j,i in itertools.product(range(yorder+1), range(xorder+1)):
        z += m[j,i] * x**i * y**j
    return z
