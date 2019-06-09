import math
import itertools
import numpy as np

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
    coeff, residuals, _, _ = np.linalg.lstsq(G, z, rcond=None)
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

