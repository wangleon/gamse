import math
import numpy as np

def get_clip_mean(x, err=None, mask=None, high=3, low=3, maxiter=5):
    """Get the mean value of an input array using the sigma-clipping method

    Args:
        x (:class:`numpy.ndarray`): The input array.
        err (:class:`numpy.ndarray`): Errors of the input array.
        mask (:class:`numpy.ndarray`): Initial mask of the input array.
        high (float): Upper rejection threshold.
        low (float): Loweer rejection threshold.
        maxiter (int): Maximum number of iterations.

    Returns:
        tuple: A tuple containing:

            * **mean** (*float*) – Mean value after the sigma-clipping.
            * **std** (*float*) – Standard deviation after the sigma-clipping.
            * **mask** (:class:`numpy.ndarray`) – Mask of accepted values in the
              input array.
    """
    x = np.array(x)
    if mask is None:
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
        new_mask = mask * (x < mean + high*std) * (x > mean - low*std)

        if mask.sum() == new_mask.sum():
            break
        else:
            mask = new_mask

    return mean, std, new_mask

def linear_fit(x,y,yerr=None,maxiter=10,high=3.0,low=3.0,full=False):
    """Fit the input arrays using a linear function.

    Args:
        x (:class:`numpy.ndarray`): The input X values.
        y (:class:`numpy.ndarray`): The input Y values.
        yerr (:class:`numpy.ndarray`): Errors of the Y array.
        maxiter (int): Maximum number of iterations.
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
    """
    x = np.array(x)
    y = np.array(y)
    if yerr is not None:
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


def iterative_polyfit(x, y, yerr=None, deg=3, mask=None, maxiter=10,
    upper_clip=None, lower_clip=None):
    """Fit data with polynomial iteratively. This is a wrap of numpy.polyfit
    function.

    Args:
        x (:class:`numpy.ndarray`): Input X values.
        y (:class:`numpy.ndarray`): Input Y values.
        yerr (:class:`numpy.ndarray`): Errors of **y**.
        deg (int): Degree of polynomial.
        mask (:class:`numpy.ndarray`): Input mask.
        maxiter (int): Maximum number of iterations.
        lower_clip (float): Lower sigma-clipping value.
        upper_clip (float): Upper sigma-clipping value.

    Returns:
        tuple: A tuple containing:

            * **coeff** (:class:`numpy.ndarray`) – Coefficients of polynomial.
            * **yfit** (:class:`numpy.ndarray`) – Fitted y values.
            * **yres** (:class:`numpy.ndarray`) – Residuals of y values.
            * **mask** (:class:`numpy.ndarray`) – Mask of y values.
            * **std** (float) – Standard deviation.
    """

    x = np.array(x)
    y = np.array(y)
    if yerr is None:
        pass
    else:
        yerr = np.array(yerr)
    if mask is None:
        mask = np.ones_like(x, dtype=np.bool)
    
    for ite in range(maxiter):
        coeff = np.polyfit(x[mask], y[mask], deg=deg)
        yfit = np.polyval(coeff, x)
        yres = y - yfit
        std = yres[mask].std(ddof=1)

        # replace np.nan by np.inf to avoid runtime warning
        yres[np.isnan(yres)] = np.inf

        # generate new mask
        # make a copy of existing mask
        new_mask = mask * np.ones_like(mask, dtype=np.bool)
        # give new mask with lower and upper clipping value
        if lower_clip is not None:
            new_mask *= (yres > -lower_clip * std)
        if upper_clip is not None:
            new_mask *= (yres < upper_clip * std)

        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    return coeff, yfit, yres, mask, std
