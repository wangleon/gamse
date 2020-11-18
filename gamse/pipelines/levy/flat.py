import numpy as np
import scipy.optimize as opt

def errfunc(p, x, y, fitfunc):
    return y - fitfunc(p, x)

def smooth_flat_flux_func(x, y, npoints):
    """
    Args:
        x ():
        y ():
        npoints (int):

    Returns:

    """
    deg = 3
    maxiter = 10

    def get_bump(p, x):
        bump, x1, x2 = p[0:3]
        bkg = np.zeros_like(x, dtype=x.dtype)
        mask = (x > x1) * (x < x2)
        bkg[mask] = bump
        return bkg

    def fitfunc(p, x):
        coeff = p[3:]
        return np.polyval(coeff, x) + get_bump(p, x)

    mask = np.ones_like(y, dtype=np.bool)
    x1, x2 = 1500/4608, 2800/4608

    # normalize x axis to (0, 1)
    xnorm = x/npoints

    for niter in range(maxiter):
        if niter==0:
            coeff0 = np.polyfit(xnorm[mask], y[mask], deg=deg)
        else:
            bkg = get_bump(p, xnorm)
            xnorm2 = xnorm-bkg
            coeff0 = np.polyfit(xnorm2[mask], y[mask], deg=deg)

        p0 = np.insert(coeff0, 0, [0.0, x1, x2])
        nparam = len(p0)
        lower_bounds = [-np.inf]*nparam
        upper_bounds = [np.inf]*nparam
        lower_bounds[0] = 0.0
        upper_bounds[0] = y.max()
        lower_bounds[1] = 1000/4608
        upper_bounds[1] = 2000/4608
        lower_bounds[2] = 2100/4608
        upper_bounds[2] = 3000/4608
        fitres = opt.least_squares(errfunc, p0,
                    bounds = [tuple(lower_bounds), tuple(upper_bounds)],
                    args   = (xnorm[mask], y[mask], fitfunc),
                    )
        p = fitres['x']

        yfit = fitfunc(p, xnorm)
        yres = y - yfit
        std = yres[mask].std()
        newmask = (yres>-3*std)*(yres<3*std)
        if newmask.sum() == mask.sum():
            break
        mask = newmask

    newy = fitfunc(p, xnorm) - get_bump(p, xnorm)
    return newy, mask
