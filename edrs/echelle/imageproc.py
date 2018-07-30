import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import scipy.interpolate as intp

def combine_images(data_lst,
        mode       = 'mean',  # mode = ['mean'|'sum']
        upper_clip = None,
        lower_clip = None,
        maxiter    = None,
        ):
    '''Combine multiple FITS images.

    Args:
        data_lst (list): A list of 2D `numpy.array`.
        mode (string): Combine mode. Either "mean" or "sum".
        upper_clip (float): Upper threshold of the sigma-clipping. Default is
            *None*.
        lower_clip (float): Lower threshold of the sigma-clipping. Default is
            *None*.
        maxiter (maxiter): Maximum number of iterations.
    Returns:
        :class:`numpy.array`.

    '''

    # if anyone of upper_clip and lower_clip is not None, then clip is True
    clip = (upper_clip is not None) or (lower_clip is not None)

    nimage = len(data_lst)
    data_lst = np.array(data_lst)

    if clip:
        mask_lst = (data_lst == data_lst.max(axis=0))
        niter = 0
        while(True):
            niter += 1
            mdata = np.ma.masked_array(data_lst, mask=mask_lst)
            m = mdata.mean(axis=0, dtype=np.float64).data
            s = mdata.std(axis=0, dtype=np.float64).data
            new_mask_lst = np.ones_like(mask_lst, dtype=np.bool)
            for i, data in enumerate(data_lst):

                # parse upper clipping
                if upper_clip is None:
                    # mask1 = [False....]
                    mask1 = np.zeros_like(data, dtype=np.bool)
                else:
                    mask1 = data > m + abs(upper_clip)*s

                # parse lower clipping
                if lower_clip is None:
                    # mask2 = [False....]
                    mask2 = np.zeros_like(data, dtype=np.bool)
                else:
                    mask2 = data < m - abs(lower_clip)*s

                new_mask_lst[i,:,:] = np.logical_or(mask1, mask2)
            if niter >= maxiter:
                break
            if new_mask_lst.sum() == mask_lst.sum():
                break
            mask_lst = new_mask_lst

        mdata = np.ma.masked_array(data_lst, mask=mask_lst)
        mean = mdata.mean(axis=0, dtype=np.float64).data

        if mode == 'mean':
            return mean
        elif mode == 'sum':
            return mean*nimage
        else:
            raise ValueError
            return None
    else:
        if mode == 'mean':
            return data_lst.mean(axis=0, dtype=np.float64)
        elif mode == 'sum':
            return data_lst.sum(axis=0, dtype=np.float64)
        else:
            raise ValueError
            return None

def make_mask():
    '''
    Generate a mask
    1: pixel does not covered by read out region of the detector
    2: bad pixel
    3: flux saturated
    4: cosmic ray
    '''
    pass


def savitzky_golay_2d(z, xwindow, ywindow, xorder, yorder, derivative=None):
    '''Savitzky-Golay 2D filter, with different window size and order along *x*
    and *y* directions.

    Args:
        z (:class:`numpy.array`): Input 2-d array.
        xwindow (integer): Window size along *x*-axis.
        ywindow (integer): Window size along *y*-axis.
        xorder (float): Degree of polynomial along *x*-axis.
        yorder (float): Degree of polynomial along *y*-axis.
        derivative (str): *None*, *col*, *row*, or *both*.
    Returns:
        :class:`numpy.array` or tuple: Output 2-d array, or a tuple containing
            derivative arries along *x*- and *y*-axes, respetively, if
            derivative = "both".
        


    '''
    if xwindow%2 == 0:
        xwindow += 1
    if ywindow%2 == 0:
        ywindow += 1

    exps = [(k-n, n) for k in range(max(xorder, yorder)+1) for n in range(k+1)
            if k-n <= xorder and n <= yorder]
    xhalf = xwindow//2
    yhalf = ywindow//2
    xind = np.arange(-xhalf, xhalf+1, dtype=np.float64)
    dx = np.repeat(xind, ywindow)
    yind = np.arange(-yhalf, yhalf+1, dtype=np.float64)
    dy = np.tile(yind, [xwindow, 1]).reshape(xwindow*ywindow,)

    A = np.empty(((xwindow*ywindow), len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0])*(dy**exp[1])

    newshape = z.shape[0] + 2*yhalf, z.shape[1] + 2*xhalf
    Z = np.zeros((newshape))
    # top band
    band = z[0,:]
    Z[:yhalf, xhalf:-xhalf] = band - np.abs(np.flipud(z[1:yhalf+1,:])-band)
    # bottom band
    band = z[-1,:]
    Z[-yhalf:, xhalf:-xhalf] = band + np.abs(np.flipud(z[-yhalf-1:-1])-band)
    # left band
    band = np.tile(z[:,0].reshape(-1,1), [1, xhalf])
    Z[yhalf:-yhalf, :xhalf] = band - np.abs(np.fliplr(z[:,1:xhalf+1])-band)
    # right band
    band = np.tile(z[:,-1].reshape(-1,1), [1, xhalf])
    Z[yhalf:-yhalf, -xhalf:] = band + np.abs(np.fliplr(z[:,-xhalf-1:-1])-band)
    # central region
    Z[yhalf:-yhalf, xhalf:-xhalf] = z
    # top left corner
    band = z[0,0]
    Z[:yhalf, :xhalf] = band - np.abs(np.flipud(np.fliplr(z[1:yhalf+1, 1:xhalf+1]))-band)
    # bottom right corner
    band = z[-1,-1]
    Z[-yhalf:, -xhalf:] = band + np.abs(np.flipud(np.fliplr(z[-yhalf-1:-1,-xhalf-1:-1]))-band)
    # top right corner
    band = Z[yhalf, -xhalf:]
    Z[:yhalf, -xhalf:] = band - np.abs(np.flipud(Z[yhalf+1:2*yhalf+1,-xhalf:])-band)
    # bottom left corner
    band = Z[-yhalf:,xhalf].reshape(-1,1)
    Z[-yhalf:, :xhalf] = band - np.abs(np.fliplr(Z[-yhalf:, xhalf+1:2*xhalf+1])-band)

    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((ywindow, xwindow))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((ywindow, xwindow))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].rehsape((ywindow, xwindow))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((ywindow, xwindow))
        r = np.linalg.pinv(A)[2].rehsape((ywindow, xwindow))
        return (scipy.signal.fftconvolve(Z, -r, mode='valid'),
                scipy.signal.fftconvolve(Z, -c, mode='valid'))

def array_to_table(array):
    '''Convert the non-zeros elements of a Numpy array to a stuctured array.

    Args:
        array (:class:`numpy.array`): Input Numpy array.
    Returns:
        :class:`numpy.dtype`: Numpy stuctured array.
    See also:
        :func:`table_to_array`
    Examples:
        Below shows an example of converting a numpy 2-d array `a` to a
        structured array `t`.
        The first few coloumns (`axis_0`, `axis_1`, ... `axis_n-1`) in `t`
        correspond to the coordinates of the *n*-dimensional input array, and
        the last column (`value`) are the elements of the input array.
        The reverse process is :func:`table_to_array`

        .. code-block:: python

            >>> import numpy as np
            >>> from edrs.echelle.imageproc import array_to_table

            >>> a = np.arange(12).reshape(3,4)
            >>> a
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]])
            >>> t = array_to_table(a)
            >>> t
            array([(0, 1,  1), (0, 2,  2), (0, 3,  3), (1, 0,  4), (1, 1,  5),
                   (1, 2,  6), (1, 3,  7), (2, 0,  8), (2, 1,  9), (2, 2, 10),
                   (2, 3, 11)], 
                  dtype=[('axis_0', '<i2'), ('axis_1', '<i2'), ('value', '<i8')])

    '''
    dimension = len(array.shape)
    types = [('axis_%d'%i, np.int16) for i in range(dimension)]
    types.append(('value', array.dtype.type))
    names, formats = list(zip(*types))
    custom = np.dtype({'names': names, 'formats': formats})
    
    table = []
    ind = np.nonzero(array)
    for coord, value in zip(zip(*ind), array[ind]):
        row = list(coord)
        row.append(value)
        row = np.array(tuple(row), dtype=custom)
        table.append(row)
    table = np.array(table, dtype=custom)
    return(table)

def table_to_array(table, shape):
    '''Convert a structured array to Numpy array.

    This is the reverse process of :func:`array_to_table`.
    For the elements of which coordinates are not listed in the table, zeros are
    filled.

    Args:
        table (:class:`numpy.dtype`): Numpy structured array.
        shape (tuple): Shape of output array.
    Returns:
        :class:`numpy.array`: Numpy array.
    See also:
        :func:`array_to_table`
    Examples:
        Below shows an example of converting a numpy 2-d array `a` to a
        structured array `t` using :func:`array_to_table`, and then converting
        `t` back to `a` using :func:`table_to_array`.

        .. code-block:: python

            >>> import numpy as np
            >>> from edrs.echelle.imageproc import array_to_table

            >>> a = np.arange(12).reshape(3,4)
            >>> a
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]])
            >>> t = array_to_table(a)
            >>> t
            array([(0, 1,  1), (0, 2,  2), (0, 3,  3), (1, 0,  4), (1, 1,  5),
                   (1, 2,  6), (1, 3,  7), (2, 0,  8), (2, 1,  9), (2, 2, 10),
                   (2, 3, 11)], 
                  dtype=[('axis_0', '<i2'), ('axis_1', '<i2'), ('value', '<i8')])
            >>> a = table_to_array(a, (3,4))
            >>> a
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]])

    '''

    array = np.zeros(shape, dtype=table.dtype[-1].type)
    coords = tuple(table[col] for col in table.dtype.names[0:-1])
    array[coords] = table['value']

    return array


def fix_pixels(data, mask, direction, method):
    '''Fix specific pixels of the CCD image by interpolating surrounding pixels.

    Args:
        data (:class:`numpy.array`): Input image as a 2-D array.
        mask (:class:`numpy.array`): Mask of pixels to be fixed. This array
            shall has the same shape as **data**.
        direction (string or integer): Interpolate along which axis (*X* = 1,
            *Y* = 0).
        method (string): Interpolationg method ('linear' means linear
            interpolation, and 'cubic' means cubic spline interpolation).
    Returns:
        :class:`numpy.array`: The fixed image as a 2-D array.
    '''
    # make a new copy of the input data
    newdata = np.copy(data)

    # determine the axis
    if isinstance(direction, str):
        direction = {'x':1, 'y':0}[direction.lower()]

    # find the rows or columns to interpolate
    masklist = mask.sum(axis=direction)

    # determine interpolation method
    k = {'linear':1, 'cubic':3}[method]

    if direction == 0:
        # fix along Y axis
        x = np.arange(data.shape[0])
        cols = np.nonzero(masklist)[0]
        for col in cols:
            m = mask[:,col]
            rm = ~m
            y = data[:,col]
            f = intp.InterpolatedUnivariateSpline(x[rm],y[rm],k=k)
            newdata[:,col][m] = f(x[m])
    elif direction == 1:
        # fix along X axis
        x = np.arange(data.shape[1])
        rows = np.nonzero(masklist)[0]
        for row in rows:
            m = mask[row,:]
            rm = ~m
            y = data[row,:]
            f = intp.InterpolatedUnivariateSpline(x[rm],y[rm],k=k)
            newdata[row,:][m] = f(x[m])
    else:
        print('direction must be 0 or 1')
        raise ValueError

    return newdata
