import os
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

def save_fits(*args):
    '''
    Save the fits file.

    A wrapper of `astropy.io.fits.writeto()`. If the output file exists, it will
    be removed without any warnings.
    
    '''
    if os.path.exists(args[0]):
        os.remove(args[0])
        logger.warning('Overwrite FITS file: %s'%args[0])
    fits.writeto(*args)
    logger.info('Save FITS file: %s'%args[0])

def combine_fits(filename_lst,dst_filename,
        mode       = 'mean',  # mode = ['mean'|'sum']
        header     = True,    # keep fits header?
        upper_clip = None,
        lower_clip = None,
        nite       = None,
        maxiter    = None,
        key_exptime= 'EXPTIME'
        ):
    '''Combine multiple FITS images.

    Args:
        filename_lst (list): A list containing names of files to be combined.
        dst_filename (str): Name of the output FITS file.
        mode (str): Combine mode. Either "mean" or "sum".
        header (bool): Whether the FITS headers are kept in the output file.
        upper_clip (float): Upper threshold of the sigma-clipping. Default is
            *None*.
        lower_cli (float): Lower threshold of the sigma-cipping. Default is
            *None*.
        nite (int): Number of iterations.
        maxiter (maxiter): Maximum number of iterations.
        key_exptime (str): Keyword of the exposuretime.
    Returns:
        No returns.

    '''

    clip = not (upper_clip == None and lower_clip == None)

    if mode in ['mean','sum']:
        for ifile,filename in enumerate(filename_lst):
            data, head = fits.getdata(filename, header=True)
            if ifile == 0:
                head0 = head
                exptime = 0.0
                if clip:
                    all_data = []
                else:
                    data_sum = np.zeros_like(data)
            if clip:
                all_data.append(data)
            else:
                data_sum += data
            exptime += head[key_exptime]

    if clip:
        all_data = np.array(all_data)
        all_mask = np.ones_like(all_data)>0
        mask = (all_data == all_data.max(axis=0))
        nite = 0
        while(True):
            nite += 1
            mdata = np.ma.masked_array(all_data, mask=mask)
            m = mdata.mean(axis=0,dtype='float64').data
            s = mdata.std(axis=0,dtype='float64').data
            new_mask = np.ones_like(mask)>0
            for i,filename in enumerate(filename_lst):
                data = all_data[i,:,:]
                if upper_clip != None:
                    mask1 = data > m + abs(upper_clip)*s
                else:
                    # mask1 = [False....]
                    mask1 = np.ones_like(data)<0
                if lower_clip != None:
                    mask2 = data < m - abs(lower_clip)*s
                else:
                    # mask2 = [False....]
                    mask2 = np.ones_like(data)<0
                new_mask[i,:,:] = np.logical_or(mask1, mask2)
            if nite >= maxiter:
                break
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask

        mdata = np.ma.masked_array(all_data, mask=mask)
        mean = mdata.mean(axis=0,dtype='float64').data
        data_sum = mean*len(filename_lst)

    if mode == 'mean':
        data_sum /= len(filename_lst)
        exptime  /= len(filename_lst)

    head0[key_exptime] = exptime

    if header:
        save_fits(dst_filename,data_sum,head0)
    else:
        save_fits(dst_filename,data_sum)

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
        xwindow (int): Window size along *x*-axis.
        ywindow (int): Window size along *y*-axis.
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

class CCDImage(object):
    '''Class for CCD image.

    Attributes:
        data_region_lst (list): List containing Data Regions.

    '''
    def __init__(self):
        self.data_region_lst = []

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
            >>> from edrs.ccdproc import array_to_table

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
    '''

    array = np.zeros(shape, dtype=table.dtype[-1].type)
    coords = [table[col] for col in table.dtype.names[0:-1]]
    array[coords] = table['value']

    return array
