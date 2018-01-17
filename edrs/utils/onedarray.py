from itertools import tee
import numpy as np

def get_edge_bin(array):
    '''
    Detect the edge indcies of a binary 1-D array.

    Args:
        array (:class:`numpy.array`): A list or Numpy 1d array, with binary
            (0/1) or boolean (True/False) values.

    Returns:
        list: A list containing starting and ending indices of the non-zero
            blocks.

    Examples:

        .. code-block:: python

            >>> a = [0,1,1,0,0,0,1,0,1]
            >>> get_edge_bin(a)
            [(1, 3), (6, 7), (8, 9)]
            >>> b = [True, False, True, True, False, False]
            >>> get_edge_bin(b)
            [(0, 1), (2, 4)]
    '''
    array1 = np.int64(array)
    array1 = np.insert(array1, 0, 0)
    array1 = np.append(array1, 0)
    tmp = array1 - np.roll(array1, 1)
    i1_lst = np.nonzero(tmp == 1)[0] - 1
    i2_lst = np.nonzero(tmp ==-1)[0] - 1
    return list(zip(i1_lst, i2_lst))

def get_local_minima(x, window=None):
    '''
    Get the local minima of a 1d array in a window.

    Args:
        x (:class:`numpy.array`): A list or Numpy 1d array.
        window (integer or :class:`numpy.array`): An odd integer as the length of searching window.
    Returns:
        tuple: A tuple containing:

            * **index** (:class:`numpy.array`): a numpy 1d array containing 
              indices of all local minima.
            * **x[index]** (:class:`numpy.array`): a numpy 1d array containing
              values of all local minima.

    '''
    x = np.array(x)
    dif = np.diff(x)
    ind = dif > 0
    tmp = np.logical_xor(ind, np.roll(ind,1))
    idx = np.logical_and(tmp,ind)
    index = np.where(idx)[0]
    if window is None:
        # window is not given
        return index, x[index]
    elif isinstance(window, int):
        # window is an integer
        # window must be an odd integer
        if window%2 != 1:
            raise ValueError
        halfwin = (window-1)//2
        index_lst = []
        for i in index:
            i1 = max(0, i-halfwin)
            i2 = min(i+halfwin+1, x.size)
            if i == x[i1:i2].argmin() + i1:
                index_lst.append(i)
        index_lst = np.array(index_lst)
        return index_lst, x[index_lst]
    elif isinstance(window, np.ndarray):
        # window is a numpy array
        if np.issubdtype(window.dtype, int):
            # window are integers
            if 0 in window%2:
                # not all of the windows are odd
                raise ValueError
            else:
                halfwin_lst = (window-1)//2
                index_lst = []
                for i in index:
                    halfwin = halfwin_lst[i]
                    i1 = max(0, i-halfwin)
                    i2 = min(i+halfwin+1, x.size)
                    if i == x[i1:i2].argmin() + i1:
                        index_lst.append(i)
                index_lst = np.array(index_lst)
                return index_lst, x[index_lst]
        else:
            # window are not integers
            print('window array are not integers')
            raise ValueError
    else:
        raise ValueError

def implete_none(lst):
    '''
    Replace the None elemnets at the beginning and the end of list by auto
    increment integers.
    
    Convert the first and last few `None` elements to auto increment integers.
    These integers are determined by the first and last integers in the input
    array.
    While the `None` elements between two integers in the input list will
    remain.

    Args:
        lst (list): A list contaning None values.
    Returns:
        newlst (list): A list containing auto increment integers.
	
    Examples:
        .. code-block:: python

            >>> a = [None,None,3,4,None,5,6,None,None]
            >>> implete_none(a)
            [1, 2, 3, 4, None, 5, 6, 7, 8]

    '''
    # filter the None values
    notnone_lst = [v for v in lst if v is not None]
    for i, v in enumerate(lst):
        if v == notnone_lst[0]:
            # first not-None element and its index
            notnone1 = i
            value1   = v
        if v == notnone_lst[-1]:
            # last not-None element and its index
            notnone2 = i
            value2   = v
    newlst = []
    for i,v in enumerate(lst):
        if i < notnone1:
            newlst.append(value1-(notnone1-i))
        elif i > notnone2:
            newlst.append(value2+(i-notnone2))
        else:
            newlst.append(v)
    return newlst


def derivative(*args, **kwargs):
    '''
    Get the first derivative of (x, y).

    Args:
        x (list or :class:`numpy.array`): X-values of the input array.
        y (list or :class:`numpy.array`): Y-values of the input array.
        points (int): Number of points used to calculate derivative (optional, default is 3).
    Returns:
        :class:`numpy.array`: Derivative of the input array.
    Notes:
        If **y** is not given, the first argument will be taken as *y*, and the
        differential of the input array will be returned.
    '''
    if len(args) == 1:
        y = np.array(args[0], dtype=np.float64)
        x = np.arange(y.size)
    elif len(args) == 2:
        x = np.array(args[0], dtype=np.float64)
        y = np.array(args[1], dtype=np.float64)
    npts = x.size
    points = kwargs.pop('points', 3)
    if points == 3:
        der = (np.roll(y,-1) - np.roll(y,1))/(np.roll(x,-1) - np.roll(x,1))
        a = np.array([-3., 4., -1.])
        der[0]  = (a*y[0:3]).sum() / (a*x[0:3]).sum()
        der[-1] = (-a[::-1]*y[-3:]).sum() / (-a[::-1]*x[-3:]).sum()
        return der

def pairwise(array):
    '''Return pairwises of an iterable arrary.

    Args:
        array (list or :class:`numpy.array`): The input iterable array.
    Returns:
        :class:`zip`: zip objects.
    '''
    a, b = tee(array)
    next(b, None)
    return zip(a, b)
