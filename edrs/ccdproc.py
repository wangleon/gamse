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
        mode (str): Combine mode. `'mean'` or `'sum'`.
        header (bool): Whether the FITS headers are kept in the output file.
        upper_clip (float): Upper threshold of the sigma-clipping. Default is
            None.
        lower_cli (float): Lower threshold of the sigma-cipping. Default is
            None.
        nite (int): Number of iterations.
        maxiter (maxiter): Maximum number of iterations.
        key_exptime (str): Keyword of the exposuretime.
    Returns:
        No returns

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
