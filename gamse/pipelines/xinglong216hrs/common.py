import os
import re
import datetime
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates as mdates

from ...echelle.trace import TraceFigureCommon
from ...utils.obslog import read_obslog
from ..reduction          import Reduction

def parse_bias_frames(logtable, config, pinfo):
    """Parse the bias images and return the bias as an array.

    Args:
        logtable ():
        config ():
        pinfo ():

    Returns:
        bias:
        bias_card_lst:

    """

    rawdata = config['data']['rawdata']
    section = config['reduce.bias']
    bias_file = section['bias_file']

    bias_data_lst = []
    bias_fileid_lst = []

    for logitem in logtable:
        if logitem['object'].strip().lower()=='bias':
            filename = os.path.join(rawdata, logitem['fileid']+'.fits')
            data, head = fits.getdata(filename, header=True)
            mask = get_mask(data, head)
            data, card_lst, overmean = correct_overscan(data, head, mask)

            # print info
            if len(bias_data_lst) == 0:
                print('* Combine Bias Images: {}'.format(bias_file))
                print(' '*2 + pinfo.get_title())
                print(' '*2 + pinfo.get_separator())
            print(' '*2 + pinfo.get_format().format(logitem, overmean))

            bias_data_lst.append(data)
            bias_fileid_lst.append(logitem['fileid'])

    # get number of bias images
    n_bias = len(bias_data_lst)

    prefix = 'HIERARCH GAMSE BIAS '
    bias_card_lst = []
    bias_card_lst.append((prefix + 'NFILE', n_bias))

    if n_bias == 0:
        # there is no bias frames
        bias = None
    else:
        for i, fileid in enumerate(bias_fileid_lst):
            bias_card_lst.append((prefix + 'FILEID {:02d}'.format(i), fileid))

        # there is bias frames
        print(' '*2 + pinfo.get_separator())

        # combine bias images
        bias_data_lst = np.array(bias_data_lst)

        bias = combine_images(bias_data_lst,
                mode       = 'mean',
                upper_clip = section.getfloat('cosmic_clip'),
                maxiter    = section.getint('maxiter'),
                mask       = (None, 'max')[n_bias>=3],
                )

        ############## bias smooth ##################
        if section.getboolean('smooth'):
            # bias needs to be smoothed
            smooth_method = section.get('smooth_method')

            h, w = bias.shape
            if smooth_method in ['gauss','gaussian']:
                # perform 2D gaussian smoothing
                smooth_sigma = section.getint('smooth_sigma')
                smooth_mode  = section.get('smooth_mode')
                bias_smooth = np.zeros_like(bias, dtype=np.float64)
                bias_smooth[0:h//2, :] = gaussian_filter(bias[0:h//2, :],
                                            sigma = smooth_sigma,
                                            mode  = smooth_mode)
                bias_smooth[h//2:h, :] = gaussian_filter(bias[h//2:h, :],
                                            sigma = smooth_sigma,
                                            mode  = smooth_mode)

                # write information to FITS header
                bias_card_lst.append((prefix + 'SMOOTH CORRECTED',  True))
                bias_card_lst.append((prefix + 'SMOOTH METHOD', 'GAUSSIAN'))
                bias_card_lst.append((prefix + 'SMOOTH SIGMA',  smooth_sigma))
                bias_card_lst.append((prefix + 'SMOOTH MODE',   smooth_mode))
            else:
                print('Unknown smooth method: ', smooth_method)
                pass

            bias = bias_smooth
        else:
            # bias not smoothed
            bias_card_lst.append((prefix + 'SMOOTH CORRECTED', False))

        # create new FITS Header for bias
        head = fits.Header()
        for card in bias_card_lst:
            head.append(card)
        fits.writeto(bias_file, bias, header=head, overwrite=True)

        message = 'Bias image written to "{}"'.format(bias_file)
        logger.info(message)
        print(message)

    return bias, bias_card_lst


def get_badpixel_mask(shape, bins):
    """Get the mask of bad pixels and columns.

    Args:
        shape (tuple): Shape of image.
        bins (tuple): CCD bins.

    Returns:
        :class:`numpy.ndarray`: 2D binary mask, where bad pixels are marked with
            *True*, others *False*.

    The bad pixels are found *empirically*.
        
    """
    mask = np.zeros(shape, dtype=np.bool)
    if bins == (1, 1) and shape == (4136, 4096):
        h, w = shape

        mask[349:352, 627:630] = True
        mask[349:h//2, 628]    = True

        mask[1604:h//2, 2452] = True

        mask[280:284,3701]   = True
        mask[274:h//2, 3702] = True
        mask[272:h//2, 3703] = True
        mask[274:282, 3704]  = True

        mask[1720:1722, 3532:3535] = True
        mask[1720, 3535]           = True
        mask[1722, 3532]           = True
        mask[1720:h//2,3533]       = True

        mask[347:349, 4082:4084] = True
        mask[347:h//2,4083]      = True

        mask[h//2:2631, 1909] = True
    else:
        print('No bad pixel information for this CCD size.')
        raise ValueError
    return mask

def get_mask(data, head):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.

    The shape of output mask is determined by the keywords in the input FITS
    header. The numbers of columns and rows are given by::
     
        N (columns) = head['NAXIS1'] - head['COVER']

        N (rows)    = head['NAXIS2'] - head['ROVER']

    where *head* is the input FITS header. 

    """

    saturation_adu = 65535

    # determine shape of output image (also the shape of science region)
    y1 = head.get('CRVAL2', 0)
    rover = head.get('ROVER', 0)
    y2 = y1 + head['NAXIS2'] - rover
    x1 = head.get('CRVAL1', 0)
    cover = head.get('COVER', 64)
    x2 = x1 + head['NAXIS1'] - cover
    newshape = (y2-y1, x2-x1)

    # find the saturation mask
    mask_sat = (data[y1:y2, x1:x2] >= saturation_adu)
    # get bad pixel mask
    rbin = head.get('RBIN', 1)
    cbin = head.get('CBIN', 1)
    bins = (rbin, cbin)
    mask_bad = get_badpixel_mask(newshape, bins=bins)

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

def fix_cr(data):
    """Cosmic ray fixing function.
    """
    m = data.mean(dtype=np.float64)
    s = data.std(dtype=np.float64)
    _mask = data > m + 3.*s
    if _mask.sum()>0:
        x = np.arange(data.size)
        f = InterpolatedUnivariateSpline(x[~_mask], data[~_mask], k=3)
        return f(x)
    else:
        return data

def correct_overscan(data, head, mask=None):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.
        mask (:class:`numpy.ndarray`): Input image mask.
    
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
            * **overmean** (*float) – Mean value of overscan pixels.
    """

    h, w = data.shape
    cover = head.get('COVER', 64)
    x1, x2 = w-cover, w

    # find the overscan level along the y-axis
    ovr_lst1 = data[0:h//2,x1+2:x2].mean(dtype=np.float64, axis=1)
    ovr_lst2 = data[h//2:h,x1+2:x2].mean(dtype=np.float64, axis=1)

    ovr_lst1_fix = fix_cr(ovr_lst1)
    ovr_lst2_fix = fix_cr(ovr_lst2)

    # apply the sav-gol fitler to the mean of overscan
    winlength = 301
    polyorder = 3
    ovrsmooth1 = savgol_filter(ovr_lst1_fix,
                    window_length=winlength, polyorder=polyorder)
    ovrsmooth2 = savgol_filter(ovr_lst2_fix,
                    window_length=winlength, polyorder=polyorder)

    # determine shape of output image (also the shape of science region)
    y1 = head.get('CRVAL2', 0)
    rover = head.get('ROVER', 0)
    y2 = y1 + head['NAXIS2'] - rover
    ymid = (y1 + y2)//2
    x1 = head.get('CRVAL1', 0)
    cover = head.get('COVER', 64)
    x2 = x1 + head['NAXIS1'] - cover
    newshape = (y2-y1, x2-x1)

    # subtract overscan
    new_data = np.zeros(newshape, dtype=np.float64)
    ovrdata1 = np.repeat([ovrsmooth1],x2-x1,axis=0).T
    ovrdata2 = np.repeat([ovrsmooth2],x2-x1,axis=0).T
    new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
    new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2
    overmean = (ovrsmooth1.mean() + ovrsmooth2.mean())/2.

    if mask is not None:
        # fix bad pixels
        bad_mask = (mask&2 > 0)
        new_data = fix_pixels(new_data, bad_mask, 'x', 'linear')

    card_lst = []
    prefix = 'HIERARCH GAMSE OVERSCAN '
    card_lst.append((prefix + 'CORRECTED', True))
    card_lst.append((prefix + 'METHOD',    'smooth:savgol'))
    card_lst.append((prefix + 'WINLEN',    winlength))
    card_lst.append((prefix + 'POLYORDER', polyorder))
    #card_lst.append((prefix+' AXIS-1',    '{}:{}'.format(x1, x2)))
    #card_lst.append((prefix+' AXIS-2',    '{}:{}'.format()))

    return new_data, card_lst, overmean

def select_calib_from_database(database_path, dateobs):
    """Select wavelength calibration file in database.

    Args:
        path (str): Path to search for the calibration files.
        dateobs (str): .
    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
                spectra.
            * **calib** (dict): Previous calibration results.
    """
    
    indexfile = os.path.join(database_path, 'index.dat')
    calibtable = read_obslog(indexfile)

    input_date = dateutil.parser.parse(dateobs)
    if input_date > datetime.datetime(2019, 1, 1):
        mask = calibtable['obsdate'] > datetime.datetime(2019, 1, 1)
    else:                         
        mask = calibtable['obsdate'] < datetime.datetime(2019, 1, 1)
    
    fileid = calibtable[mask][-1]['fileid']

    filename = 'wlcalib.{}.fits'.format(fileid)
    filepath = os.path.join(database_path, filename)
         
    # load spec, calib, and aperset from selected FITS file
    f = fits.open(filepath)
    head = f[0].header
    spec = f[1].data
    f.close()

    calib = get_calib_from_header(head)

    return spec, calib

class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.43,0.86])
        self.ax2 = self.add_axes([0.52,0.50,0.43,0.40])
        self.ax3 = self.add_axes([0.52,0.10,0.43,0.40])
        self.ax4 = self.ax3.twinx()

class Xinglong216HRS(Reduction):

    def __init__(self):
        super(Xinglong216HRS, self).__init__(instrument='Xinglong216HRS')

    def config_ccd(self):
        '''Set CCD images configurations.
        '''
        self.ccd_config

    def overscan(self):
        '''
        Overscan correction for Xinglong 2.16m Telescope HRS.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **skip**,    *bool*, Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,  *str*,  Suffix of the corrected files.
           **plot**,    *bool*, Plot the overscan levels if *yes*.
           **var_fig**, *str*,  Filename of the overscan variation figure.

        '''


        def fix_cr(a):
            m = a.mean(dtype=np.float64)
            s = a.std(dtype=np.float64)
            mask = a > m + 3.*s
            if mask.sum()>0:
                x = np.arange(a.size)
                f = InterpolatedUnivariateSpline(x[~mask],a[~mask],k=3)
                return f(x)
            else:
                return a
        
        # find output suffix for fits
        self.output_suffix = self.config.get('overscan', 'suffix')

        if self.config.getboolean('overscan', 'skip'):
            logger.info('Skip [overscan] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        # keywords for mask
        saturation_adu = 65535

        # path alias
        rawdata = self.paths['rawdata']
        midproc = self.paths['midproc']
        report  = self.paths['report']

        # loop over all files (bias, dark, ThAr, flat...)
        # to correct for the overscan

        # prepare the item list
        item_lst = [item for item in self.log]

        for i, item in enumerate(item_lst):
            logger.info('Correct overscan for item %3d: "%s"'%(
                         item.frameid, item.fileid))

            # read FITS data
            filename = '%s%s.fits'%(item.fileid, self.input_suffix)
            filepath = os.path.join(rawdata, filename)
            data, head = fits.getdata(filepath, header=True)

            h, w = data.shape
            x1, x2 = w-head['COVER'], w

            # find the overscan level along the y-axis
            ovr_lst1 = data[0:h//2,x1+2:x2].mean(dtype=np.float64, axis=1)
            ovr_lst2 = data[h//2:h,x1+2:x2].mean(dtype=np.float64, axis=1)

            ovr_lst1_fix = fix_cr(ovr_lst1)
            ovr_lst2_fix = fix_cr(ovr_lst2)

            # apply the sav-gol fitler to the mean of overscan
            ovrsmooth1 = savgol_filter(ovr_lst1_fix, window_length=301, polyorder=3)
            ovrsmooth2 = savgol_filter(ovr_lst2_fix, window_length=301, polyorder=3)

            # plot the overscan regions
            if i%5 == 0:
                fig = plt.figure(figsize=(10,6), dpi=150)

            ax1 = fig.add_axes([0.08, 0.83-(i%5)*0.185, 0.42, 0.15])
            ax2 = fig.add_axes([0.55, 0.83-(i%5)*0.185, 0.42, 0.15])

            ax1.plot([0,0],[ovr_lst1_fix.min(), ovr_lst1_fix.max()], 'w-', alpha=0)
            _y1, _y2 = ax1.get_ylim()
            ax1.plot(np.arange(0, h//2), ovr_lst1, 'r-', alpha=0.3)
            ax1.set_ylim(_y1, _y2)

            ax2.plot([0,0],[ovr_lst2_fix.min(), ovr_lst2_fix.max()], 'w-', alpha=0)
            _y1, _y2 = ax2.get_ylim()
            ax2.plot(np.arange(h//2, h), ovr_lst2, 'b-', alpha=0.3)
            ax2.set_ylim(_y1, _y2)

            ax1.plot(np.arange(0, h//2), ovrsmooth1, 'm', ls='-')
            ax2.plot(np.arange(h//2, h), ovrsmooth2, 'c', ls='-')
            ax1.set_ylabel('ADU')
            ax2.set_ylabel('')
            ax1.set_xlim(0, h//2-1)
            ax2.set_xlim(h//2, h-1)
            for ax in [ax1, ax2]:
                _x1, _x2 = ax.get_xlim()
                _y1, _y2 = ax.get_ylim()
                _x = 0.95*_x1 + 0.05*_x2
                _y = 0.20*_y1 + 0.80*_y2
                ax.text(_x, _y, item.fileid, fontsize=9)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.xaxis.set_major_locator(tck.MultipleLocator(500))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            if i%5==4 or i==len(item_lst)-1:
                ax1.set_xlabel('Y (pixel)')
                ax2.set_xlabel('Y (pixel)')
                figname = 'overscan_%02d.png'%(i//5+1)
                figfile = os.path.join(report, figname)
                fig.savefig(figfile)
                logger.info('Save image: %s'%figfile)
                plt.close(fig)

            # determine shape of output image (also the shape of science region)
            y1 = head['CRVAL2']
            y2 = y1 + head['NAXIS2'] - head['ROVER']
            ymid = (y1 + y2)//2
            x1 = head['CRVAL1']
            x2 = x1 + head['NAXIS1'] - head['COVER']
            newshape = (y2-y1, x2-x1)

            # find the saturation mask
            mask_sat = (data[y1:y2,x1:x2]>=saturation_adu)
            # get bad pixel mask
            bins = (head['RBIN'], head['CBIN'])
            mask_bad = self._get_badpixel_mask(newshape, bins=bins)

            mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2
            # save the mask
            mask_table = array_to_table(mask)
            maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
            maskpath = os.path.join(midproc, maskname)
            fits.writeto(maskpath, mask_table, overwrite=True)

            # subtract overscan
            new_data = np.zeros(newshape, dtype=np.float64)
            ovrdata1 = np.transpose(np.repeat([ovrsmooth1],x2-x1,axis=0))
            ovrdata2 = np.transpose(np.repeat([ovrsmooth2],x2-x1,axis=0))
            new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
            new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2

            # fix bad pixels
            new_data = fix_pixels(new_data, mask_bad, 'x', 'linear')

            # update fits header
            head['HIERARCH GAMSE OVERSCAN']        = True
            head['HIERARCH GAMSE OVERSCAN METHOD'] = 'smooth'

            # save data
            outname = '%s%s.fits'%(item.fileid, self.output_suffix)
            outpath = os.path.join(midproc, outname)
            fits.writeto(outpath, new_data, head, overwrite=True)
            print('Correct Overscan {} -> {}'.format(filename, outname))


        logger.info('Overscan corrected. Change suffix: %s -> %s'%
                    (self.input_suffix, self.output_suffix))
        self.input_suffix = self.output_suffix

    def _get_badpixel_mask(self, shape, bins):
        '''Get bad-pixel mask.


        Args:
            shape (tuple): Shape of the science data region.
            bins (tuple): Number of pixel bins of (ROW, COLUMN).
        Returns:
            :class:`numpy.array`: Binary mask indicating the bad pixels. The
                shape of the mask is the same as the input shape.

        The bad pixels are found when readout mode = Left Top & Bottom.

        '''
        mask = np.zeros(shape, dtype=np.bool)
        if bins == (1, 1) and shape == (4136, 4096):
            h, w = shape

            mask[349:352, 627:630] = True
            mask[349:h//2, 628]    = True

            mask[1604:h//2, 2452] = True

            mask[280:284,3701]   = True
            mask[274:h//2, 3702] = True
            mask[272:h//2, 3703] = True
            mask[274:282, 3704]  = True

            mask[1720:1722, 3532:3535] = True
            mask[1720, 3535]           = True
            mask[1722, 3532]           = True
            mask[1720:h//2,3533]       = True

            mask[347:349, 4082:4084] = True
            mask[347:h//2,4083]      = True

            mask[h//2:2631, 1909] = True
        else:
            print('No bad pixel information for this CCD size.')
            raise ValueError
        return mask



    def bias(self):
        '''Bias corrrection for Xinglong 2.16m Telescope HRS.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **skip**,        *bool*,  Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,      *str*,   Suffix of the corrected files.
           **cosmic_clip**, *float*, Upper clipping threshold to remove cosmic-rays.
           **file**,        *str*,   Name of bias file.

        '''
        self.output_suffix = self.config.get('bias', 'suffix')

        if self.config.getboolean('bias', 'skip'):
            logger.info('Skip [bias] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        midproc = self.paths['midproc']

        bias_id_lst = self.find_bias()

        if len(bias_id_lst) == 0:
            # no bias frame found. quit this method.
            # update suffix
            logger.info('No bias found.')
            return True

        infile_lst = [os.path.join(midproc,
                        '%s%s.fits'%(item.fileid, self.input_suffix))
                        for item in self.log if item.frameid in bias_id_lst]

        # import and stack all bias files in a data cube
        tmp = [fits.getdata(filename, header=True) for filename in infile_lst]
        all_data, all_head = list(zip(*tmp))
        all_data = np.array(all_data)

        if self.config.has_option('bias', 'cosmic_clip'):
            # use sigma-clipping method to mask cosmic rays
            cosmic_clip = self.config.getfloat('bias', 'cosmic_clip')

            all_mask = np.ones_like(all_data, dtype=np.bool)

            mask = (all_data == all_data.max(axis=0))
        
            niter = 0
            maxiter = 2
            while niter <= maxiter:
                niter += 1
                mdata = np.ma.masked_array(all_data, mask=mask)
                # calculate mean and standard deviation
                mean = mdata.mean(axis=0, dtype=np.float64).data
                std  = mdata.std(axis=0, dtype=np.float64, ddof=1).data
                # initialize new mask
                new_mask = np.ones_like(mask)>0
                # masking all the upper outliers.
                for i in np.arange(all_data.shape[0]):
                    new_mask[i,:,:] = all_data[i,:,:] > mean + cosmic_clip*std
                mask = new_mask

            mdata = np.ma.masked_array(all_data, mask=mask)
            bias = mdata.mean(axis=0, dtype=np.float64).data
        else:
            # no sigma clipping
            bias = alldata.mean(axis=0, dtype=np.float64)

        # create new FITS Header for bias
        head = fits.Header()
        head['HIERARCH GAMSE BIAS NFILE'] = len(bias_id_lst)

        # get final bias filename from the config file
        bias_file = self.config.get('bias', 'file')

        if self.config.has_option('bias', 'smooth_method'):
            # perform smoothing for bias
            smooth_method = self.config.get('bias', 'smooth_method')
            smooth_method = smooth_method.strip().lower()

            logger.info('Smoothing bias: %s'%smooth_method)

            if smooth_method in ['gauss','gaussian']:
                # perform 2D gaussian smoothing

                smooth_sigma = self.config.getint('bias', 'smooth_sigma')
                smooth_mode  = self.config.get('bias', 'smooth_mode')

                logger.info('Smoothing bias: sigma = %f'%smooth_sigma)
                logger.info('Smoothing bias: mode = %s'%smooth_mode)

                h, w = bias.shape
                bias_smooth = np.zeros((h, w), dtype=np.float64)
                bias_smooth[0:h/2,:] = gaussian_filter(bias[0:h/2,:],
                                                       smooth_sigma,
                                                       mode=smooth_mode)
                bias_smooth[h/2:h,:] = gaussian_filter(bias[h/2:h,:],
                                                       smooth_sigma,
                                                       mode=smooth_mode)

                logger.info('Smoothing bias: Update bias FITS header')

                head['HIERARCH GAMSE BIAS SMOOTH']        = True
                head['HIERARCH GAMSE BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                head['HIERARCH GAMSE BIAS SMOOTH SIGMA']  = smooth_sigma
                head['HIERARCH GAMSE BIAS SMOOTH MODE']   = smooth_mode

            else:
                pass

            # bias_data is a proxy for bias to be corrected for each frame
            bias_data = bias_smooth

            # plot comparison between un-smoothed and smoothed data
            self.plot_bias_smooth(bias, bias_smooth)

        else:
            # no smoothing
            logger.info('No smoothing parameter for bias. Skip bias smoothing')
            head['HIERARCH GAMSE BIAS SMOOTH'] = False
            bias_data = bias

        # save the bias to FITS
        fits.writeto(bias_file, bias_data, head, overwrite=True)
        
        self.plot_bias_variation(all_data, all_head, time_key='DATE-STA')

        # finally all files are corrected for the bias
        for item in self.log:
            if item.frameid not in bias_id_lst:
                infile  = '%s%s.fits'%(item.fileid, self.input_suffix)
                outfile = '%s%s.fits'%(item.fileid, self.output_suffix)
                inpath  = os.path.join(midproc, infile)
                outpath = os.path.join(midproc, outfile)
                data, head = fits.getdata(inpath, header=True)
                data_new = data - bias_data
                # write information into FITS header
                head['HIERARCH GAMSE BIAS'] = True
                # save the bias corrected data
                fits.writeto(outpath, data_new, head, overwrite=True)
                info = ['Correct bias for item no. %d.'%item.frameid,
                        'Save bias corrected file: "%s"'%outpath]
                logger.info((os.linesep+'  ').join(info))
                print('Correct bias: {} => {}'.format(infile, outfile))

        # update suffix
        logger.info('Bias corrected. Change suffix: %s -> %s'%
                    (self.input_suffix, self.output_suffix))
        self.input_suffix = self.output_suffix
        return True

all_columns = [
        ('frameid', 'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',  'str',   '{:^12s}', '{0[fileid]:12s}'),
        ('imgtype', 'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',  'str',   '{:^12s}', '{0[object]:12s}'),
        ('i2cell',  'bool',  '{:^6s}',  '{0[i2cell]!s: <6}'),
        ('exptime', 'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate', 'time',  '{:^23s}', '{0[obsdate]:}'),
        ('nsat',    'int',   '{:^10s}', '{0[nsat]:10d}'),
        ('q95',     'int',   '{:^10s}', '{0[q95]:10d}'),
        ]

def print_wrapper(string, item):
    """A wrapper for log printing for Xinglong216HRS pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype = item['imgtype']
    obj     = item['object']

    if len(obj)>=4 and obj[0:4].lower()=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif len(obj)>=4 and obj[0:4].lower()=='thar':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

