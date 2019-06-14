import os
import re
import shutil
import datetime
import logging
logger = logging.getLogger(__name__)
import configparser
import dateutil.parser

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time  import Time
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates  as mdates

from ..echelle.imageproc import combine_images, array_to_table
from ..echelle.trace import find_apertures, load_aperture_set, TraceFigureCommon
from ..echelle.flat import get_fiber_flat, mosaic_flat_auto, mosaic_images
from ..echelle.extract import extract_aperset
from ..echelle.wlcalib import (wlcalib, recalib, select_calib_from_database,
                               self_reference_singlefiber,
                               wl_reference_singlefiber, get_time_weight,
                               find_caliblamp_offset)
from ..echelle.background import find_background, simple_debackground
from ..utils.onedarray import get_local_minima
from ..utils.regression import iterative_polyfit
from ..utils.obslog import read_obslog
from .common import plot_background_aspect1, FormattedInfo
from .reduction          import Reduction

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
            * **head** (:class:`astropy.io.fits.Header`) – Updated FITS header.
            * **overmean** (*float*) – Mean value of overscan pixels.
    """
    h, w = data.shape
    overdata1 = data[:, 0:20]
    overdata2 = data[:, w-18:w]
    overdata_tot = np.hstack((overdata1, overdata2))

    # find the overscan level along the y axis
    # 1 for the left region, 2 for the right region
    # calculate the mean of overscan region along x direction
    ovr_lst1 = overdata1.mean(dtype=np.float64, axis=1)
    ovr_lst2 = overdata2.mean(dtype=np.float64, axis=1)
    
    # only used the upper ~1/2 regions for calculating mean of overscan
    #vy1, vy2 = h//2, h
    vy1, vy2 = 0, h//2
    # find the mean and standard deviation for left & right overscan
    ovrmean1 = ovr_lst1[vy1:vy2].mean(dtype=np.float64)
    ovrmean2 = ovr_lst2[vy1:vy2].mean(dtype=np.float64)
    ovrstd1  = ovr_lst1[vy1:vy2].std(dtype=np.float64, ddof=1)
    ovrstd2  = ovr_lst2[vy1:vy2].std(dtype=np.float64, ddof=1)

    # subtract overscan
    new_data = data[:,20:2068] - ovrmean1
    
    # update fits header
    # head['BLANK'] is only valid for integer arrays.
    if 'BLANK' in head:
        del head['BLANK']

    head['HIERARCH GAMSE OVERSCAN']        = True
    head['HIERARCH GAMSE OVERSCAN METHOD'] = 'mean'
    head['HIERARCH GAMSE OVERSCAN AXIS-1'] = '1:20'
    head['HIERARCH GAMSE OVERSCAN AXIS-2'] = '%d:%d'%(vy1,vy2)
    head['HIERARCH GAMSE OVERSCAN MEAN']   = ovrmean1
    head['HIERARCH GAMSE OVERSCAN STDEV']  = ovrstd1

    return new_data, head, ovrmean1

def get_mask(data, head):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.
    """
    # saturated CCD count
    saturation_adu = 63000

    mask_sat = (data[:, 20:-20] >= saturation_adu)

    mask_bad = np.zeros_like(data[:, 20:-20], dtype=np.int16)
    # currently no bad pixels in FOCES CCD

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

class FOCES(Reduction):
    """Reduction pipleline for FOCES.
    """

    def __init__(self):
        super(FOCES, self).__init__(instrument='FOCES')

    def overscan(self):
        '''
        Correct overscan for FOCES FITS images.
        
        The overscan is corrected for each FITS image listed in the observing
        log.
        FOCES images has two overscan regions, lying in the left and right
        sides of the CCD, respectively.
        The ADUs in the right side are always higher than those in the left
        side.
        Therefore, only the one on the left region is used to correct the
        overscan level.
        The mean values along the *y*-axies are calculated and smoothed.
        Then are subtracted for every pixel in the science region.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **skip**,    *bool*,   Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,  *string*, Suffix of the corrected files.
           **plot**,    *bool*,   Plot the overscan levels if *yes*.
           **var_fig**, *string*, Filename of the overscan variation figure.

        '''

        # find output suffix for fits
        self.output_suffix = self.config.get('overscan', 'suffix')

        if self.config.getboolean('overscan', 'skip'):
            logger.info('Skip [overscan] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        self.report_file.write('    <h2>Overscan</h2>'+os.linesep)
        t_lst, frameid_lst, fileid_lst = [], [], []
        ovr1_lst, ovr1_std_lst = [], []
        ovr2_lst, ovr2_std_lst = [], []
        
        # saturated CCD count
        saturation_adu = 63000
    
        # path alias
        midproc = self.paths['midproc']
        rawdata = self.paths['rawdata']
        report  = self.paths['report']

        # loop over all files to correct for the overscan
        # prepare the item list
        item_lst = [item for item in self.log]

        for i, item in enumerate(item_lst):
            logger.info('Correct overscan for item %3d: "%s"'%(
                         item.frameid, item.fileid))

            # read in of the data
            filename = '%s%s.fits'%(item.fileid, self.input_suffix)
            filepath = os.path.join(rawdata, filename)
            data, head = fits.getdata(filepath, header=True)
    
            h, w = data.shape
            overdata1 = data[:,0:20]
            overdata2 = data[:,w-15:w]
            overdata_tot = np.hstack((overdata1, overdata2))
    
            # find the overscan level along the y axis
            # 1 for the left region, 2 for the right region
            # calculate the mean of overscan region along x direction
            ovr_lst1 = overdata1.mean(dtype=np.float64, axis=1)
            ovr_lst2 = overdata2.mean(dtype=np.float64, axis=1)
            # only used the upper ~1/2 regions for culculating mean of overscan
            vy1, vy2 = h//2, h
            # find the mean and standard deviation for left & right overscan
            ovrmean1 = ovr_lst1[vy1:vy2].mean(dtype=np.float64)
            ovrmean2 = ovr_lst2[vy1:vy2].mean(dtype=np.float64)
            ovrstd1  = ovr_lst1[vy1:vy2].std(dtype=np.float64, ddof=1)
            ovrstd2  = ovr_lst2[vy1:vy2].std(dtype=np.float64, ddof=1)
    
            # plot the overscan regions
            if i%5 == 0:
                fig = plt.figure(figsize=(8,6), dpi=150)

            ax1 = fig.add_axes([0.10, 0.83-(i%5)*0.185, 0.42, 0.15])
            ax2 = fig.add_axes([0.55, 0.83-(i%5)*0.185, 0.42, 0.15])
            ax1.plot(ovr_lst1, 'r-', alpha=0.3, lw=0.5)
            ax2.plot(ovr_lst2, 'b-', alpha=0.3, lw=0.5)
            y = np.arange(vy1, vy2)
            ax1.plot(y, ovr_lst1[vy1:vy2], 'r-', alpha=0.7, lw=0.5)
            ax2.plot(y, ovr_lst2[vy1:vy2], 'b-', alpha=0.7, lw=0.5)
            _x1, _x2 = 0, ovr_lst1.size-1
            ax1.plot([_x1,_x2], [ovrmean1,         ovrmean1],         'm-')
            ax1.plot([_x1,_x2], [ovrmean1-ovrstd1, ovrmean1-ovrstd1], 'm:')
            ax1.plot([_x1,_x2], [ovrmean1+ovrstd1, ovrmean1+ovrstd1], 'm:')
            ax2.plot([_x1,_x2], [ovrmean2,         ovrmean2],         'c-')
            ax2.plot([_x1,_x2], [ovrmean2-ovrstd2, ovrmean2-ovrstd2], 'c:')
            ax2.plot([_x1,_x2], [ovrmean2+ovrstd2, ovrmean2+ovrstd2], 'c:')
            ax1.set_ylabel('ADU')
            ax2.set_ylabel('')
            _y11, _y12 = ax1.get_ylim()
            _y21, _y22 = ax2.get_ylim()
            _y1 = min(_y11, _y21)
            _y2 = max(_y12, _y22)
            _x = 0.95*_x1 + 0.05*_x2
            _y = 0.20*_y1 + 0.80*_y2
            _text = '%s (%s)'%(item.fileid, item.objectname)
            ax1.text(_x, _y, _text, fontsize=9)
            for ax in [ax1, ax2]:
                ax.set_xlim(_x1, _x2)
                ax.set_ylim(_y1, _y2)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
            ax2.set_yticklabels([])
            if i%5==4 or i==len(item_lst)-1:
                ax1.set_xlabel('Y (pixel)',fontsize=9)
                ax2.set_xlabel('Y (pixel)',fontsize=9)
                figname = 'overscan_%02d.png'%(i//5+1)
                figpath = os.path.join(report, figname)
                fig.savefig(figpath)
                logger.info('Save image: %s'%figpath)
                plt.close(fig)
                self.report_file.write(
                    '        <img src="%s">'%figname+os.linesep
                    )
    
            # find saturated pixels and saved them in FITS files
            mask_sat   = (data[:,20:2068]>=saturation_adu)
            mask       = np.int16(mask_sat)*4
            mask_table = array_to_table(mask)
            maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
            maskpath = os.path.join(midproc, maskname)
            # save the mask.
            fits.writeto(maskpath, mask_table, overwrite=True)
    
            # subtract overscan
            new_data = data[:,20:2068] - ovrmean1
    
            # update fits header
            # head['BLANK'] is only valid for integer arrays.
            if 'BLANK' in head:
                del head['BLANK']
            head['HIERARCH GAMSE OVERSCAN']        = True
            head['HIERARCH GAMSE OVERSCAN METHOD'] = 'mean'
            head['HIERARCH GAMSE OVERSCAN AXIS-1'] = '1:20'
            head['HIERARCH GAMSE OVERSCAN AXIS-2'] = '%d:%d'%(vy1,vy2)
            head['HIERARCH GAMSE OVERSCAN MEAN']   = ovrmean1
            head['HIERARCH GAMSE OVERSCAN STDEV']  = ovrstd1
    
            # save data
            outname = '%s%s.fits'%(item.fileid, self.output_suffix)
            outpath = os.path.join(midproc, outname)
            fits.writeto(outpath, new_data, head, overwrite=True)
            print('Correct Overscan {} -> {}'.format(filename, outname))
    
            # quality check of the mean value of the overscan with time
            # therefore the time and the mean overscan values are stored in 
            # a list to be analyzed later
            #t_lst.append(head['UTC-STA'])
            t_lst.append(head['FRAME'])
            frameid_lst.append(item.frameid)
            fileid_lst.append(item.fileid)
            ovr1_lst.append(ovrmean1)
            ovr2_lst.append(ovrmean2)
            ovr1_std_lst.append(ovrstd1)
            ovr2_std_lst.append(ovrstd2)

    
        info_lst = ['Overscan summary:',
                    ', '.join(['frameid','fileid','UTC-STA',
                           'left_overscan','right_overscan','total_overscan']),
                   ]
        for i in range(len(t_lst)):
            message = ' '.join([
                '%4d'%frameid_lst[i],
                '%20s'%fileid_lst[i],
                '%s'%t_lst[i],
                u'%7.2f \xb1 %5.2f'%(ovr1_lst[i],ovr1_std_lst[i]),
                u'%7.2f \xb1 %5.2f'%(ovr2_lst[i],ovr2_std_lst[i]),
                #u'%7.2f \xb1 %5.2f'%(ovr_lst[i],ovr_std_lst[i]),
                ])
            info_lst.append(message)
        separator = os.linesep + '  '
        logger.info(separator.join(info_lst))

        # plot overscan variation
        figfile = os.path.join(report, 'overscan_variation.png')
        plot_overscan_variation(t_lst, ovr1_lst, figfile)
        logger.info('Save the variation of overscan figure: "%s"'%figfile)

        logger.info('Overscan corrected. Change suffix: %s -> %s'%
                    (self.input_suffix, self.output_suffix))
        self.input_suffix = self.output_suffix


    def bias(self):
        """Bias correction.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 60

           **skip**,          *bool*,    Skip this step if *yes* and **mode** = *'debug'*
           **suffix**,        *string*,  Suffix of the corrected files.
           **cosmic_clip**,   *float*,   Upper clipping threshold to remove cosmic-rays.
           **smooth_method**, *string*,  "Method of smoothing, including *gaussian*."
           **smooth_sigma**,  *integer*, Sigma of the smoothing filter.
           **smooth_mode**,   *string*,  Mode of the smoothing.
           **file**,          *string*,  Name of bias file.

        To calculate the correct bias level for every individual pixel position
        several individual steps are performed. In the beginning a datacube
        containing all the individual bias frames is created. The group of bias
        values at a given position (e.g. pixel position (230, 425)) is called
        'stack'. All the following steps are performed for every stack
        individually. 10 bias frames are recommended in each observational run.
        
        #. masking cosmics and outlier
            #. masking the cosmics  => mask the highest value in each stack
            #. masking the outliers => mask every value which is more than 10
               std away from the mean of its stack (kappa-sigma-clipping)
        #. calculating the mean bias values for each stack and create a
           mean-bias-file
        #. smooth the mean-bias-file with a 2d gaussian filter
        #. final_bias = mean_bias - smooth_bias
          
        """
        # find output suffix for fits
        self.output_suffix = self.config.get('bias', 'suffix')

        if self.config.getboolean('bias', 'skip'):
            logger.info('Skip [bias] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        self.report_file.write('    <h2>Bias Correction</h2>'+os.linesep)

        # path alias
        midproc = self.paths['midproc']
        report  = self.paths['report']

        bias_id_lst = self._find_bias()

        infile_lst = [os.path.join(midproc,
                        '%s%s.fits'%(item.fileid, self.input_suffix))
                        for item in self.log if item.frameid in bias_id_lst]
        
        # import and stack all bias files in a data cube
        tmp = [fits.getdata(filename, header=True) for filename in infile_lst]
        all_data, all_head = zip(*tmp)
        all_data = np.array(all_data)

        if self.config.has_option('bias', 'cosmic_clip'):
            # use sigma-clipping method to mask cosmic rays
            cosmic_clip = self.config.getfloat('bias', 'cosmic_clip')

            # a data cube with the same dimensions as the bias data cube but
            # entierly filled with 'True' is created
            all_mask = np.ones_like(all_data, dtype=np.bool)
        
            # mask cosmics  => mask the highest value in each stack.
            # this method not only the cosmics but in most of the cases 'normal'
            # bias values are masked. Since the bias values are so close
            # together excluding one value even if its a meaningfull value
            # does not change the final result (this is also an additional
            # reason why 10 bias frames are recommended)

            # mask the maximum layer in each pixel in the data cube
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

            # calculate the mean bias values for each stack and create a 
            # mean-bias-file.
            # determine the mean-bias frame by calculating the mean of each
            # stack taking into account the masked areas/values
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
            # smooth 2D bias
            smooth_method = self.config.get('bias', 'smooth_method')
            smooth_method = smooth_method.strip().lower()

            logger.info('Smoothing bias: %s'%smooth_method)

            if smooth_method.lower().strip() == 'gaussian':
                # 2D gaussian smoothing

                smooth_sigma = self.config.getint('bias', 'smooth_sigma')
                smooth_mode  = self.config.get('bias',    'smooth_mode')

                logger.info('Smoothing bias: sigma = %f'%smooth_sigma)
                logger.info('Smoothing bias: mode = %s'%smooth_mode)

                bias_smooth = gaussian_filter(bias, smooth_sigma, mode=smooth_mode)

                logger.info('Smoothing bias: Update bias FITS header')

                head['HIERARCH GAMSE BIAS SMOOTH']        = True
                head['HIERARCH GAMSE BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                head['HIERARCH GAMSE BIAS SMOOTH SIGMA']  = smooth_sigma
                head['HIERARCH GAMSE BIAS SMOOTH MODE']   = smooth_mode

            else:
                print('Unknown smoothing method: %s'%smooth_method)
                pass

            # bias_data is a proxy for bias to be corrected for each frame
            bias_data = bias_smooth

            # plot comparison between un-smoothed and smoothed data
            comp_figfile = os.path.join(report, 'bias_smooth.png')
            hist_figfile = os.path.join(report, 'bias_smooth_hist.png')
            plot_bias_smooth(bias, bias_smooth, comp_figfile, hist_figfile)
            self.report_file.write(
                ' '*8 + '<img src="%s" alt="smoothed bias">'%comp_figfile +
                os.linesep)
            logger.info('Plot smoothed bias in figure: "%s"'%comp_figfile)
            self.report_file.write(
                ' '*8 + '<img src="%s"'%hist_figfile +
                'alt="histogram of smoothed bias">' +
                os.linesep)
            logger.info('Plot histograms of smoothed bias in figure: "%s"'%hist_figfile)

        else:
            # no smoothing
            logger.info('No smoothing parameter for bias. Skip bias smoothing')
            head['HIERARCH GAMSE BIAS SMOOTH'] = False
            bias_data = bias
        
        # save the bias to FITS
        fits.writeto(bias_file, bias_data, head, overwrite=True)
        
        self.plot_bias_variation(all_data, all_head)

        # finally all files are corrected for the bias
        for item in self.log:
            if item.frameid in bias_id_lst:
                continue
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
   
    def plot_bias_variation(self, data_lst, head_lst, time_key='FRAME'):
        """Plot the variation of bias level with time.
        A figure will be generated in the report directory of the reduction. The
        name of the figure is given in the config file.
        """
    
        mean_lst, std_lst = [], []
        time_lst = []
        center_lst = []
        for ifile, (data, head) in enumerate(zip(data_lst, head_lst)):
            mean = np.zeros((3,3), dtype=np.float64)
            std  = np.zeros((3,3), dtype=np.float64)
            for y, x in [(y, x) for y in range(3) for x in range(3)]:
                yc = y*500 + 500
                xc = x*500 + 500
                smalldata = data[yc-100:yc+100,xc-100:xc+100]
                mean[y,x] = smalldata.mean(dtype=np.float64)
                std[y,x]  = smalldata.std(dtype=np.float64, ddof=1)
                if ifile == 0:
                    center_lst.append((xc,yc))
            mean_lst.append(mean)
            std_lst.append(std)
            time_lst.append(head[time_key])
        mean_lst = np.array(mean_lst)
        std_lst  = np.array(std_lst)
    
        # convert time string to delta minutes relative to the first image
        date_lst = [dateutil.parser.parse(t) for t in time_lst]
        datenums = mdates.date2num(date_lst)
        minutes = [(d - datenums[0])*24.*60. for d in datenums]
    
        # write the bias levels into run log
        message = ['Variation of bias level with time:',
                   'time, delta_minutes, mean values']
        for k in range(len(time_lst)):
            info = ['%s'%time_lst[k], '%7.3f'%minutes[k]]
            for j, i in [(j, i) for j in range(3) for i in range(3)]:
                info.append('%7.2f'%mean_lst[k,j,i])
            message.append(' '.join(info))
        logger.info((os.linesep+'  ').join(message))
    
        # create figure
        fig = plt.figure(figsize=(8,6), dpi=150)
        z1, z2 = 999, -999
        for j, i in [(j, i) for j in range(3) for i in range(3)]:
            ax = fig.add_axes([0.1+i*0.3, 0.7-j*0.3, 0.26, 0.26])
            #ax.plot(minutes, mean_lst[:,i],'o-',alpha=0.7)
            #if i <= 5:
            #    ax.set_xticklabels([])
            ax.plot(mean_lst[:,j,i],'o-',alpha=0.7)
            y1,y2 = ax.get_ylim()
            # searching for the minumum and maximum y display ranges
            z1 = min(y1, z1)
            z2 = max(y2, z2)
        for j, i in [(j, i) for j in range(3) for i in range(3)]:
            k = j*3 + i
            ax = fig.get_axes()[k]
            ax.set_ylim(z1,z2)
            x1, x2 = ax.get_xlim()
            ax.text(0.7*x1+0.3*x2, 0.2*z1+0.8*z2,
                    'x,y=(%4d,%4d)'%(center_lst[k][0],center_lst[k][1]),
                    fontsize=9)
            if j == 2:
                #ax.set_xlabel('Time (min)')
                ax.set_xlabel('Frame',fontsize=11)
            if i == 0:
                ax.set_ylabel('Mean ADU',fontsize=11)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(10)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(10)

        # save the figure
        figfile = os.path.join(self.paths['report'], 'bias_variation.png')
        self.report_file.write('        <img src="%s">'%figfile
            +os.linesep)
        fig.savefig(figfile)
        logger.info('Plot variation of bias with time in figure: "%s"'%figfile)
        plt.close(fig)

all_columns = [
        ('frameid', 'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',  'str',   '{:^40s}', '{0[fileid]:40s}'),
        ('imgtype', 'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',  'str',   '{:^12s}', '{0[object]:12s}'),
        ('exptime', 'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate', 'time',  '{:^23s}', '{0[obsdate]:}'),
        ('nsat',    'int',   '{:^7s}',  '{0[nsat]:7d}'),
        ('q95',     'int',   '{:^6s}',  '{0[q95]:6d}'),
        ]

def print_wrapper(string, item):
    """A wrapper for log printing for FOCES pipeline.

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

def make_obslog(path):
    """Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (str): Path to the raw FITS files.

    """
    
    # standard naming convenction for fileid
    name_pattern = '^\d{8}_\d{4}_FOC\d{4}_\w{3}\d$'

    fname_lst = sorted(os.listdir(path))

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid',  'S40'),  ('imgtype', 'S3'),
        ('object',  'S12'), ('exptime', 'f4'),
        ('obsdate', Time),  ('nsat',    'i4'),   ('q95',     'i4'),
        ])

    # prepare infomation to print
    pinfo = FormattedInfo(all_columns,
            ['frameid', 'fileid', 'imgtype', 'object', 'exptime', 'obsdate',
             'nsat', 'q95'])

    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    #print(pinfo.get_dtype())
    print(pinfo.get_separator())

    # start scanning the raw files
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid = fname[0:-5]
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        # old FOCES data are 3-dimensional arrays
        if data.ndim == 3: scidata = data[0, 20:-20]
        else:              scidata = data[:,20:-20]
            
        obsdate = Time(head['FRAME'])
        exptime = head['EXPOSURE']

        if re.match(name_pattern, fileid) is not None:
            # fileid matches the standard FOCES naming convention
            if fileid[22:25]=='BIA':
                imgtype, objectname = 'cal', 'Bias'
            elif fileid[22:25]=='FLA':
                imgtype, objectname = 'cal', 'Flat'
            elif fileid[22:25]=='THA':
                imgtype, objectname = 'cal', 'ThAr'
            else:
                objectname = 'Unknown'
                imgtype = ('cal', 'sci')[fileid[22:25]=='SCI']
        else:
            # fileid does not follow the naming convetion
            imgtype, objectname = 'cal', ''

        # determine the total number of saturated pixels
        saturation = (data>=63000).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        item = [0, fileid, imgtype, objectname, exptime, obsdate, saturation,
                quantile95]
        logtable.add_row(item)
        # get table Row object. (not elegant!)
        item = logtable[-1]

        # print log item with colors
        string = pinfo.get_format(has_esc=False).format(item)
        print(print_wrapper(string, item))
    print(pinfo.get_separator())

    logtable.sort('obsdate')

    # allocate frameid
    prev_frameid = -1
    for item in logtable:

        if re.match(name_pattern, item['fileid']) is None:
            # fileid follows the standard name convention of FOCES
            frameid = prev_frameid + 1
        else:
            # doesn't follow
            frameid = int(item['fileid'].split('_')[1])

        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(frameid,
                prev_frameid))

        item['frameid'] = frameid

        prev_frameid = frameid

    # determine filename of logtable.
    # use the obsdate of the second frame. Here assume total number of files>2
    obsdate = logtable[1]['obsdate'].iso[0:10]
    outname = '{}.obslog'.format(obsdate)
    if os.path.exists(outname):
        i = 0
        while(True):
            i += 1
            outname = '{}.{}.obslog'.format(obsdate, i)
            if not os.path.exists(outname):
                outfilename = outname
                break
    else:
        outfilename = outname

    # save the logtable
    loginfo = FormattedInfo(all_columns)
    outfile = open(outfilename, 'w')
    outfile.write(loginfo.get_title()+os.linesep)
    outfile.write(loginfo.get_dtype()+os.linesep)
    outfile.write(loginfo.get_separator()+os.linesep)
    for row in logtable:
        outfile.write(loginfo.get_format(has_esc=False).format(row)+os.linesep)
    outfile.close()


def get_primary_header(input_lst):
    """Return a list of header records with length of 80 characters.
    The order and datatypes of the records follow the FOCES FITS standards.

    Args:
        input_lst (tuple): A tuple containing the keywords and their values

    Returns:
        *list*: A list containing the records

    """
    lst = [
        # 12345678    12345678901234567890123456789012345678901234567
        ('SIMPLE'  , 'file does conform to FITS standard'             ),
        ('BITPIX'  , 'number of bits per data pixel'                  ),
        ('NAXIS'   , 'number of data axes'                            ),
        ('NAXIS1'  , 'length of data axis 1'                          ),
        ('NAXIS2'  , 'length of data axis 2'                          ),
        ('BSCALE'  , 'factor to linearly scale the data pixel values' ),
        ('BZERO'   , 'offset to linearly scale the data pixel values' ),
        ('BUNIT'   , 'physical unit of the data pixel values'         ),
        ('BLANK'   , 'value representing undefined physical values'   ),
        ('DISPAXIS', 'main dispersion axis of the spectral data'      ),
        ('DATATYPE', 'type of data (calibration/science)'             ),
        ('OBJECT'  , 'object observed'                                ),
        ('DATE-OBS', 'start date of observation run'                  ),
        ('MJD-OBS' , 'Modified Julian Date of observation run'        ),
        ('TIMESYS' , 'time system'                                    ),
        ('FRAMEID' , 'frame ID in observation run'                    ),
        ('RA'      , 'right ascension of object'                      ),
        ('DEC'     , 'declination of object'                          ),
        ('RADESYS' , 'name of reference frame'                        ),
        ('EQUINOX' , 'epoch of the mean equator and equinox in years' ),
        ('EXPTIME' , 'exposure time in seconds'                       ),
        ('PHO-OFF' , 'offset of photon middle time'                   ),
        ('UTC-STA' , 'UTC at start of exposure'                       ),
        ('UTC-MID' , 'UTC at middle of exposure'                      ),
        ('UTC-PHO' , 'UTC at photon middle of exposure'               ),
        ('UTC-END' , 'UTC at end of exposure'                         ),
        ('LT-STA'  , 'local time at start of exposure'                ),
        ('LT-MID'  , 'local time at middle of exposure'               ),
        ('LT-PHO'  , 'local time at photon middle of exposure'        ),
        ('LT-END'  , 'local time at end of exposure'                  ),
        ('LST-STA' , 'local sidereal time at start'                   ),
        ('LST-MID' , 'local sidereal time at middle'                  ),
        ('LST-PHO' , 'local sidereal time at photon middle'           ),
        ('LST-END' , 'local sidereal time at end'                     ),
        ('MJD-STA' , 'Modified Julian Date of UTC-STA'                ),
        ('MJD-MID' , 'Modified Julian Date of UTC-MID'                ),
        ('MJD-PHO' , 'Modified Julian Date of UTC-PHO'                ),
        ('MJD-END' , 'Modified Julian Date of UTC-END'                ),
        ('AIRM-STA', 'airmass at start of exposure'                   ),
        ('AIRM-MID', 'airmass at middle of exposure'                  ),
        ('AIRM-PHO', 'airmass at photon middle of exposure'           ),
        ('AIRM-END', 'airmass at end of exposure'                     ),
        ('AIRMASS' , 'effective airmass during exposure'              ),
        ('ALT-STA' , 'telescope altitude at start'                    ),
        ('ALT-MID' , 'telescope altitude at middle'                   ),
        ('ALT-PHO' , 'telescope altitude at photon middle'            ),
        ('ALT-END' , 'telescope altitude at end'                      ),
        ('AZ-STA'  , 'telescope azimuth at start'                     ),
        ('AZ-MID'  , 'telescope azimuth at middle'                    ),
        ('AZ-PHO'  , 'telescope azimuth at photon middle'             ),
        ('AZ-END'  , 'telescope azimuth at end'                       ),
        ('MOON-AGE', 'days past new moon at middle of exposure'       ),
        ('MOON-ALT', 'moon altitude at middle of exposure'            ),
        ('MOON-AZ' , 'moon azimuth at middle of exposure'             ),
        ('MOON-DIS', 'angular distance to moon (in degree)'           ),
        ('TWI-END' , 'end time of astronomical twilight in UTC'       ),
        ('TWI-STA' , 'start time of astronomical twilight in UTC'     ),
        ('PROP-ID' , 'proposal ID'                                    ),
        ('PROP-TIT', 'title of proposal'                              ),
        ('PROP-PI' , 'principal investigator of proposal'             ),
        ('OBSERVER', 'people who acquire the data'                    ),
        ('OBSERVAT', 'observatory where the data is acquired'         ),
        ('TELESCOP', 'telescope used to acquire the data'             ),
        ('OBS-LONG', 'longitude of the telescope'                     ), 
        ('OBS-LAT' , 'latitude of the telescope'                      ),
        ('OBS-ALT' , 'altitude of the telescope in meter'             ),
        ('INSTRUME', 'instrument used to acquire the data'            ),
        ('SETUP-ID', 'ID of the instrument setup'                     ),
        ('SLT-WID' , 'slit width (in mm)'                             ),
        ('SLT-LEN' , 'slit length (in mm)'                            ),
        ('NCHANNEL', 'number of simultaneous channels'                ),
        ('CHANNEL1', 'object of channel 1'                            ),
        ('CHANNEL2', 'object of channel 2'                            ),
        ('FILTER1' , 'filter in channel 1'                            ),
        ('FILTER2' , 'filter in channel 2'                            ),
        ('EXPMETER', 'usage of exposure meter'                        ),
        ('SHAK_STA', 'status of fiber shaker (on/off)'                ),
        ('SHAK_FRE', 'frequency of fiber shaker (in Hz)'              ),
        ('SHAK_AMP', 'amplitude of fiber shaker'                      ),
        ('DETECTOR', 'detector used to acquire the data'              ),
        ('GAIN'    , 'readout gain of detector (in electron/ADU)'     ),
        ('RO-SPEED', 'read out speed of detector'                     ),
        ('RO-NOISE', 'read out noise of detector'                     ),
        ('BINAXIS1', 'binning factor of data axis 1'                  ),
        ('BINAXIS2', 'binning factor of data axis 2'                  ),
        ('TEMP-DET', 'temperature of detector (in degree)'            ),
        ('TEMP-BOX', 'temperature inside instrument box (in degree)'  ),
        ('TEMP-ROO', 'temperature inside instrument room (in degree)' ),
        ('PRES-BOX', 'pressure inside instrument box (in hPa)'        ),
        ('DATE'    , 'file creation date'                             ),
        ('ORI-NAME', 'original filename'                              ),
        ('ORIGIN'  , 'organization responsible for the FITS file'     ),
        ('HEADVER' , 'version of header'                              ),
        ]
    now = datetime.datetime.now()
    header_lst = []
    for key, comment in lst:
        if key in input_lst.keys():
            value = input_lst[key]
        else:
            value = None
        if type(value) == type('a'):
            value = "'%-8s'"%value
            value = value.ljust(20)
        elif type(value) == type(u'a'):
            value = value.encode('ascii','replace')
            value = "'%-8s'"%value
            value = value.ljust(20)
        elif type(value) == type(1):
            value = '%20d'%value
        elif type(value) == type(1.0):
            if key[0:4]=='MJD-':
                # for any keywords related to MJD, keep 6 decimal places.
                # for reference, 1 sec = 1.16e-5 days
                value = '%20.6f'%value
            else:
                value = str(value).rjust(20)
                value = value.replace('e','E')
        elif type(value) == type(now):
            # if value is a python datetime object
            value = "'%04d-%02d-%02dT%02d:%02d:%02d.%03d'"%(
                    value.year, value.month, value.day,
                    value.hour, value.minute, value.second,
                    int(round(value.microsecond*1e-3))
                    )
        elif value == True:
            value = 'T'.rjust(20)
        elif value == False:
            value = 'F'.rjust(20)
        elif value == None:
            value = "''".ljust(20)
        else:
            print('Unknown value: {}'.format(value))
        string = '%-8s= %s / %s'%(key,value,comment)
        if len(string)>=80:
            string = string[0:80]
        else:
            string = string.ljust(80)

        header_lst.append(string)

    return header_lst

def plot_overscan_variation(t_lst, overscan_lst, figfile):
    """Plot the variation of overscan.
    """
    
    # Quality check plot of the mean overscan value over time 
    fig = plt.figure(figsize=(8,6), dpi=150)
    ax2  = fig.add_axes([0.1,0.60,0.85,0.35])
    ax1  = fig.add_axes([0.1,0.15,0.85,0.35])
    #conversion of the DATE-string to a number
    date_lst = [dateutil.parser.parse(t) for t in t_lst]
    datenums = mdates.date2num(date_lst)

    ax1.plot_date(datenums, overscan_lst, 'r-', label='mean')
    ax2.plot(overscan_lst, 'r-', label='mean')
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right')
        leg.get_frame().set_alpha(0.1)
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Frame')
    ax1.set_ylabel('Overscan mean ADU')
    ax2.set_ylabel('Overscan mean ADU')
    # adjust x and y limit
    y11,y12 = ax1.get_ylim()
    y21,y22 = ax2.get_ylim()
    z1 = min(y11,y21)
    z2 = max(y21,y22)
    ax1.set_ylim(z1,z2)
    ax2.set_ylim(z1,z2)
    ax2.set_xlim(0, len(overscan_lst)-1)
    # adjust rotation angle of ticks in time axis
    plt.setp(ax1.get_xticklabels(),rotation=30)

    # save figure
    fig.savefig(figfile)
    plt.close(fig)

def plot_bias_smooth(bias, bias_smooth, comp_figfile, hist_figfile):
    """Plot the bias, smoothed bias, and residual after smoothing.

    A figure will be generated in the report directory of the reduction.
    The name of the figure is given in the config file.

    Args:
        bias (:class:`numpy.ndarray`): Bias array.
        bias_smooth (:class:`numpy.ndarray`): Smoothed bias array.
        comp_figfile (str): Filename of the comparison figure.
        hist_figfile (str): Filename of the histogram figure.
    """
    h, w = bias.shape
    # calculate the residual between bias and smoothed bias data
    bias_res = bias - bias_smooth

    fig1 = plt.figure(figsize=(12,4), dpi=150)
    ax1 = fig1.add_axes([0.055, 0.12, 0.25, 0.75])
    ax2 = fig1.add_axes([0.355, 0.12, 0.25, 0.75])
    ax3 = fig1.add_axes([0.655, 0.12, 0.25, 0.75])
    mean = bias.mean(dtype=np.float64)
    std  = bias.std(dtype=np.float64, ddof=1)
    vmin = mean - 2.*std
    vmax = mean + 2.*std
    cax1 = ax1.imshow(bias,        vmin=vmin, vmax=vmax, cmap='gray')
    cax2 = ax2.imshow(bias_smooth, vmin=vmin, vmax=vmax, cmap='gray')
    cax3 = ax3.imshow(bias_res,    vmin=vmin, vmax=vmax, cmap='gray')
    cbar_ax = fig1.add_axes([0.925, 0.12, 0.02, 0.75])
    cbar = fig1.colorbar(cax1, cax=cbar_ax)
    ax1.set_title('bias')
    ax2.set_title('bias_smooth')
    ax3.set_title('bias - bias_smooth')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(0, bias.shape[1]-1)
        ax.set_ylim(bias.shape[1]-1, 0)
        ax.set_xlabel('X', fontsize=11)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(10)
    # only show y label in the left panel
    ax1.set_ylabel('Y',fontsize=11)
    
    # plot the histogram of smoothed bias
    # prepare the bin list
    bins = np.linspace(-4, 4, 40+1)
    
    # prepare the gaussian fitting and error function
    fitfunc = lambda p,x:p[0]*np.exp(-0.5*(x-p[1])**2/p[2]**2)
    errfunc = lambda p,x,y: y-fitfunc(p,x)
    
    # create figure
    fig2 = plt.figure(figsize=(8,6), dpi=150)
    for i, j in [(i, j) for i in range(3) for j in range(3)]:
        ax = fig2.add_axes([0.1+j*0.3, 0.7-i*0.3, 0.27, 0.27])
    
    labels = 'abcdefghi'
    alpha  = 0.7
    # plot both bias and smoothed bias
    for idata,data in enumerate([bias,bias_res]):
        message = ['Parameters for gaussian fitting of the histograms',
                   'y, x, A, center, sigma']
        for iy, ix in [(iy, ix) for iy in range(3) for ix in range(3)]:
            yc = iy*(h//4) + h//4
            xc = ix*(w//4) + w//4
            x1, x2 = xc-200, xc+200
            y1, y2 = yc-200, yc+200
            ax1.plot([x1,x2], [y1,y1], 'm-', alpha=alpha)
            ax1.plot([x1,x2], [y2,y2], 'm-', alpha=alpha)
            ax1.plot([x1,x1], [y1,y2], 'm-', alpha=alpha)
            ax1.plot([x2,x2], [y1,y2], 'm-', alpha=alpha)
            ax3.plot([x1,x2], [y1,y1], 'c-', alpha=alpha)
            ax3.plot([x1,x2], [y2,y2], 'c-', alpha=alpha)
            ax3.plot([x1,x1], [y1,y2], 'c-', alpha=alpha)
            ax3.plot([x2,x2], [y1,y2], 'c-', alpha=alpha)
            ax1.text(xc-50,yc-20,'(%s)'%labels[iy*3+ix],color='m')
            ax3.text(xc-50,yc-20,'(%s)'%labels[iy*3+ix],color='c')
            data_cut = data[y1:y2,x1:x2]
            y,_ = np.histogram(data_cut, bins=bins)
            x = (np.roll(bins,1) + bins)/2
            x = x[1:]
            # use least square minimization function in scipy
            p1,succ = opt.leastsq(errfunc,[y.max(),0.,1.],args=(x,y))
            ax = fig2.get_axes()[iy*3+ix]
            color1 = ('r', 'b')[idata]
            color2 = ('m', 'c')[idata]
            # plot the histogram
            ax.bar(x, y, align='center', color=color1, width=0.2, alpha=0.5)
            # plot the gaussian fitting of histogram
            xnew = np.linspace(x[0], x[-1], 201)
            ax.plot(xnew, fitfunc(p1, xnew), color2+'-', lw=2)
            ax.set_xlim(-4, 4)
            x1,x2 = ax.get_xlim()
            y1,y2 = ax.get_ylim()
            message.append('%4d %4d %+10.8e %+10.8e %+6.3f'%(
                            yc, xc, p1[0], p1[1], p1[2]))
    
        # write the fitting parameters into running log
        logger.info((os.linesep+'  ').join(message))
   
    # find maximum y in different axes
    max_y = 0
    for iax, ax in enumerate(fig2.get_axes()):
        y1, y2 = ax.get_ylim()
        if y2 > max_y:
            max_y = y2
    
    # set y range for all axes
    for iax, ax in enumerate(fig2.get_axes()):
        x1, x2 = ax.get_xlim()
        ax.text(0.9*x1+0.1*x2, 0.2*y1+0.8*y2, '(%s)'%labels[iax],
                fontsize=12)
        ax.set_ylim(0, max_y)
    
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
    
        if iax in [0, 3, 6]:
            ax.set_ylabel('$N$', fontsize=11)
        else:
            ax.set_yticklabels([])
        if iax in [6, 7, 8]:
            ax.set_xlabel('Counts', fontsize=11)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(9)

    # save figures
    fig1.savefig(comp_figfile)
    fig2.savefig(hist_figfile)
    plt.close(fig1)
    plt.close(fig2)


class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.43,0.86])
        self.ax2 = self.add_axes([0.52,0.50,0.43,0.40])
        self.ax3 = self.add_axes([0.52,0.10,0.43,0.40])
        self.ax4 = self.ax3.twinx()


def reduce():
    """2D to 1D pipeline for FOCES on the 2m Fraunhofer Telescope in Wendelstein
    Observatory.
    """

    # find obs log
    logname_lst = [fname for fname in os.listdir(os.curdir)
                        if fname[-7:]=='.obslog']
    if len(logname_lst)==0:
        print('No observation log found')
        exit()
    elif len(logname_lst)>1:
        print('Multiple observation log found:')
        for logname in sorted(logname_lst):
            print('  '+logname)
    else:
        pass

    # read obs log
    logtable = read_obslog(logname_lst[0])

    # load config files
    config_file_lst = []
    # find built-in config file
    config_path = os.path.join(os.path.dirname(__file__), '../data/config')
    config_file = os.path.join(config_path, 'FOCES.cfg')
    if os.path.exists(config_file):
        config_file_lst.append(config_file)

    # find local config file
    for fname in os.listdir(os.curdir):
        if fname[-4:]=='.cfg':
            config_file_lst.append(fname)

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # extract keywords from config file
    section = config['data']
    rawdata     = section.get('rawdata')
    statime_key = section.get('statime_key')
    exptime_key = section.get('exptime_key')
    direction   = section.get('direction')
    multi_fiber = section.get('multi_fiber')
    section = config['reduce']
    midproc     = section.get('midproc')
    onedspec    = section.get('onedspec')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):   os.mkdir(report)
    if not os.path.exists(onedspec): os.mkdir(onedspec)
    if not os.path.exists(midproc):  os.mkdir(midproc)

    # initialize printing infomation
    pinfo1 = FormattedInfo(all_columns, ['frameid', 'fileid', 'imgtype',
                'object', 'exptime', 'obsdate', 'nsat', 'q95'])
    pinfo2 = pinfo1.add_columns([('overscan', 'float', '{:^8s}', '{1:8.2f}')])

    # count the number of fibers
    n_fiber = 1
    if multi_fiber:
        for logitem in logtable:
            n = len(logitem['object'].split(';'))
            n_fiber = max(n_fiber, n)
        message = ', '.join(['multi_fiber = {}'.format(multi_fiber),
                             'number of fiber = {}'.format(n_fiber)])
        print(message)

    ################################ parse bias ################################
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        bias, head = fits.getdata(bias_file, header=True)
        bias_card_lst = [card for card in head.cards
                            if card.keyword[0:10]=='GAMSE BIAS']
        logger.info('Load bias from image: %s'%bias_file)
    else:
        # read each individual CCD
        bias_data_lst = []
        bias_head_lst = []
        bias_fileid_lst = []

        for logitem in logtable:
            if logitem['object'].strip().lower()=='bias':
                fname = logitem['fileid']+'.fits'
                filename = os.path.join(rawdata, fname)
                data, head = fits.getdata(filename, header=True)
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data, head)
                data, head, overmean = correct_overscan(data, head, mask)

                # print info
                if len(bias_data_lst) == 0:
                    print('* Combine Bias Images: {}'.format(bias_file))
                    print(' '*2 + pinfo2.get_separator())
                    print(' '*2 + pinfo2.get_title())
                    print(' '*2 + pinfo2.get_separator())
                string = pinfo2.get_format().format(logitem, overmean)
                print(' '*2 + print_wrapper(string, logitem))

                bias_data_lst.append(data)
                bias_head_lst.append(head)
                bias_fileid_lst.append(logitem['fileid'])

        n_bias = len(bias_data_lst)         # number of bias images
        has_bias = n_bias > 0

        if has_bias:
            # there is bias frames
            print(' '*2 + pinfo2.get_separator())

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            section = config['reduce.bias']
            bias = combine_images(bias_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('cosmic_clip'),
                    maxiter    = section.getint('maxiter'),
                    mask       = (None, 'max')[n_bias>=3],
                    )

            # initialize card list for bias header
            bias_card_lst = []
            bias_card_lst.append(('HIERARCH GAMSE BIAS NFILE', n_bias))

            # move cards related to OVERSCAN corrections from individual headers
            # to bias header
            for ifile, (head, fileid) in enumerate(zip(bias_head_lst,
                                                       bias_fileid_lst)):
                key_prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ifile)
                bias_card_lst.append((key_prefix+' FILEID', fileid))

                # move OVERSCAN cards to bias header
                for card in head.cards:
                    mobj = re.match('^GAMSE (OVERSCAN[\s\S]*)', card.keyword)
                    if mobj is not None:
                        newkey = key_prefix + ' ' + mobj.group(1)
                        bias_card_lst.append((newkey, card.value))

            ############## bias smooth ##################
            key_prefix = 'HIERARCH GAMSE BIAS SMOOTH'
            section = config['reduce.bias']
            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                if smooth_method in ['gauss', 'gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')

                    bias_smooth = gaussian_filter(bias,
                                    sigma=smooth_sigma, mode=smooth_mode)

                    # write information to FITS header
                    bias_card_lst.append((key_prefix,           True))
                    bias_card_lst.append((key_prefix+' METHOD', 'GAUSSIAN'))
                    bias_card_lst.append((key_prefix+' SIGMA',  smooth_sigma))
                    bias_card_lst.append((key_prefix+' MODE',   smooth_mode))
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth
            else:
                # bias not smoothed
                bias_card_lst.append((key_prefix, False))

            # create new FITS Header for bias
            head = fits.Header()
            for card in bias_card_lst:
                head.append(card)
            fits.writeto(bias_file, bias, header=head, overwrite=True)

            message = 'Bias image written to "{}"'.format(bias_file)
            logger.info(message)
            print(message)

        else:
            # no bias found
            pass

    ######################### find flat groups #################################
    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)

    # initialize flat_groups for both single fiber and multi-fibers
    flat_groups = {chr(ifiber+65): {} for ifiber in range(n_fiber)}
    # flat_groups = {'A':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}
    #                'B':{'flat_M': [fileid1, fileid2, ...],
    #                     'flat_N': [fileid1, fileid2, ...]}}

    for logitem in logtable:
        if multi_fiber:
            fiberobj_lst = [v.strip() for v in logitem['object'].split(';')]
        else:
            fiberobj_lst = [logitem['object']]

        if n_fiber > len(fiberobj_lst):
            continue

        for ifiber in range(n_fiber):
            fiber = chr(ifiber+65)
            objname = fiberobj_lst[ifiber].lower().strip()
            mobj = re.match('^flat[\s\S]*', objname)
            if mobj is not None:
                # the object name of the channel matches "flat ???"
            
                # check the lengthes of names for other channels
                # if this list has no elements (only one fiber) or has no
                # names, this frame is a single-channel flat
                other_lst = [v for i, name in enumerate(fiberobj_lst)
                                    if i != ifiber and len(name)>0]
                if len(other_lst)>0:
                    # this frame is not a single chanel flat. Skip
                    continue

                # find a proper name (flatname) for this flat
                if objname=='flat':
                    # no special names given, use exptime
                    flatname = '{:g}'.format(logitem['exptime'])
                else:
                    # flatname is given. replace space with "_"
                    # remove "flat" before the objectname. e.g.,
                    # "Flat Red" becomes "Red" 
                    char = objname[4:].strip()
                    flatname = char.replace(' ','_')
            
                # add flatname to flat_groups
                if flatname not in flat_groups[fiber]:
                    flat_groups[fiber][flatname] = []
                flat_groups[fiber][flatname].append(logitem)

    '''
    # print the flat_groups
    for ifiber in range(n_fiber):
        fiber = chr(ifiber+65)
        print(fiber)
        for flatname, item_lst in flat_groups[fiber].items():
            print(flatname)
            for item in item_lst:
                print(fiber, flatname, item['fileid'], item['exptime'])
    '''

    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_norm_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    flat_mask_lst = {fiber: {} for fiber in sorted(flat_groups.keys())}
    aperset_lst   = {fiber: {} for fiber in sorted(flat_groups.keys())}

    # first combine the flats
    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        for flatname, item_lst in sorted(fiber_flat_lst.items()):
            nflat = len(item_lst)       # number of flat fieldings

            if multi_fiber:
                # multi-fiber
                flat_filename = os.path.join(midproc,
                        'flat_{}_{}.fits'.format(fiber, flatname))
                aperset_filename = os.path.join(midproc,
                        'trace_flat_{}_{}.trc'.format(fiber, flatname))
                aperset_regname = os.path.join(midproc,
                        'trace_flat_{}_{}.reg'.format(fiber, flatname))
                trace_figname = os.path.join(report,
                        'trace_flat_{}_{}.{}'.format(fiber, flatname,
                            fig_format))
            else:
                # single-fiber
                flat_filename = os.path.join(midproc,
                        'flat_{}.fits'.format(flatname))
                aperset_filename = os.path.join(midproc,
                        'trace_flat_{}.trc'.format(flatname))
                aperset_regname = os.path.join(midproc,
                        'trace_flat_{}.reg'.format(flatname))
                trace_figname = os.path.join(report,
                        'trace_flat_{}.{}'.format(flatname, fig_format))

            # get flat_data and mask_array for each flat group
            if mode=='debug' and os.path.exists(flat_filename) \
                and os.path.exists(aperset_filename):
                # read flat data and mask array
                hdu_lst = fits.open(flat_filename)
                flat_data  = hdu_lst[0].data
                exptime    = hdu_lst[0].header[exptime_key]
                mask_array = hdu_lst[1].data
                hdu_lst.close()
                aperset = load_aperture_set(aperset_filename)
            else:
                # if the above conditions are not satisfied, comine each flat
                data_lst = []
                head_lst = []
                exptime_lst = []

                print('* Combine {} Flat Images: {}'.format(nflat, flat_filename))
                print(' '*2 + pinfo2.get_separator())
                print(' '*2 + pinfo2.get_title())
                print(' '*2 + pinfo2.get_separator())

                for i_item, logitem in enumerate(item_lst):
                    # read each individual flat frame
                    filename = os.path.join(rawdata, logitem['fileid']+'.fits')
                    data, head = fits.getdata(filename, header=True)
                    exptime_lst.append(head[exptime_key])
                    if data.ndim == 3:
                        data = data[0,:,:]
                    mask = get_mask(data, head)
                    sat_mask = (mask&4>0)
                    bad_mask = (mask&2>0)
                    if i_item == 0:
                        allmask = np.zeros_like(mask, dtype=np.int16)
                    allmask += sat_mask

                    # correct overscan for flat
                    data, head, overmean = correct_overscan(data, head, mask)

                    # correct bias for flat, if has bias
                    if has_bias:
                        data = data - bias
                        logger.info('Bias corrected')
                    else:
                        logger.info('No bias. skipped bias correction')

                    # print info
                    string = pinfo2.get_format().format(logitem, overmean)
                    print(' '*2 + print_wrapper(string, logitem))

                    data_lst.append(data)

                print(' '*2 + pinfo2.get_separator())

                if nflat == 1:
                    flat_data = data_lst[0]
                else:
                    data_lst = np.array(data_lst)
                    flat_data = combine_images(data_lst,
                                    mode       = 'mean',
                                    upper_clip = 10,
                                    maxiter    = 5,
                                    mask       = (None, 'max')[nflat>3],
                                    )

                # get mean exposure time and write it to header
                head = fits.Header()
                exptime = np.array(exptime_lst).mean()
                head[exptime_key] = exptime

                # find saturation mask
                sat_mask = allmask > nflat/2.
                mask_array = np.int16(sat_mask)*4 + np.int16(bad_mask)*2

                # pack results and save to fits
                hdu_lst = fits.HDUList([
                            fits.PrimaryHDU(flat_data, head),
                            fits.ImageHDU(mask_array),
                            ])
                hdu_lst.writeto(flat_filename, overwrite=True)

                # now flt_data and mask_array are prepared

                # create the trace figure
                tracefig = TraceFigure()

                # if debackground before detecting the orders, then we loose the 
                # ability to detect the weak blue orders.
                #xnodes = np.arange(0, flat_data.shape[1], 200)
                #flat_debkg = simple_debackground(flat_data, mask_array, xnodes,
                # smooth=5)
                #aperset = find_apertures(flat_debkg, mask_array,
                section = config['reduce.trace']
                aperset = find_apertures(flat_data, mask_array,
                            scan_step  = section.getint('scan_step'),
                            minimum    = section.getfloat('minimum'),
                            separation = section.get('separation'),
                            align_deg  = section.getint('align_deg'),
                            filling    = section.getfloat('filling'),
                            degree     = section.getint('degree'),
                            display    = section.getboolean('display'),
                            fig        = tracefig,
                            )

                # save the trace figure
                tracefig.adjust_positions()
                tracefig.suptitle('Trace for {}'.format(flat_filename), fontsize=15)
                tracefig.savefig(trace_figname)

                aperset.save_txt(aperset_filename)
                aperset.save_reg(aperset_regname)

            # append the flat data and mask
            flat_data_lst[fiber][flatname] = flat_data
            flat_norm_lst[fiber][flatname] = flat_data/exptime
            flat_mask_lst[fiber][flatname] = mask_array
            aperset_lst[fiber][flatname]   = aperset

    ########################### Get flat fielding ##############################
    flatmap_lst = {}

    for fiber, fiber_group in sorted(flat_groups.items()):
        for flatname in sorted(fiber_group.keys()):

            # get filename of flat
            if multi_fiber:
                flat_filename = os.path.join(midproc,
                        'flat_{}_{}.fits'.format(fiber, flatname))
            else:
                flat_filename = os.path.join(midproc,
                        'flat_{}.fits'.format(flatname))

            hdu_lst = fits.open(flat_filename, mode='update')
            if len(hdu_lst)>=3:
                # sensitivity map already exists in fits file
                flatmap = hdu_lst[2].data
                hdu_lst.close()
            else:
                # do flat fielding
                print('*** Start parsing flat fielding: %s ***'%flat_filename)
                if multi_fiber:
                    fig_aperpar = {
                        'debug': os.path.join(report,
                                'flat_aperpar_{}_{}_%03d.{}'.format(
                                    fiber, flatname, fig_format)),
                        'normal': None,
                        }[mode]

                    fig_slit = os.path.join(report,
                                    'slit_flat_{}_{}.{}'.format(
                                        fiber, flatname, fig_format))
                else:
                    fig_aperpar = {
                        'debug': os.path.join(report,
                                'flat_aperpar_{}_%03d.{}'.format(
                                    flatname, fig_format)),
                        'normal': None,
                        }[mode]

                    fig_slit = os.path.join(report,
                                    'slit_flat_{}.{}'.format(
                                        flatname, fig_format))
    
                section = config['reduce.flat']
    
                flatmap = get_fiber_flat(
                            data            = flat_data_lst[fiber][flatname],
                            mask            = flat_mask_lst[fiber][flatname],
                            apertureset     = aperset_lst[fiber][flatname],
                            slit_step       = section.getint('slit_step'),
                            nflat           = len(flat_groups[fiber][flatname]),
                            q_threshold     = section.getfloat('q_threshold'),
                            smooth_A_func   = smooth_aperpar_A,
                            smooth_k_func   = smooth_aperpar_k,
                            smooth_c_func   = smooth_aperpar_c,
                            smooth_bkg_func = smooth_aperpar_bkg,
                            fig_aperpar     = fig_aperpar,
                            fig_overlap     = None,
                            fig_slit        = fig_slit,
                            slit_file       = None,
                            )
                
                # append the sensivity map to fits file
                hdu_lst.append(fits.ImageHDU(flatmap))
                # write back to the original file
                hdu_lst.flush()
    
            # append the flatmap
            if fiber not in flatmap_lst:
                flatmap_lst[fiber] = {}
            flatmap_lst[fiber][flatname] = flatmap
    
            # continue to the next colored flat
        # continue to the next fiber

    ############################# Mosaic Flats #################################
    flat_file = os.path.join(midproc, 'flat.fits')
    trac_file = os.path.join(midproc, 'trace.trc')
    treg_file = os.path.join(midproc, 'trace.reg')

    master_aperset = {}

    for fiber, fiber_flat_lst in sorted(flat_groups.items()):
        # determine the mosaiced flat filename
        if multi_fiber:
            flat_fiber_file = os.path.join(midproc,
                                'flat_{}.fits'.format(fiber))
            trac_fiber_file = os.path.join(midproc,
                                'trace_{}.trc'.format(fiber))
            treg_fiber_file = os.path.join(midproc,
                                'trace_{}.reg'.format(fiber))
        else:
            flat_fiber_file = flat_file
            trac_fiber_file = trac_file
            treg_fiber_file = treg_file

        if len(fiber_flat_lst) == 1:
            # there's only 1 kind of flat
            flatname = list(fiber_flat_lst)[0]

            # copy the flat fits
            if multi_fiber:
                oriname = 'flat_{}_{}.fits'.format(fiber, flatname)
            else:
                oriname = 'flat_{}.fits'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), flat_fiber_file)

            # copy the trc file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.trc'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.trc'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), trac_fiber_file)

            # copy the reg file
            if multi_fiber:
                oriname = 'trace_flat_{}_{}.reg'.format(fiber, flatname)
            else:
                oriname = 'trace_flat_{}.reg'.format(flatname)
            shutil.copyfile(os.path.join(midproc, oriname), treg_fiber_file)


            flat_map = flatmap_lst[fiber][flatname]
    
            # no need to mosaic aperset
            master_aperset[fiber] = list(aperset_lst[fiber].values())[0]
        else:
            # mosaic apertures
            section = config['reduce.flat']
            master_aperset[fiber] = mosaic_flat_auto(
                    aperture_set_lst = aperset_lst[fiber],
                    max_count        = section.getfloat('mosaic_maxcount'),
                    )
            # mosaic original flat images
            flat_data = mosaic_images(flat_data_lst[fiber],
                                        master_aperset[fiber])
            # mosaic flat mask images
            mask_data = mosaic_images(flat_mask_lst[fiber],
                                        master_aperset[fiber])
            # mosaic sensitivity map
            flat_map = mosaic_images(flatmap_lst[fiber],
                                        master_aperset[fiber])
            # mosaic exptime-normalized flat images
            flat_norm = mosaic_images(flat_norm_lst[fiber],
                                        master_aperset[fiber])
    
            # pack and save to fits file
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(flat_data),
                        fits.ImageHDU(mask_data),
                        fits.ImageHDU(flat_map),
                        fits.ImageHDU(flat_norm),
                        ])
            hdu_lst.writeto(flat_fiber_file, overwrite=True)
    
            master_aperset[fiber].save_txt(trac_fiber_file)
            master_aperset[fiber].save_reg(treg_fiber_file)


    exit()
    ############################## Extract ThAr ################################

    # get the data shape
    h, w = flat_map.shape

    # define dtype of 1-d spectra
    types = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux',       (np.float32, w)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})
    
    calib_lst = {}
    count_thar = 0
    for logitem in logtable:

        if logitem['object'].strip().lower() != 'thar':
            continue

        count_thar += 1
        fileid = logitem['fileid']

        filename = os.path.join(rawdata, fileid+'.fits')
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data, head)

        # correct overscan for ThAr
        data, head, overmean = correct_overscan(data, head, mask)

        # correct bias for ThAr, if has bias
        if has_bias:
            data = data - bias
            logger.info('Bias corrected')
        else:
            logger.info('No bias. skipped bias correction')

        section = config['reduce.extract']
        spectra1d = extract_aperset(data, mask,
                    apertureset = master_aperset,
                    lower_limit = section.getfloat('lower_limit'),
                    upper_limit = section.getfloat('upper_limit'),
                    )
        head = master_aperset.to_fitsheader(head, channel=None)
    
        spec = []
        for aper, item in sorted(spectra1d.items()):
            flux_sum = item['flux_sum']
            spec.append((aper, 0, flux_sum.size,
                    np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
        spec = np.array(spec, dtype=spectype)
    
        wlcalib_fig = os.path.join(report,
                'wlcalib_{}.{}'.format(fileid, fig_format))

        section = config['reduce.wlcalib']

        if count_thar == 1:
            # this is the first ThAr frame in this observing run
            if section.getboolean('search_database'):
                # find previouse calibration results
                database_path = section.get('database_path')
                search_path = os.path.join(database_path,
                                            'FOCES/wlcalib')

                result = select_calib_from_database(
                    search_path, statime_key, head[statime_key],
                    channel=None)
                ref_spec, ref_calib = result
    
                if ref_spec is None or ref_calib is None:
                    # if failed, pop up a calibration window and
                    # identify the wavelengths manually
                    calib = wlcalib(spec,
                        filename      = fileid+'.fits',
                        figfilename   = wlcalib_fig,
                        channel       = None,
                        linelist      = section.get('linelist'),
                        window_size   = section.getint('window_size'),
                        xorder        = section.getint('xorder'),
                        yorder        = section.getint('yorder'),
                        maxiter       = section.getint('maxiter'),
                        clipping      = section.getfloat('clipping'),
                        snr_threshold = section.getfloat(
                                            'snr_threshold'),
                        )
                else:
                    # if success, run recalib
                    # determine the direction
                    ref_direction = ref_calib['direction']
                    aperture_k = ((-1, 1)[direction[1]==ref_direction[1]],
                                    None)[direction[1]=='?']
                    pixel_k = ((-1, 1)[direction[2]==ref_direction[2]],
                                None)[direction[2]=='?']

                    result = find_caliblamp_offset(ref_spec, spec,
                                aperture_k = aperture_k,
                                pixel_k    = pixel_k,
                                )
                    aperture_koffset = (result[0], result[1])
                    pixel_koffset    = (result[2], result[3])

                    print('Aperture offset =', aperture_koffset)
                    print('Pixel offset =', pixel_koffset)

                    use = section.getboolean('use_prev_fitpar')
                    xorder      = (section.getint('xorder'), None)[use]
                    yorder      = (section.getint('yorder'), None)[use]
                    maxiter     = (section.getint('maxiter'), None)[use]
                    clipping    = (section.getfloat('clipping'), None)[use]
                    window_size = (section.getint('window_size'), None)[use]
                    q_threshold = (section.getfloat('q_threshold'), None)[use]

                    calib = recalib(spec,
                        filename         = fileid+'.fits',
                        figfilename      = wlcalib_fig,
                        ref_spec         = ref_spec,
                        channel          = None,
                        linelist         = section.get('linelist'),
                        aperture_koffset = aperture_koffset,
                        pixel_koffset    = pixel_koffset,
                        ref_calib        = ref_calib,
                        xorder           = xorder,
                        yorder           = yorder,
                        maxiter          = maxiter,
                        clipping         = clipping,
                        window_size      = window_size,
                        q_threshold      = q_threshold,
                        direction        = direction,
                        )
            else:
                # do not search the database
                calib = wlcalib(spec,
                    filename      = fileid+'.fits',
                    figfilename   = wlcalib_fig,
                    channel       = None,
                    identfilename = section.get('ident_file', None),
                    linelist      = section.get('linelist'),
                    window_size   = section.getint('window_size'),
                    xorder        = section.getint('xorder'),
                    yorder        = section.getint('yorder'),
                    maxiter       = section.getint('maxiter'),
                    clipping      = section.getfloat('clipping'),
                    q_threshold   = section.getfloat('q_threshold'),
                    )

            # then use this ThAr as the reference
            ref_calib = calib
            ref_spec  = spec
        else:
            # for other ThArs, no aperture offset
            calib = recalib(spec,
                filename      = fileid+'.fits',
                figfilename   = wlcalib_fig,
                ref_spec      = ref_spec,
                channel       = None,
                linelist      = section.get('linelist'),
                aperture_offset = 0,
                coeff         = ref_calib['coeff'],
                npixel        = ref_calib['npixel'],
                window_size   = ref_calib['window_size'],
                xorder        = ref_calib['xorder'],
                yorder        = ref_calib['yorder'],
                maxiter       = ref_calib['maxiter'],
                clipping      = ref_calib['clipping'],
                snr_threshold = ref_calib['snr_threshold'],
                k             = ref_calib['k'],
                offset        = ref_calib['offset'],
                )
        
        hdu_lst = self_reference_singlefiber(spec, head, calib)
        filename = os.path.join(onedspec, fileid+oned_suffix+'.fits')
        hdu_lst.writeto(filename, overwrite=True)

        # add more infos in calib
        calib['fileid']   = fileid
        calib['date-obs'] = head[statime_key]
        calib['exptime']  = head[exptime_key]
        # pack to calib_lst
        calib_lst[logitem['frameid']] = calib

    for frameid, calib in sorted(calib_lst.items()):
        print(' [{:3d}] {} - {:4d}/{:4d} r.m.s. = {:7.5f}'.format(frameid,
                calib['fileid'], calib['nuse'], calib['ntot'], calib['std']))
    
    # print promotion and read input frameid list
    string = input('select references: ')
    ref_frameid_lst = [int(s) for s in string.split(',')
                                if len(s.strip())>0 and
                                s.strip().isdigit() and
                                int(s) in calib_lst]
    ref_calib_lst    = [calib_lst[frameid]
                            for frameid in ref_frameid_lst]
    ref_datetime_lst = [calib_lst[frameid]['date-obs']
                            for frameid in ref_frameid_lst]

    #################### Extract Science Spectrum ##############################

    for logitem in logtable:
        
        # logitem alias
        fileid  = logitem['fileid']
        imgtype = logitem['imgtype']

        if imgtype == 'sci':

            filename = os.path.join(rawdata, fileid+'.fits')

            logger.info(
                'FileID: {} ({}) - start reduction: {}'.format(
                fileid, imgtype, filename)
                )

            # read raw data
            data, head = fits.getdata(filename, header=True)
            if data.ndim == 3:
                data = data[0,:,:]
            mask = get_mask(data, head)

            # correct overscan
            data, head, overmean = correct_overscan(data, head, mask)
            message = 'FileID: {} - overscan corrected'.format(fileid)

            logger.info(message)
            print(message)

            # correct bias
            if has_bias:
                data = data - bias
                message = 'FileID: {} - bias corrected. mean value = {}'.format(
                            fileid, bias.mean())
            else:
                message = 'FileID: {} - no bias'.format(fileid)

            logger.info(message)
            print(message)

            # correct flat
            data = data/flat_map

            message = 'FileID: {} - flat corrected'.format(fileid)
            logger.info(message)
            print(message)

            # correct background
            fig_sec = os.path.join(report,
                      'bkg_{}_sec.{}'.format(fileid, fig_format))

            section = config['reduce.background']
            stray = find_background(data, mask,
                    aperturesets = master_aperset,
                    ncols        = section.getint('ncols'),
                    distance     = section.getfloat('distance'),
                    yorder       = section.getint('yorder'),
                    fig_section  = fig_sec,
                    )
            data = data - stray

            # plot stray light
            fig_stray = os.path.join(report,
                        'bkg_{}_stray.{}'.format(fileid, fig_format))
            plot_background_aspect1(data+stray, stray, fig_stray)

            # generate two figures for each background
            #plot_background_aspect1_alt(data+stray, stray,
            #    os.path.join(report, 'bkg_%s_stray1.%s'%(fileid, fig_format)),
            #    os.path.join(report, 'bkg_%s_stray2.%s'%(fileid, fig_format)))

            message = 'FileID: {} - background corrected'.format(fileid)
            logger.info(message)
            print(message)

            # extract 1d spectrum
            section = config['reduce.extract']
            spectra1d = extract_aperset(data, mask,
                        apertureset = master_aperset,
                        lower_limit = section.getfloat('lower_limit'),
                        upper_limit = section.getfloat('upper_limit'),
                        )

            message = 'FileID: {} - 1D spectra of {} orders extracted'.format(
                        fileid, len(spectra1d))
            logger.info(message)
            print(message)

            # pack spectrum
            spec = []
            for aper, item in sorted(spectra1d.items()):
                flux_sum = item['flux_sum']
                spec.append((aper, 0, flux_sum.size,
                        np.zeros_like(flux_sum, dtype=np.float64), flux_sum))
            spec = np.array(spec, dtype=spectype)

            # wavelength calibration
            weight_lst = get_time_weight(ref_datetime_lst, head[statime_key])

            message = 'FileID: {} - wavelength calibration weights: {}'.format(
                        fileid, ','.join(['%8.4f'%w for w in weight_lst]))

            logger.info(message)
            print(message)

            spec, head = wl_reference_singlefiber(spec, head,
                            ref_calib_lst, weight_lst)

            # pack and save wavelength referenced spectra
            hdu_lst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(spec),
                        ])
            filename = os.path.join(onedspec, fileid+oned_suffix+'.fits')
            hdu_lst.writeto(filename, overwrite=True)

            message = 'FileID: {} - Spectra written to {}'.format(
                        fileid, filename)
            logger.info(message)
            print(message)


def smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *A* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    
    """

    has_fringe_lst = []
    aperpar = np.array([np.nan]*w)
    xpiece_lst     = np.array([np.nan]*newx_lst.size)
    ypiece_res_lst = np.array([np.nan]*newx_lst.size)
    mask_rej_lst   = np.array([np.nan]*newx_lst.size)
    allx = np.arange(w)
    # the dtype of xpiece_lst and ypiece_lst is np.float64

    # first try, scan every segment. find fringe by checking the local maximum
    # points after smoothing. Meanwhile, save the smoothing results in case the
    # smoothing will be used afterwards.
    for group in group_lst:
        i1, i2 = group[0], group[-1]
        p1, p2 = newx_lst[i1], newx_lst[i2]
        m = fitmask[group]
        xpiece = newx_lst[group]
        ypiece = ypara[group]
        # now fill the NaN values in ypiece
        if (~m).sum() > 0:
            f = InterpolatedUnivariateSpline(xpiece[m], ypiece[m], k=3)
            ypiece = f(xpiece)
        # now xpiece and ypiece are ready

        _m = np.ones_like(ypiece, dtype=np.bool)
        for ite in range(3):
            f = InterpolatedUnivariateSpline(xpiece[_m], ypiece[_m], k=3)
            ypiece2 = f(xpiece)
            win_len = (11, 21)[ypiece2.size>23]
            ysmooth = savgol_filter(ypiece2, window_length=win_len, polyorder=3)
            res = ypiece - ysmooth
            std = res.std()
            _new_m = np.abs(res) < 3*std

            # prevent extrapolation at the boundaries
            if _new_m.size > 3:
                _new_m[0:3] = True
                _new_m[-3:] = True
            _new_m = _m*_new_m

            if _new_m.sum() == _m.sum():
                break
            _m = _new_m
        # now xpiece, ypiece, ypiece2, ysmooth, res, and _m have the same
        # lengths and meanings on their positions of elements

        f = InterpolatedUnivariateSpline(xpiece, ysmooth, k=3)
        _x = np.arange(p1, p2+1)

        aperpar[_x] = f(_x)
        xpiece_lst[group] = xpiece
        ypiece_res_lst[group] = res
        mask_rej_lst[group] = ~_m

        # find out if this order is affected by fringes, by checking the
        # distribution of local maximum points
        imax, ymax = get_local_minima(-ysmooth, window=5)
        if len(imax) > 0:
            x = xpiece[imax]
        else:
            x = []
        # determine how many pixels in each bin.
        # if w=4000, then 500 pix. if w=2000, then 250 pix.
        npixbin = w//8
        bins = np.linspace(p1, p2, int(p2-p1)/npixbin+2)
        hist, _ = np.histogram(x, bins)

        n_nonzerobins = np.nonzero(hist)[0].size
        n_zerobins = hist.size - n_nonzerobins

        if p2-p1<w/8 or n_zerobins<=1 or \
            n_zerobins<n_nonzerobins or n_nonzerobins>=3:
            # there's fringe
            has_fringe = True
        else:
            # no fringe
            has_fringe = False
        has_fringe_lst.append(has_fringe)

    # use global polynomial fitting if this order is affected by fringe and the
    # following conditions are satisified
    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2 and \
        has_fringe_lst.count(True) == len(has_fringe_lst):
        # fit polynomial over the whole order

        # prepare xpiece and y piece
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # fit with poly
        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
            upper_clip=3)
        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst = ~_m
    else:
        # scan again
        # fit polynomial for every segment
        for group, has_fringe in zip(group_lst, has_fringe_lst):
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            if has_fringe:
                deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]
            else:
                deg = 7
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, np.log(ypiece), deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)
            ypiece_fit = np.exp(ypiece_fit)
            ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.exp(np.polyval(coeff, ii/w))
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *k* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_c`
        * :func:`smooth_aperpar_bkg`
    """

    allx = np.arange(w)

    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*w)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/w)
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *c* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_bkg`
    """
    return smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, w)

def smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, w):
    """Smooth *bkg* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        w (int): Length of flat.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`) – Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`) – Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`) – Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`) – Mask of sampling
              pixels in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`gamse.echelle.flat.get_fiber_flat`
        * :func:`smooth_aperpar_A`
        * :func:`smooth_aperpar_k`
        * :func:`smooth_aperpar_c`
    """

    allx = np.arange(w)

    # fit for bkg
    if len(group_lst) > 1 and newx_lst[group_lst[0][0]] < w/2 and \
        newx_lst[group_lst[-1][-1]] > w/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        xspan = xpiece[-1] - xpiece[0]
        deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/w, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/w)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*w)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            deg = (((1, 2)[xspan>w/8], 3)[xspan>w/4], 4)[xspan>w/2]

            scale = ('linear','log')[(ypiece<=0).sum()==0]
            if scale=='log':
                ypiece = np.log(ypiece)

            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/w, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            if scale=='log':
                ypiece = np.exp(ypiece)
                ypiece_fit = np.exp(ypiece_fit)
                ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/w)
            if scale=='log':
                aperpar[ii] = np.exp(aperpar[ii])
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst
