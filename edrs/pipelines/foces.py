import os
import re
import datetime
import logging
logger = logging.getLogger(__name__)
import configparser

import dateutil.parser


import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates  as mdates
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter

from ..utils             import obslog
from ..echelle.imageproc import (combine_images, array_to_table,
                                 table_to_array)
from ..echelle.trace import find_apertures, load_aperture_set
from ..echelle.flat import get_fiber_flat, mosaic_flat_auto, mosaic_images
from ..echelle.extract import extract_aperset
from ..echelle.wvcalib import (wvcalib, recalib, select_calib_from_database,
                               self_reference_singlefiber,
                               wv_reference_singlefiber, get_time_weight)
from .reduction          import Reduction

def correct_overscan(data, head, mask=None):
    '''Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.
        mask (:class:`numpy.ndarray`): Input image mask.
    
    Returns:
        tuple: A tuple containing:

            * data (:class:`numpy.ndarray`): The output image with overscan
              corrected.
            * head (:class:`astropy.io.fits.Header`): The updated FITS header.
    '''
    h, w = data.shape
    overdata1 = data[:, 0:20]
    overdata2 = data[:, w-18:w]
    overdata_tot = np.hstack((overdata1, overdata2))

    # find the overscan level along the y axis
    # 1 for the left region, 2 for the right region
    # calculate the mean of overscan region along x direction
    ovr_lst1 = overdata1.mean(dtype=np.float64, axis=1)
    ovr_lst2 = overdata2.mean(dtype=np.float64, axis=1)
    
    # only used the upper ~1/2 regions for culculating mean of overscan
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

    head['HIERARCH EDRS OVERSCAN']        = True
    head['HIERARCH EDRS OVERSCAN METHOD'] = 'mean'
    head['HIERARCH EDRS OVERSCAN AXIS-1'] = '1:20'
    head['HIERARCH EDRS OVERSCAN AXIS-2'] = '%d:%d'%(vy1,vy2)
    head['HIERARCH EDRS OVERSCAN MEAN']   = ovrmean1
    head['HIERARCH EDRS OVERSCAN STDEV']  = ovrstd1

    return new_data, head

def get_mask(data, head):
    '''Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        :class:`numpy.ndarray`: Image mask.
    '''
    # saturated CCD count
    saturation_adu = 63000

    mask_sat = (data[:, 20:-20] >= saturation_adu)

    mask_bad = np.zeros_like(data[:, 20:-20], dtype=np.int16)
    # currently no bad pixels in FOCES CCD

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask

class FOCES(Reduction):
    '''Reduction pipleline for FOCES.
    '''

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
            head['HIERARCH EDRS OVERSCAN']        = True
            head['HIERARCH EDRS OVERSCAN METHOD'] = 'mean'
            head['HIERARCH EDRS OVERSCAN AXIS-1'] = '1:20'
            head['HIERARCH EDRS OVERSCAN AXIS-2'] = '%d:%d'%(vy1,vy2)
            head['HIERARCH EDRS OVERSCAN MEAN']   = ovrmean1
            head['HIERARCH EDRS OVERSCAN STDEV']  = ovrstd1
    
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
        '''
        Bias correction.

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
          
        '''
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
        head['HIERARCH EDRS BIAS NFILE'] = len(bias_id_lst)

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

                head['HIERARCH EDRS BIAS SMOOTH']        = True
                head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode

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
            head['HIERARCH EDRS BIAS SMOOTH'] = False
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
            head['HIERARCH EDRS BIAS'] = True
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
        '''
        Plot the variation of bias level with time.
        A figure will be generated in the report directory of the reduction. The
        name of the figure is given in the config file.
        '''
    
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

def make_log(path):
    '''
    Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (str): Path to the raw FITS files.

    '''
    
    # standard naming convenction for fileid
    name_pattern = '^\d{8}_\d{4}_FOC\d{4}_\w{3}\d$'

    # scan the raw files
    fname_lst = sorted(os.listdir(path))
    log = obslog.Log()
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid  = fname[0:-5]
        obsdate = None
        exptime = None
        data, head = fits.getdata(os.path.join(path, fname), header=True)

        if data.ndim == 3:
            # old FOCES data are 3-dimensional arrays
            scidata = data[0, 20:-20]
        elif data.ndim == 2:
            scidata = data[:,20:-20]
        else:
            print('Unknow dimension of data array')
            raise ValueError

        obsdate = head['FRAME']
        exptime = head['EXPOSURE']

        if re.match(name_pattern, fileid) is not None:
            # fileid matches the standard FOCES naming convention
            if fileid[22:25]=='BIA':
                imagetype, objectname = 'cal', 'Bias'
            elif fileid[22:25]=='FLA':
                imagetype, objectname = 'cal', 'Flat'
            elif fileid[22:25]=='THA':
                imagetype, objectname = 'cal', 'ThAr'
            else:
                objectname = 'Unknown'
                if fileid[22:25]=='SCI':
                    imagetype = 'sci'
                else:
                    imagetype = 'cal'
        else:
            # fileid does not follow the naming convetion
            imagetype, objectname = 'cal', ''

        # determine the fraction of saturated pixels permillage
        mask_sat = (scidata>=63000)
        prop = mask_sat.sum()/scidata.size*1e3

        # find the brightness index in the central region
        h, w = scidata.shape
        data1 = scidata[int(h*0.3):int(h*0.7), w//2-2:w//2+3]
        brightness = np.median(data1,axis=1).mean()

        item = obslog.LogItem(
                   fileid     = fileid,
                   obsdate    = obsdate,
                   exptime    = exptime,
                   objectname = objectname,
                   imagetype  = imagetype,
                   saturation = prop,
                   brightness = brightness,
                   )
        log.add_item(item)

    log.sort('obsdate')

    # make info list
    all_info_lst = []
    column_lst = [
            ('frameid',    'i'),
            ('fileid',     's'),
            ('imagetype',  's'),
            ('objectname', 's'),
            ('exptime',    'f'),
            ('obsdate',    's'),
            ('saturation', 'f'),
            ('brightness', 'f'),
            ]
    columns = ['%s (%s)'%(_name, _type) for _name, _type in column_lst]

    prev_frameid = -1
    for item in log:

        if re.match(name_pattern, item.fileid) is None:
            frameid = prev_frameid + 1
        else:
            frameid = int(item.fileid.split('_')[1])

        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(frameid, prev_frameid))

        info_lst = [
                    str(frameid),
                    str(item.fileid),
                    str(item.imagetype),
                    str(item.objectname),
                    '%g'%item.exptime,
                    str(item.obsdate),
                    '%.3f'%item.saturation,
                    '%.1f'%item.brightness,
                ]
        prev_frameid = frameid
        all_info_lst.append(info_lst)

    # find the maximum length of each column
    length = []
    for info_lst in all_info_lst:
        length.append([len(info) for info in info_lst])
    length = np.array(length)
    maxlen = length.max(axis=0)

    # find the output format for each column
    for info_lst in all_info_lst:
        for i, info in enumerate(info_lst):
            if columns[i] in ['filename','object']:
                fmt = '%%-%ds'%maxlen[i]
            else:
                fmt = '%%%ds'%maxlen[i]
            info_lst[i] = fmt%(info_lst[i])

    # write the obslog into an ascii file
    #date = log[0].fileid.split('_')[0]
    #outfilename = '%s-%s-%s.log'%(date[0:4],date[4:6],date[6:8])
    #outfile = open(outfilename,'w')
    string = '% columns = ' + ', '.join(columns)
    #outfile.write(string+os.linesep)
    print(string)
    for info_lst in all_info_lst:
        string = ' | '.join(info_lst)
        string = ' '+string
        #outfile.write(string+os.linesep)
        print(string)
    #outfile.close()


def get_primary_header(input_lst):
    '''
    Return a list of header records with length of 80 characters.
    The order and datatypes of the records follow the FOCES FITS standards.

    Args:
        input_lst (tuple): A tuple containing the keywords and their values

    Returns:
        *list*: A list containing the records

    '''
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
    '''
    Plot the variation of overscan.
    '''
    
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
    '''
    Plot the bias, smoothed bias, and residual after smoothing.

    A figure will be generated in the report directory of the reduction.
    The name of the figure is given in the config file.

    Args:
        bias (:class:`numpy.ndarray`): Bias array.
        bias_smooth (:class:`numpy.ndarray`): Smoothed bias array.
        comp_figfile (str): Filename of the comparison figure.
        hist_figfile (str): Filename of the histogram figure.
    '''
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

def reduce():
    '''
    2D to 1D pipeline for FOCES on the 2m Fraunhofer Telescope.
    '''

    # read obs log
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

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
    rawdata = config['data']['rawdata']
    section = config['reduce']
    midproc = section['midproc']
    result  = section['result']
    report  = section['report']
    mode    = section.get('mode', 'normal')

    # create folders if not exist
    if not os.path.exists(report):
        os.mkdir(report)
    if not os.path.exists(result):
        os.mkdir(result)
    if not os.path.exists(midproc):
        os.mkdir(midproc)

    ########################### parse bias #############################
    section = config['reduce.bias']
    bias_file = section['bias_file']

    if os.path.exists(bias_file):
        bias = fits.getdata(bias_file)
        has_bias = True
        logger.info('Load bias from image: %s'%bias_file)
    else:
        bias_lst = []
        for item in log:
            if item.objectname[0].strip().lower()=='bias':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data, head)
                data, head = correct_overscan(data, head, mask)
                bias_lst.append(data)

        if len(bias_lst)>0:
            # there is bias frames

            bias = combine_images(bias_lst,
                                mode       = 'mean',
                                upper_clip = section.getfloat('cosmic_clip'),
                                maxiter    = section.getint('maxiter'),
                                )

            # create new FITS Header for bias
            head = fits.Header()
            head['HIERARCH EDRS BIAS NFILE'] = len(bias_lst)

            ############## bias smooth ##################
            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                h, w = bias.shape
                if smooth_method in ['gauss','gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')

                    bias_smooth = gaussian_filter(bias,
                                    sigma=smooth_sigma, mode=smooth_mode)

                    # write information to FITS header
                    head['HIERARCH EDRS BIAS SMOOTH']        = True
                    head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                    head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                    head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth
            else:
                # bias not smoothed
                head['HIERARCH EDRS BIAS SMOOTH'] = False

            fits.writeto(bias_file, bias, header=head, overwrite=True)
            has_bias = True
            logger.info('Bias image written to "bias.fits"')
        else:
            # no bias in this dataset
            has_bias = False

    ######################### find flat groups #################################
    # initialize flat_groups for single fiber
    flat_groups = {}
    for item in log:
        name = item.objectname[0]
        g = name.split()
        if len(g)>0 and g[0].lower().strip() == 'flat':
            # the object name of the channel matches "flat ???"

            # find a proper name for this flat
            if name.lower().strip()=='flat':
                # no special names given, use "flat_A_15"
                flatname = 'flat_%g'%(item.exptime)
            else:
                # flatname is given. replace space with "_"
                # remove "flat" before the objectname. e.g.,
                # "Flat Red" becomes "Red" 
                char = name[4:].strip()
                flatname = 'flat_%s'%(char.replace(' ','_'))

            # add flatname to flat_groups
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(item.fileid)

    ################# Combine the flats and trace the orders ###################
    flat_data_lst = {}
    flat_mask_lst = {}
    aperset_lst   = {}

    # first combine the flats
    for flatname, fileids in flat_groups.items():
        flat_filename    = os.path.join(midproc, '%s.fits'%flatname)
        mask_filename    = os.path.join(midproc, '%s_msk.fits'%flatname)
        aperset_filename = os.path.join(midproc, 'trace_%s.trc'%flatname)
        aperset_regname  = os.path.join(midproc, 'trace_%s.reg'%flatname)

        # get flat_data and mask_array
        if os.path.exists(flat_filename) and os.path.exists(mask_filename) and \
           os.path.exists(aperset_filename):
            flat_data  = fits.getdata(flat_filename)
            mask_table = fits.getdata(mask_filename)
            mask_array = table_to_array(mask_table, flat_data.shape)
            aperset = load_aperture_set(aperset_filename)
        else:
            data_lst = []
            for ifile, fileid in enumerate(fileids):
                filename = os.path.join(rawdata, '%s.fits'%fileid)
                data, head = fits.getdata(filename, header=True)
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data, head)
                sat_mask = (mask&4>0)
                bad_mask = (mask&2>0)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += sat_mask

                # correct overscan for flat
                data, head = correct_overscan(data, head, mask)

                # correct bias for flat, if has bias
                if has_bias:
                    data = data - bias
                    logger.info('Bias corrected')
                else:
                    logger.info('No bias. skipped bias correction')

                data_lst.append(data)
            nflat = len(data_lst)
            print('combine %d images for %s'%(nflat, flatname))
            flat_data = combine_images(data_lst, mode='mean',
                                        upper_clip=10, maxiter=5)
            # write combine result to fits
            fits.writeto(flat_filename, flat_data, overwrite=True)
            # find saturation mask
            sat_mask = allmask > nflat/2.
            mask_array = np.int16(sat_mask)*4 + np.int16(bad_mask)*2
            mask_table = array_to_table(mask_array)
            # write mask to fits
            fits.writeto(mask_filename, mask_table, overwrite=True)
            # now flt_data and mask_array are prepared

            section = config['reduce.trace']
            aperset = find_apertures(flat_data, mask_array,
                        scan_step  = section.getint('scan_step'),
                        minimum    = section.getfloat('minimum'),
                        seperation = section.getfloat('seperation'),
                        sep_der    = section.getfloat('sep_der'),
                        filling    = section.getfloat('filling'),
                        degree     = section.getint('degree'),
                        display    = section.getboolean('display'),
                        filename   = flat_filename,
                        fig_file   = os.path.join(report, 'trace_%s.png'%flatname),
                        )
            aperset.save_txt(aperset_filename)
            aperset.save_reg(aperset_regname)

        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = mask_array
        aperset_lst[flatname]   = aperset

    ########################### Get flat fielding ##############################
    flatmap_lst = {}
    for flatname in sorted(flat_groups.keys()):
        flatmap_filename = os.path.join(midproc, '%s_rsp.fits'%flatname)
        if os.path.exists(flatmap_filename):
            flatmap = fits.getdata(flatmap_filename)
        else:
            # do flat fielding
            print('*** Start parsing flat fielding: %s ***'%flatname)
            fig_aperpar = {
                'debug': os.path.join(report, 'flat_aperpar_'+flatname+'_%03d.png'),
                'normal': None,
                }[mode]

            section = config['reduce.flat']
            flatmap = get_fiber_flat(
                        data        = flat_data_lst[flatname],
                        mask        = flat_mask_lst[flatname],
                        apertureset = aperset_lst[flatname],
                        slit_step   = section.getint('slit_step'),
                        nflat       = len(flat_groups[flatname]),
                        q_threshold = section.getfloat('q_threshold'),
                        param_deg   = section.getint('param_deg'),
                        fig_aperpar = fig_aperpar,
                        fig_overlap = None,
                        fig_slit    = os.path.join(report, '%s_slit.png'%flatname),
                        slit_file   = None,
                        )
            
            # save flat result to fits file
            fits.writeto(flatmap_filename, flatmap, overwrite=True)

        # append the flatmap
        flatmap_lst[flatname] = flatmap

    ############################# Mosaic Flats #################################
    if len(flat_groups) == 1:
        flatname = flat_groups.keys()[0]
        shutil.copyfile(os.path.join(midproc, '%s.fits'%flatname),
                        os.path.join(midproc, 'flat.fits'))
        shutil.copyfile(os.path.join(midproc, '%s_msk.fits'%flatname),
                        os.path.join(midproc, 'flat_msk.fits'))
        shutil.copyfile(os.path.join(midproc, 'trace_%s.trc'),
                        os.path.join(midproc, 'trace.trc'))
        shutil.copyfile(os.path.join(midproc, 'trace_%s.reg'),
                        os.path.join(midproc, 'trace.reg'))
        shutil.copyfile(os.path.join(midproc, '%s_rsp.fits'%flatname),
                        os.path.join(midproc, 'flat_rsp.fits'))
        flat_map = fits.getdata(os.path.join(midproc, 'flat_rsp.fits'))
    else:
        # mosaic apertures
        mosaic_maxcount = config['reduce.flat'].getfloat('mosaic_maxcount')
        mosaic_aperset = mosaic_flat_auto(
                aperture_set_lst = aperset_lst,
                max_count        = mosaic_maxcount,
                )
        # mosaic original flat images
        flat_data = mosaic_images(flat_data_lst, mosaic_aperset)
        fits.writeto(os.path.join(midproc, 'flat.fits'), flat_data, overwrite=True)

        # mosaic flat mask images
        mask_data = mosaic_images(flat_mask_lst, mosaic_aperset)
        mask_table = array_to_table(mask_data)
        fits.writeto(os.path.join(midproc, 'flat_msk.fits'), mask_table, overwrite=True)

        # mosaic sensitivity map
        flat_map = mosaic_images(flatmap_lst, mosaic_aperset)
        fits.writeto(os.path.join(midproc, 'flat_rsp.fits'), flat_map, overwrite=True)

        mosaic_aperset.save_txt(os.path.join(midproc, 'trace.trc'))
        mosaic_aperset.save_reg(os.path.join(midproc, 'trace.reg'))

    ############################## Extract ThAr ################################

    if True:
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
        _names, _formats = list(zip(*types))
        spectype = np.dtype({'names': _names, 'formats': _formats})
    
        calib_lst = {}
        count_thar = 0
        for item in log:
            if item.objectname[0].strip().lower()=='thar':
                count_thar += 1
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                if data.ndim == 3:
                    data = data[0,:,:]
                mask = get_mask(data, head)

                # correct overscan for ThAr
                data, head = correct_overscan(data, head, mask)

                # correct bias for ThAr, if has bias
                if has_bias:
                    data = data - bias
                    logger.info('Bias corrected')
                else:
                    logger.info('No bias. skipped bias correction')

                section = config['reduce.extract']
                spectra1d = extract_aperset(data, mask,
                            apertureset = mosaic_aperset,
                            lower_limit = section.getfloat('lower_limit'),
                            upper_limit = section.getfloat('upper_limit'),
                            )
                head = mosaic_aperset.to_fitsheader(head, channel=None)
    
                spec = []
                for aper, _item in sorted(spectra1d.items()):
                    flux_sum = _item['flux_sum']
                    spec.append(
                             (aper, 0, flux_sum.size,
                              np.zeros_like(flux_sum, dtype=np.float64),
                              flux_sum)
                             )
                spec = np.array(spec, dtype=spectype)
    
                section = config['reduce.wvcalib']

                if count_thar == 1:
                    # this is the first ThAr frame in this observing run
                    if section.getboolean('search_database'):
                        # find previouse calibration results
                        database_path = section.get('database_path')
                        search_path = os.path.join(database_path, 'FOCES/wlcalib')
                        ref_spec, ref_calib, ref_aperset = select_calib_from_database(
                            'FOCES', 'FRAME', head['FRAME'], channel=None)
    
                        # if failed, pop up a calibration window and identify
                        # the wavelengths manually
                        if ref_spec is None or ref_calib is None:
                            calib = wvcalib(spec,
                                filename      = '%s.fits'%item.fileid,
                                figfilename   = os.path.join(report, 'wvcalib_%s.png'%item.fileid),
                                channel       = None,
                                linelist      = section.get('linelist'),
                                window_size   = section.getint('window_size'),
                                xorder        = section.getint('xorder'),
                                yorder        = section.getint('yorder'),
                                maxiter       = section.getint('maxiter'),
                                clipping      = section.getfloat('clipping'),
                                snr_threshold = section.getfloat('snr_threshold'),
                                )
                        else:
                            # if success, run recalib
                            aper_offset = ref_aperset.find_aper_offset(mosaic_aperset)
                            calib = recalib(spec,
                                filename      = '%s.fits'%item.fileid,
                                figfilename   = os.path.join(report, 'wvcalib_%s.png'%item.fileid),
                                ref_spec      = ref_spec,
                                channel       = None,
                                linelist      = section.get('linelist'),
                                aperture_offset = aper_offset,
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
                    else:
                        # do not search the database
                        calib = wvcalib(spec,
                            filename      = '%s.fits'%item.fileid,
                            figfilename   = os.path.join(report, 'wvcalib_%s.png'%item.fileid),
                            channel       = None,
                            identfilename = section.get('ident_file', None),
                            linelist      = section.get('linelist'),
                            window_size   = section.getint('window_size'),
                            xorder        = section.getint('xorder'),
                            yorder        = section.getint('yorder'),
                            maxiter       = section.getint('maxiter'),
                            clipping      = section.getfloat('clipping'),
                            snr_threshold = section.getfloat('snr_threshold'),
                            )

                    # then use this ThAr as the reference
                    ref_calib = calib
                    ref_spec  = spec
                else:
                    # for other ThArs, no aperture offset
                    calib = recalib(spec,
                        filename      = '%s.fits'%item.fileid,
                        figfilename   = os.path.join(report, 'wvcalib_%s.png'%item.fileid),
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
                filename = os.path.join(result, '%s_wlc.fits'%item.fileid)
                hdu_lst.writeto(filename, overwrite=True)

                # add more infos in calib
                calib['fileid']   = item.fileid
                calib['date-obs'] = head['FRAME']
                calib['exptime']  = head['EXPOSURE']
                # pack to calib_lst
                calib_lst[item.frameid] = calib

        for frameid, calib in sorted(calib_lst.items()):
            print(' [%3d] %s - %4d/%4d r.m.s = %7.5f'%(frameid,
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
