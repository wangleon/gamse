import os
import re
import time
import shutil
import logging
import dateutil.parser

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from mpl_toolkits.mplot3d import Axes3D

from ..utils.config  import read_config
from ..utils.obslog  import read_log, parse_num_seq, find_log
from ..ccdproc       import save_fits, table_to_array, array_to_table

class Reduction(object):
    '''General echelle reduction.

    Attributes:
        config (:class:`configparser.ConfigParser`): Config of the reduction.
        log (:class:`stella.utils.obslog.Log`): Observing log.
        paths (tuple): A tuple containing following paths:

            * *'rawdata'*: Path to raw images.
            * *'midproc'*: Path to mid-process files.
            * *'report'*: Path to report file.
            * *'report_img'*: Path to images used in report.

        input_surfix (string): Surfix of filenames before each step.
        output_surfix (string): Surfix of filenames after each step.
        mask_surfix (string): Surfix of mask filenames.

    '''

    def __init__(self):

        # read config file
        self.load_config()

        # read log file
        self.load_log()


    def reduce(self):
        '''
        Main loop of the reduction procedure.
        '''
        # initiliaze file surfix
        self.input_surfix  = ''
        self.output_surfix = ''
        self.mask_surfix = self.config.get('reduction', 'mask_surfix')
        # read steps from config file
        steps_string = self.config.get('reduction', 'steps')

        self.report_file = open(
                os.path.join(self.paths['report'], 'index.html'), 'w')
        self.report_file.write(
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'+
            ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'+
            os.linesep+
            '<html xmlns="http://www.w3.org/1999/xhtml">'+os.linesep+
            '<head>'+os.linesep+
            '    <meta http-equiv="content-type" content="text/html; charset=utf-8" />'+os.linesep+
            '    <title>Reduction Report</title>'+os.linesep+
            '</head>'+os.linesep+
            '<body>'+os.linesep+
            '    <h1>Reduction Report</h1>'+os.linesep)

        # main loop
        step_lst = [v.strip() for v in steps_string.split(',') ]
        logger.info('Reduction steps = %s'%(','.join(step_lst)))
        for step in step_lst:
            if hasattr(self, step):
                getattr(self, step)()
            else:
                logger.error('Unknown step: %s'%step)

        self.report_file.write(
            '</body>'+os.linesep+
            '</html>'
            )

    def load_config(self):
        '''
        Load config file and check the paths.

        .. csv-table:: Accepted options in config file
           :header: "Option", "Type", "Description"
           :widths: 20, 10, 50

           "**obslog_file**",     "*string*", "(*optional*) Name of the observing log file."
           "**path.data**",       "*string*", "Path to the raw images."
           "**path.midproc**",    "*string*", "Path to the mid-process files."
           "**path.report**",     "*string*", "Path to the report file."
           "**path.report_img**", "*string*", "(*optional*) Path to the images of report file."

        '''
        self.config = read_config()

        # get a dict of paths
        self.paths = {}
        for option in self.config.options('reduction'):
            if option[0:5]=='path.':
                self.paths[option[5:]] = self.config.get('reduction', option)

        # Check whether the necessary paths exist

        # check if data path exist
        if not os.path.exists(self.paths['data']):
            logger.error('data path: "%s" does not exist'%self.paths['data'])
            exit()

        # check if midproc path exist
        if not os.path.exists(self.paths['midproc']):
            os.mkdir(self.paths['midproc'])
            logger.info('Create a new directory (midproc path: "%s")'%
                        self.paths['midproc'])

        # check if report path exists
        if not os.path.exists(self.paths['report']):
            os.mkdir(self.paths['report'])
            logger.info('Create a new directory (report path: "%s")'%
                        self.paths['report'])

        # check if image subdirectory of report path exists
        if 'report_img' not in self.paths:
            self.paths['report_img'] = os.path.join(
                                       self.paths['report'], 'images')
        if not os.path.exists(self.paths['report_img']):
            os.mkdir(self.paths['report_img'])
            logger.info('Create a new directory: "%s"'%self.paths['report_img'])

    def find_flat_groups(self):
        '''Find flat groups.
        
        A flat group is a series of flat frames that have the same exposure
        times.
        Temperatures of flat lamps are often much lower than that of celestial
        objects, and therefore flat fieldings are always well illuminated in the
        red part than the blue part of the spectrum.
        Flat images with different exposure times are often obtained during an
        observing run to illuminate different part of the CCD.
        '''

        # initialize flat_groups at different channels
        flat_groups = {chr(ich+65): {} for ich in range(self.nchannels)}

        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            for item in self.log:
                name = item.objectname[ichannel]
                g = name.split()
                if len(g)>0 and g[0].lower().strip()=='flat' and \
                    max([len(v) for i, v in enumerate(item.objectname)
                            if i != ichannel])==0:
                    # this frame is a single channel flat
                    if name.lower().strip()=='flat':
                        flatname = 'flat_%s_%.3f'%(channel, item.exptime)
                    else:
                        flatname = name.replace(' ','_')

                    if flatname not in flat_groups[channel]:
                        flat_groups[channel][flatname] = []
                    flat_groups[channel][flatname].append(item)
        
        self.flat_groups = flat_groups

    def load_log(self):
        '''
        Read the observing log file.
        '''
        if self.config.has_option('reduction', 'obslog_file'):
            obslog_file = self.config.get('reduction','obslog_file')
        else:
            obslog_file = find_log(os.curdir)

        logger.info('obslog_file = "%s"'%obslog_file)
        if not os.path.exists(obslog_file):
            logger.error('Cannot find observing log file: %s'%obslog_file)
            exit()
        self.log = read_log(obslog_file)

        # copy some keywords from the log to the Reduction instance
        self.nchannels = self.log.nchannels

        # find flat groups
        self.find_flat_groups()

    def plot_overscan_variation(self, t_lst, overscan_lst):
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
        figpath = os.path.join(self.paths['report_img'],
                    'overscan_variation.png')
        fig.savefig(figpath)
        logger.info('Save the variation of overscan figure: "%s"'%figpath)
        plt.close(fig)


    def find_bias(self):
        '''Find bias frames.

        Scan the log file and find items with "objectname" containg "bias".

        Returns:
            list: A list containing the IDs of bias frames.
        '''
        
        # find bias in log
        bias_id_lst = []
        for item in self.log:
            for name in item.objectname:
                if name.lower().strip() == 'bias':
                    bias_id_lst.append(item.frameid)
        return bias_id_lst

    def find_trace(self):
        '''Find trace frames.

        Scan the log file and find items with "objectname" containg "bias".

        Returns:
            dict: A dict containing the items of trace frames in each channel.
        '''
        trace_lst = {}
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            
            for item in self.log:
                if item.objectname[ichannel].lower().strip() == 'trace' and \
                    max([len(v) for i, v in enumerate(item.objectname)
                            if i != ichannel])==0:
                    # this frame is a single channle trace image
                    if channel not in trace_lst:
                        trace_lst[channel] = []
                    trace_lst[channel].append(item)

        return trace_lst

    def bias(self):
        '''
        Bias correction.

        .. csv-table:: Accepted options in config file
           :header: "Option", "Type", "Description"
           :widths: 20, 10, 50

           "**bias.skip**",          "*bool*",    "Skip this step if *yes* and **mode** = *'debug'*."
           "**bias.surfix**",        "*string*",  "Surfix of the corrected files."
           "**bias.file**",          "*string*",  "Name of bias file."
           "**bias.cosmic_clip**",   "*float*",   "Upper clipping threshold to remove cosmic-rays."
           "**bias.smooth_method**", "*string*",  "Method of smoothing, including *Gauss*."
           "**bias.smooth_sigma**",  "*integer*", "Sigma of the smoothing filter."
           "**bias.smooth_mode**",   "*string*",  "Mode of the smoothing."

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
        # find output surfix for fits
        self.output_surfix = self.config.get('reduction','bias.surfix')

        if self.config.getboolean('reduction', 'bias.skip'):
            logger.info('Skip [bias] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        self.report_file.write('    <h2>Bias Correction</h2>'+os.linesep)

        bias_id_lst = self.find_bias()

        infile_lst = [os.path.join(self.paths['midproc'],
                        '%s%s.fits'%(item.fileid, self.input_surfix))
                        for item in self.log if item.frameid in bias_id_lst]
        
        # import and stack all bias files in a data cube
        tmp = [fits.getdata(filename, header=True) for filename in infile_lst]
        all_data, all_head = zip(*tmp)
        all_data = np.array(all_data)

        if self.config.has_option('reduction', 'bias.cosmic_clip'):
            # use sigma-clipping method to mask cosmic rays
            cosmic_clip = self.config.getfloat('reduction', 'bias.cosmic_clip')

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
        bias_file = self.config.get('reduction', 'bias.file')

        if self.config.has_option('reduction', 'bias.smooth_method'):
            # perform smoothing for bias
            smooth_method = self.config.get('reduction', 'bias.smooth_method')
            smooth_method = smooth_method.strip().lower()

            logger.info('Smoothing bias: %s'%smooth_method)

            if smooth_method.lower().strip() == 'gaussian':
                # perform 2D gaussian smoothing

                smooth_sigma = self.config.getint('reduction',
                                                  'bias.smooth_sigma')
                smooth_mode  = self.config.get('reduction',
                                               'bias.smooth_mode')

                logger.info('Smoothing bias: sigma = %f'%smooth_sigma)
                logger.info('Smoothing bias: mode = %s'%smooth_mode)

                from scipy.ndimage.filters import gaussian_filter
                bias_smooth = gaussian_filter(bias, smooth_sigma,
                                                mode=smooth_mode)

                logger.info('Smoothing bias: Update bias FITS header')

                head['HIERARCH EDRS BIAS SMOOTH']        = True
                head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode

            else:
                pass

            # bias_data is a proxy for bias to be corrected for each frame
            bias_data = bias_smooth

            # plot comparison between un-smoothed and smoothed data
            self.plot_bias_smooth(bias, bias_smooth)

        else:
            # no smoothing
            logger.info('No smoothing parameter for bias. Skip bias smoothing')
            head['HIERARCH EDRS BIAS SMOOTH'] = False
            bias_data = bias
        
        # save the bias to FITS
        save_fits(bias_file, bias_data, head)
        
        self.plot_bias_variation(all_data, all_head)

        # finally all files are corrected for the bias
        for item in self.log:
            if item.frameid in bias_id_lst:
                continue
            infile  = '%s%s.fits'%(item.fileid, self.input_surfix)
            outfile = '%s%s.fits'%(item.fileid, self.output_surfix)
            inpath  = os.path.join(self.paths['midproc'], infile)
            outpath = os.path.join(self.paths['midproc'], outfile)
            data, head = fits.getdata(inpath, header=True)
            data_new = data - bias_data
            # write information into FITS header
            head['HIERARCH EDRS BIAS'] = True
            # save the bias corrected data
            save_fits(outpath, data_new, head)
            info = ['Correct bias for item no. %d.'%item.frameid,
                    'Save bias corrected file: "%s"'%outpath]
            logger.info((os.linesep+'  ').join(info))
            print('Correct bias: {} => {}'.format(infile, outfile))
        
        
        
        # update surfix
        logger.info('Bias corrected. Change surfix: %s -> %s'%
                    (self.input_surfix, self.output_surfix))
        self.input_surfix = self.output_surfix
   
    def plot_bias_variation(self, data_lst, head_lst, time_key='UTC-STA'):
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
        figpath = os.path.join(self.paths['report_img'], 'bias_variation.png')
        self.report_file.write('        <img src="images/bias_variation.png">'+os.linesep)
        fig.savefig(figpath)
        logger.info('Plot variation of bias with time in figure: "%s"'%figpath)
        plt.close(fig)
    
    def plot_bias_smooth(self, bias, bias_smooth):
        '''
        Plot the bias, smoothed bias, and residual after smoothing.

        A figure will be generated in the report directory of the reduction.
        The name of the figure is given in the config file.

        Args:
            bias (:class:`numpy.array`): Bias array.
            bias_smooth (:class:`numpy.array`): Smoothed bias array.

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
                p1,succ = leastsq(errfunc,[y.max(),0.,1.],args=(x,y))
                ax = fig2.get_axes()[iy*3+ix]
                if idata == 0:
                    color1, color2 = 'r', 'm'
                else:
                    color1, color2 = 'b', 'c'
                # plot the histogram
                ax.bar(x, y, align='center', color=color1, width=0.2,
                       alpha=0.5)
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

        # save the first figure
        figpath1 = os.path.join(self.paths['report_img'], 'bias_smooth.png')
        self.report_file.write(
            ' '*8 + '<img src="images/bias_smooth.png" alt="smoothed bias">' +
            os.linesep)
        fig1.savefig(figpath1)
        logger.info('Plot smoothed bias in figure: "%s"'%figpath1)
        plt.close(fig1)

        # save the second figure
        figpath2 = os.path.join(self.paths['report_img'],'bias_smooth_hist.png')
        self.report_file.write(
            ' '*8 + '<img src="images/bias_smooth_hist.png" ' +
            'alt="histogram of smoothed bias">' + os.linesep)
        fig2.savefig(figpath2)
        logger.info('Plot histograms of smoothed bias in figure: "%s"'%figpath2)
        plt.close(fig2)
    

    def combine_flat(self, item_list, flatname):
        '''
        Combine flat fielding frames.

        Args:
            item_list (list): List of flat items.
            flatname (string): Name of the input flat set.
        Returns:
            No returns.
        '''

        nfile = len(item_list)
    
        # create a header object for combined flat
        newhead = fits.Header()
        newhead['HIERARCH EDRS FLAT TYPE']  = 'mean'
        newhead['HIERARCH EDRS FLAT NFILE'] = nfile
    
        # combine flat
        ifile = 0
        for item in item_list:
            ifile += 1
            # load image data
            basename = '%s%s.fits'%(item.fileid, self.input_surfix)
            filename = os.path.join(self.paths['midproc'], basename)
            data, head = fits.getdata(filename, header=True)

            # if this one is the first input image, initialize the final
            # data array, the mask, and the total exposure time.
            if ifile == 1:
                data_sum = np.zeros_like(data, dtype=np.float64)
                all_sat_mask = np.zeros_like(data, dtype=np.bool)
                total_exptime = 0.

            # add the data array and exposure time
            data_sum += data
            total_exptime += head['EXPTIME']

            # load mask data
            maskname = '%s%s.fits'%(item.fileid, self.mask_surfix)
            maskpath = os.path.join(self.paths['midproc'], maskname)
            mtable = fits.getdata(maskpath)
            if mtable.size==0:
                mdata = np.zeros_like(data, dtype=np.int16)
            else:
                mdata = table_to_array(mtable, data.shape)
            # find saturation pixels
            sat_mask = (mdata&4 == 4)
            # get new all saturation mask
            all_sat_mask = (all_sat_mask|sat_mask)

            # save the filename of each combined flat in to header
            newhead['HIERARCH EDRS FLAT FILE %d'%ifile] = basename
    
        # calculate the mean flat and mean exposure time
        data_mean = data_sum/float(nfile)
        mexptime  = total_exptime/float(nfile)
        newhead['HIERARCH EDRS FLAT MEANEXPTIME'] = mexptime
        #newhead['EXPTIME'] = exptime
    
        outfile = '%s.fits'%flatname
        outpath = os.path.join(self.paths['midproc'], outfile)
        save_fits(outpath, data_mean, newhead)
        logger.info('Save combined flat image: "%s"'%outpath)
        print('save %s'%outfile)

        # save the mask for each individual flat frame
        mdata = np.int16(all_sat_mask)*4
        mtable = array_to_table(mdata)
        outfile = '%s%s.fits'%(flatname, self.mask_surfix)
        outpath = os.path.join(self.paths['midproc'], outfile)
        save_fits(outpath, mtable)
        logger.info('Save mask image for combined flat: "%s"'%outpath)

    def combine_trace(self):
        '''
        Combine trace frames.
        '''

        trace_file   = self.config.get('reduction', 'trace.file')

        string = self.config.get('reduction', 'fileid.trace')
        id_lst = parse_num_seq(string)

        nfile = len(id_lst)
    
        # create a header object for combined trace
        newhead = fits.Header()
        newhead['HIERARCH EDRS TRACE TYPE'] = 'sum'
        newhead['HIERARCH EDRS TRACE NFIEL'] = nfile
            
        # combine trace
        ifile = 0
        for i, item in self.log:
            if item.frameid not in id_lst:
                continue
            ifile += 1

            # read data file
            filename = '%s%s.fits'%(item.fileid, self.input_surfix)
            filepath = os.path.join(self.paths['midproc'], filename)
            data, head = fits.getdata(filepath, header=True)

            # read mask
            maskname = '%s%s.fits'%(item.fileid, self.mask_surfix)
            maskpath = os.path.join(self.paths['midproc'], maskname)
            mask = fits.getdata(maskpath)

            # if this file is the first one in the trace list, initialize
            # the summed image and the final mask
            if item.frameid == id_lst[0]:
                data_sum = np.zeros_like(data, dtype=np.float64)
                all_mask = np.zeros_like(mask, dtype=np.int16)
                total_exptime = 0.

            # combine data array, compute mask, and compute new exptime
            data_sum += data
            all_mask = (all_mask|mask)
            total_exptime += head['EXPTIME']

            # save the filename of each combined trace into header
            newhead['HIERARCH EDRS TRACE FILE %d'%ifile] = filename

        newhead['EXPTIME'] = total_exptime

        # save the final combined image
        save_fits(trace_file, data_sum, newhead)
        logger.info('Save trace image: "%s"'%trace_file)

    def flat2(self):
        '''
        Flat fielding correction
        '''

        if self.config.getboolean('reduction', 'flat.skip'):
            logger.info('Skip [flat] according to the config file')
            #self.input_surfix = self.output_surfix
            return True

        from ..echelle.flat import mosaic_flat

        flat_file = self.config.get('reduction','flat.file')
    
        if len(self.flat_groups) == 1:
    
            # only 1 type of flat
            flatname = self.flat_groups.keys()[0]
            infile  = os.path.join(self.paths['midproc'], '%s.fits'%flatname)
   
            if infile != flat_file:
                # copy the flat
                shutil.copyfile(infile, flat_file)
                logger.info('Copy "%s" to "%s" as flat'%(infile, flat_file))
            else:
                logger.info('Flat file: %s'%flat_file)
    
        elif len(self.flat_groups) > 1:
            # mosaic flat
            filename_lst = [os.path.join(self.paths['midproc'],
                                         '%s.fits'%flatname)
                            for flatname in sorted(self.flat_groups.keys())]
    
            # get filename from config file
            mosaicfile = self.config.get('reduction','flat.mosaic_file')
            regfile    = self.config.get('reduction','flat.mosaic_reg_file')
    
            # mosaic flats with different colors (different exposure times)
            mosaic_flat(filename_lst, flat_file, mosaicfile, regfile,
                        disp_axis   = 1,
                        mask_surfix = self.mask_surfix
                        )
    
            # write to the running log
            message = ['Mosaic flat images:']
            for filename in filename_lst:
                message.append('"%s"'%filename)
            message.append('Final flat image:     "%s"'%outfile)
            message.append('Mosaic boundary file: "%s"'%mosaicfile)
            message.append('Mosaic region   file: "%s"'%regfile)
            logger.info((os.linesep+'  ').join(message))

    def trace(self):
        '''
        Find the order locations.

        An ascii file in which the order locations are saved.
        The ascii file contains several keywords and their values, and the
        polynomial coefficients of each found order.

        .. csv-table:: Accepted options in config file
           :header: "Option", "Type", "Description"
           :widths: 20, 10, 50

           **trace.skip**,       *bool*,    "Skip this step if *yes* and **mode** = *'debug'*."
           **trace.file**,       *string*,  "Name of the trace file."
           **trace.scan_step**,  *integer*, "Steps of pixels used to scan along the main dispersion direction."
           **trace.minimum**,    *float*,   "Minimum value to filter the input image."
           **trace.seperation**, *float*,   "Estimated order seperations (in pixel) along the cross-dispersion."
           **trace.filling**,    *float*,   "Fraction of detected pixels to total step of scanning."
           **trace.display**,    *bool*,    "Display a figure on screen if *yes*."
           **trace.degree**,     *integer*, "Degree of polynomial used to describe the positions of orders."
        '''
        if self.config.getboolean('reduction', 'trace.skip'):
            logger.info('Skip [trace] according to the config file')
            return True

        from ..echelle.trace import find_apertures, load_aperture_set

        # find the parameters for order tracing
        kwargs = {
            'minimum'   : self.config.getfloat('reduction', 'trace.minimum'),
            'scan_step' : self.config.getint('reduction', 'trace.scan_step'),
            'seperation': self.config.getfloat('reduction', 'trace.seperation'),
            'filling'   : self.config.getfloat('reduction', 'trace.filling'),
            'display'   : self.config.getboolean('reduction', 'trace.display'),
            'degree'    : self.config.getint('reduction', 'trace.degree'),
            }

        trace_lst = self.find_trace()

        aperture_set_lst = {}

        for ichannel in range(self.nchannels):
            channel = chr(ichannel + 65)
            print(ichannel, channel)
            # initialize aperture_set_lst, containing the order locations from
            # different tracing files (either trace or flat image)
            aperture_set_lst[channel] = {}

            if channel in trace_lst:
                if len(trace_lst[channel]) > 1:
                    filename_lst = [os.path.join(self.paths['midproc'],
                                    '%s%s.fits'%(item.fileid, self.input_surfix))
                                    for item in trace_lst[channel]
                                    ]
                    tracename = 'trace_%s'%channel
                    dst_filename = os.path.join(self.paths['midproc'],
                                   '%s.fits'%tracename)
                    combine_fits(filename_lst, dst_filename, mode='sum')
                    data = fits.getdata(dst_filename)

                    mask = np.zeros_like(data, dtype=np.bool)
                    for item in trace_lst[channel]:
                        mask_file = os.path.join(
                                    self.paths['midproc'],
                                    '%s%s.fits'%(item.fileid, self.mask_surfix)
                                    )
                        mask_table = fits.getdata(mask_file)
                        imask = table_to_array(mask_table, data.shape)
                        imask = (imask&4 == 4)
                        mask = (mask|imask)
                else:
                    item = trace_lst[channel][0]
                    tracename = item.fileid
                    filename = os.path.join(
                                self.paths['midproc'],
                                '%s%s.fits'%(item.fileid, self.input_surfix)
                                )
                    data = fits.getdata(filename)
                    mask_file = os.path.join(
                                self.paths['midproc'],
                                '%s%s.fits'%(item.fileid, self.mask_surfix)
                                )
                    mask_table = fits.getdata(mask_file)
                    mask = table_to_array(mask_table, data.shape)
                    mask = (mask&4 == 4)

                trace_result_file = os.path.join(
                                        self.paths['midproc'],
                                        '%s_trc.txt'%tracename
                                    )
                reg_file = os.path.join(
                                self.paths['midproc'],
                                '%s_trc.reg'%tracename
                            )
                fig_file = os.path.join(
                            self.paths['report_img'],
                            'trace_%s.png'%tracename
                            )

                kwargs.update({'mask'       : mask,
                               'filename'   : trace_file,
                               'trace_file' : trace_result_file,
                               'reg_file'   : reg_file,
                               'fig_file'   : fig_file,
                               })
                aperture_set = find_apertures(data, **kwargs)

                logger.info('Found %d orders in "%s.fits"'%(
                            len(aperture_set), trace_file))

                aperture_set_lst[channel][tracename] = aperture_set

            else:
                # no trace file for this channel. use flat instead.
                logger.info('Cannot find trace images for channel %s. Use flat images instead'%channel)

                for flatname, item_lst in sorted(self.flat_groups[channel].items()):
                    logger.info('Begin processing flat component: %s'%flatname)
                    flatpath = os.path.join(
                                self.paths['midproc'],
                                '%s.fits'%flatname
                                )

                    # combine flat for this sub-group
                    self.combine_flat(item_lst, flatname)

                    data = fits.getdata(flatpath)
                    mask_file = os.path.join(
                                self.paths['midproc'],
                                '%s%s.fits'%(flatname, self.mask_surfix)
                                )
                    mask_table = fits.getdata(mask_file)
                    if mask_table.size==0:
                        mask = np.zeros_like(data, dtype=np.int16)
                    else:
                        mask = table_to_array(mask_table, data.shape)
                    mask = (mask&4 == 4)

                    # determine the result file and figure file
                    trace_result_file = os.path.join(
                                        self.paths['midproc'],
                                        '%s_trc.txt'%flatname
                                        )
                    reg_file = os.path.join(
                                self.paths['midproc'],
                                '%s_trc.reg'%flatname
                                )
                    fig_file = os.path.join(
                                self.paths['report_img'],
                                'trace_%s.png'%flatname
                                )
                    # find the apertures

                    kwargs.update({'mask'       : mask,
                                   'filename'   : flatpath,
                                   'trace_file' : trace_result_file,
                                   'reg_file'   : reg_file,
                                   'fig_file'   : fig_file,
                                   })
                    aperture_set = find_apertures(data, **kwargs)

                    logger.info('Found %d apertures in "%s.fits"'%(
                                len(aperture_set), flatname))

                    aperture_set_lst[channel][flatname] = aperture_set

        self.aperture_set_lst = aperture_set_lst

    def flat(self):
        '''
        Flat fielding correction.

        .. csv-table:: Accepted options in config file
           :header: "Option", "Type", "Description"
           :widths: 20, 10, 50

           **flat.skip**,            *bool*,    "Skip this step if *yes* and **mode** = *'debug'*."
           **flat.surfix**,          *string*,  "Surfix of the flat correceted files."
           **flat.cosmic_clip**,     *float*,   "Upper clipping threshold to remove cosmis-rays."
           **flat.file**,            *string*,  "Name of the trace file."
           **flat.mosaic_method**,   *string*,  "Method of mosaic."
           **flat.mosaic_file**,     *string*,  "Name of the mosaic file."
           **flat.mosaic_reg_file**, *string*,  "Name of the mosaic .reg file."
           **flat.mosaic_maxcount**, *integer*, "Maximum count of the flat mosaic."
        '''

        if self.config.getboolean('reduction', 'flat.skip'):
            logger.info('Skip [flat] according to the config file')
            #self.input_surfix = self.output_surfix
            return True

        flat_file = self.config.get('reduction','flat.file')

        if len(self.flat_groups) == 1:
    
            # only 1 type of flat
            flatname = self.flat_groups.keys()[0]
            infile  = os.path.join(self.paths['midproc'], '%s.fits'%flatname)
   
            if infile != flat_file:
                # copy the flat
                shutil.copyfile(infile, flat_file)
                logger.info('Copy "%s" to "%s" as flat'%(infile, flat_file))
            else:
                logger.info('Flat file: %s'%flat_file)

        elif len(self.flat_groups) > 1:
            # mosaic flat

            from ..echelle.flat import mosaic_flat_auto, mosaic_flat_interact

            # get filename from config file
            mosaic_method = self.config.get('reduction','flat.mosaic_method')
            mosaic_file   = self.config.get('reduction','flat.mosaic_file')
            reg_file      = self.config.get('reduction','flat.mosaic_reg_file')
            max_count     = self.config.getfloat('reduction','flat.mosaic_maxcount')

            # prepare input list
            filename_lst = [os.path.join(self.paths['midproc'],
                            '%s.fits'%flatname)
                            for flatname in sorted(self.flat_groups.keys())]

            if mosaic_method == 'interact':
                mosaic_flat_interact(filename_lst = filename_lst,
                                     outfile      = flat_file,
                                     mosaic_file  = mosaic_file,
                                     reg_file     = reg_file,
                                     disp_axis    = 0,
                                     mask_surfix  = self.mask_surfix,
                                     )
            elif mosaic_method == 'auto':
                mosaic_flat_auto(filename_lst  = filename_lst,
                                 outfile       = flat_file,
                                 order_set_lst = self.order_set_lst,
                                 max_count     = max_count,
                                 )
            else:
                logger.error('Unknown flat mosaic method: %s'%mosaic_method)

            # write to the running log
            message = ['Mosaic flat images:']
            for filename in filename_lst:
                message.append('"%s"'%filename)
            message.append('Final flat image:     "%s"'%flat_file)
            message.append('Mosaic boundary file: "%s"'%mosaic_file)
            message.append('Mosaic region   file: "%s"'%reg_file)
            logger.info((os.linesep+'  ').join(message))

    def trace2(self):
        '''
        Find the order locations of the input FITS image.
    
        '''

        from ..echelle.flat  import load_mosaic
    
        # find the parameters for order tracing
        trace_file   = self.config.get('reduction', 'trace.file')
        scan_step    = self.config.getint('reduction', 'trace.scan_step')
        threshold    = self.config.getfloat('reduction', 'trace.threshold')
        animation    = self.config.getboolean('reduction', 'trace.animation')
        display      = self.config.getboolean('reduction', 'trace.display')
        display_time = self.config.getfloat('reduction', 'trace.display_time')
        degree       = self.config.getint('reduction', 'trace.degree')
        clipping     = self.config.getfloat('reduction', 'trace.clipping')
        maxiter      = self.config.getint('reduction', 'trace.maxiter')
        trace_fig    = self.config.get('reduction', 'trace.fig')
    
        # final_order_coeff_lst contains all the coefficients of orders
        final_order_coeff_lst = []
        # order_souce_lst contains which file does the coeff come from,
        # and must have the same length with final_order_coeff_lst
        order_source_lst = []
        # display_data is an array used to plot in the trace_fig
        display_data = None
        # is_mosaic = True if mosaic
        is_mosaic = False
    
        if self.config.has_option('reduction', 'fileid.trace'):
            # if there's "fileid.trace" keyword in the "reduction" section of
            # config file, combine the file listed.
    
            string = self.config.get('reduction', 'fileid.trace')
            id_lst = parse_num_seq(string)
            
            nfile = len(id_lst)
    
            # create a header object for combined trace
            newhead = fits.Header()
            newhead['HIERARCH EDRS TRACE TYPE'] = 'sum'
            newhead['HIERARCH EDRS TRACE NFIEL'] = nfile
    
            # combine trace
            ifile = 0
            for i, item in log:
                if item.frameid not in id_lst:
                    continue
                ifile += 1
                basename = '%s%s.fits'%(item.fileid, self.input_surfix)
                filename = os.path.join(self.paths['midproc'], basename)
                data, head = fits.getdata(filename, header=True)
    
                # if this file is the first one in the trace list, initialize
                # the summed image
                if item.frameid == id_lst[0]:
                    data_sum = np.zeros_like(data, dtype=np.float64)
                    total_exptime = 0.
    
                # combine image
                data_sum += data
                total_exptime += head['EXPTIME']
    
                # save the filename of each combined trace into header
                newhead['HIERARCH EDRS TRACE FILE %d'%ifile] = basename
    
            newhead['EXPTIME'] = total_exptime
    
            # save the final combined image
            save_fits(trace_file, data_sum, newhead)
            logger.info('Save trace image: "%s"'%trace_file)
    
            # determine the node file
            basename = os.path.basename(trace_file)
            node_file = os.path.join(self.paths['midproc'],
                        '%s_trc_nodes.txt'%basename[0:-5])

            # find the orders
            coeff_lst = find_orders(data_sum,
                                    scan_step    = scan_step,
                                    threshold    = threshold,
                                    display      = display,
                                    animation    = animation,
                                    display_time = display_time,
                                    degree       = degree,
                                    clipping     = clipping,
                                    maxiter      = maxiter,
                                    node_file    = node_file,
                                    )
            for coeff in coeff_lst:
                final_order_coeff_lst.append(coeff)
                order_source_lst.append(trace_file)
            # prepare data for display
            display_data = data_sum
            is_mosaic = False

        else:
            # there's no "fileid.trace" keyword in config file, trace the
            # orders using flat file.

            logger.info('Cannot find trace images. Use flat images instead')

            # read flat data
            flat_file = self.config.get('reduction', 'flat.file')
            if not os.path.exists(flat_file):
                logger.error('flat must be done before tracing')
            flat_data = fits.getdata(flat_file)
            # prepare data for display
            display_data = flat_data
    
            # use flat image to trace the orders
            if len(self.flat_groups) == 1:
                # in the case that there are only 1 kind of flat
                logger.info('Find orders in "%s"'%flat_file)
                flat_data = fits.getdata(flat_file)

                # determine the node file
                basename = os.path.basename(flat_file)
                node_file = os.path.join(self.paths['midproc'],
                            '%s_trc_nodes.txt'%basename[0:-5])

                coeff_lst = find_orders(flat_data,
                                        scan_step    = scan_step,
                                        threshold    = threshold,
                                        display      = display,
                                        animation    = animation,
                                        display_time = display_time,
                                        degree       = degree,
                                        clipping     = clipping,
                                        maxiter      = maxiter,
                                        node_file    = node_file,
                                        )
                for coeff in coeff_lst:
                    final_order_coeff_lst.append(coeff)
                    order_source_lst.append(flat_file)
                is_mosaic = False
    
            else:
                # in the case that there are more than 1 kinds of flat

                # load boundaries and select areas information from mosaic file
                mosaic_file = self.config.get('reduction',
                                              'flat.mosaic_file')
    
                logger.info('Prepare for finding orders in multiple flats')
    
                bound_coeff_lst, select_area = load_mosaic(mosaic_file)
    
                # initialize coeff list for order locations for each file
                coeff_group = {}
                for flatname in self.flat_groups.keys():
                    filename = os.path.join(self.paths['midproc'],
                                            '%s.fits'%flatname)
                    # count the number of areas
                    if True not in select_area[filename]:
                        logger.info('Skip %s when finding orders'%flatname)
                        continue
                    data = fits.getdata(filename)
                    h, w = data.shape

                    # determine the result file
                    basename = os.path.basename(filename)
                    result_file = os.path.join(self.paths['midproc'],
                                '%s_trace.txt'%basename[0:-5])

                    coeff_lst = find_orders(data,
                                            scan_step    = scan_step,
                                            threshold    = threshold,
                                            display      = display,
                                            animation    = animation,
                                            display_time = display_time,
                                            degree       = degree,
                                            clipping     = clipping,
                                            maxiter      = maxiter,
                                            result_file  = result_file,
                                            )
                    logger.info('Found %d orders in "%s"'%(
                                 len(coeff_lst), filename))
                    basename = os.path.basename(filename)
                    coeff_group[basename] = coeff_lst
                
                # turn select_area into selec_file
                # select_area: ['1.fits':[1,0,0],
                #               '2.fits':[0,1,0],
                #               '3.fits':[0,0,1],]
                # select_file: ['1.fits', '2.fits', '3.fits']
                n_areas = len(select_area.values()[0])
                select_file = []
                for area in range(n_areas):
                    for filename, select in select_area.items():
                        if select[area]:
                            select_file.append(filename)
                            continue
    
                # mosaic the locations of orders
    
                x = np.arange(0, w, 100)
                boundary_line1 = np.repeat(-np.inf, x.size)
                for ifile, filename in enumerate(select_file):
                    if ifile <= len(bound_coeff_lst)-1:
                        boundary_coeff = bound_coeff_lst[ifile]
                        boundary_line2 = np.polyval(boundary_coeff,x)
                    elif ifile == len(bound_coeff_lst):
                        boundary_line2 = np.repeat(np.inf, x.size)
    
                    basename = os.path.basename(filename)
                    coeff_lst = coeff_group[basename]
    
                    # scan every order of a flat image
                    for o in sorted(coeff_lst.keys()):
                        print(coeff_lst[o])
                        coeff = coeff_lst[o]['center']
                        newy = np.polyval(coeff,np.float32(x)/w)*h
                        m1 = newy > boundary_line1
                        m2 = newy < boundary_line2
                        if (~m1).sum()==0 and (~m2).sum()==0:
                            final_order_coeff_lst.append(coeff)
                            order_source_lst.append(basename)
                            logger.info(
                                'Order %d from %s is selected as order %d'%(
                                o, filename, len(final_order_coeff_lst))
                                )
                        elif m2.sum()==0:
                            break
    
                    boundary_line1 = boundary_line2
                is_mosaic = True
   
        # now we have final_order_coeff_lst and order_source_lst, and
        # display_data

        fig = plt.figure(figsize=(7,7), dpi=150)
        ax = fig.gca()
        ax.imshow(display_data, cmap='gray')
        h, w = display_data.shape
        x = np.arange(w)
        colors = 'rgbcmyk'
        if is_mosaic:
            for coeff in bound_coeff_lst:
                y = np.polyval(coeff, x)
                ax.plot(x, y, '--', color='m')
        for i, coeff in enumerate(final_order_coeff_lst):
            basename = order_source_lst[i]
            color = colors[i%len(colors)]
            y = np.polyval(coeff, x)
            ax.plot(x, y, '-', color=color)
        ax.set_xlim(0, w-1)
        ax.set_ylim(0, h-1)
        fig.savefig(trace_fig)
        logger.info('Save the final figures in "%s"'%trace_fig)
        plt.close(fig)
    
        # open a text file for saving the polynomials of the order location
        order_loc_file = self.config.get('reduction','trace.location_file')
        outfile = open(order_loc_file, 'w')
        outfile.write('NAXIS1   = %d'%w+os.linesep)
        outfile.write('NAXIS2   = %d'%h+os.linesep)
        outfile.write('DISPAXIS = 1'+os.linesep)
        for i, coeff in enumerate(final_order_coeff_lst):
            # save the coefficients of the polynomial
            string = 'aperture "%s" '%order_source_lst[i]
            string_lst = ['%+15.10e'%c for c in coeff]
            string += ' '.join(string_lst)
            outfile.write(string + os.linesep)
        outfile.close()
        logger.info('Write order locations into "%s"'%order_loc_file)

    def background(self):
        '''
        Subtract the background for 2D images.
        '''

        from ..echelle.background import correct_background
        
        # find output surfix for fits
        self.output_surfix = self.config.get('reduction','background.surfix')

        if self.config.getboolean('reduction', 'background.skip'):
            logger.info('Skip [background] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        order_loc_file = self.config.get('reduction', 'trace.location_file')
        order_lst, info = load_order_locations(order_loc_file)
    
        display = self.config.getboolean('reduction', 'background.display')

        # get fitting parameters
        xorder  = self.config.getint('reduction', 'background.xorder')
        yorder  = self.config.getint('reduction', 'background.yorder')
        maxiter = self.config.getint('reduction', 'background.maxiter')
        upper_clipping = self.config.getfloat('reduction',
                                              'background.upper_clipping')
        lower_clipping = self.config.getfloat('reduction',
                                              'background.lower_clipping')
        expand_grid    = self.config.getboolean('reduction',
                                                'background.expand_grid')

        if display:
            fig1 = plt.figure(figsize=(12,6), dpi=150)
            ax11 = fig1.add_axes([0.10, 0.15, 0.35, 0.70])
            ax12 = fig1.add_axes([0.53, 0.15, 0.35, 0.70])
            ax13 = fig1.add_axes([0.92, 0.15, 0.02, 0.70])
    
            fig2 = plt.figure(figsize=(12,6), dpi=150)
            ax21 = fig2.add_subplot(121, projection='3d')
            ax22 = fig2.add_subplot(122, projection='3d')
    
            plt.show(block=False)
    
    
        # prepare the file queue
        infile_lst, mskfile_lst, outfile_lst, scafile_lst = [], [], [], []
        # different files use differenet scales
        scale_lst = []
    
        # check combined flat
        if self.config.has_option('reduction', 'fileid.flat'):
            # there only 1 flats in the dataset
            flat_file = self.config.get('reduction', 'flat.file')
            msk_file  = '%s%s.fits'%(flat_file[0:-5], self.mask_surfix)
            out_file  = '%s%s.fits'%(flat_file[0:-5], self.output_surfix)
            sca_file  = '%s_sca.fits'%(flat_file[0:-5])
            infile_lst.append(flat_file)
            mskfile_lst.append(msk_file)
            outfile_lst.append(out_file)
            scafile_lst.append(sca_file)
            scale_lst.append('linear')
            logger.info('Add "%s" to the background file queue'%flat_file)
    
        if len(self.flat_groups)>0:
            for flatname in sorted(self.flat_groups.keys()):
                infilename  = os.path.join(self.paths['midproc'],
                                '%s.fits'%flatname)
                mskfilename = os.path.join(self.paths['midproc'],
                                '%s%s.fits'%(flatname, self.mask_surfix))
                outfilename = os.path.join(self.paths['midproc'],
                                '%s%s.fits'%(flatname, self.output_surfix))
                scafilename = os.path.join(self.paths['midproc'],
                                '%s_sca.fits'%flatname)
                infile_lst.append(infilename)
                mskfile_lst.append(mskfilename)
                outfile_lst.append(outfilename)
                scafile_lst.append(scafilename)
                scale_lst.append('linear')
                logger.info('Add "%s" to the background file queue'%infilename)
    
        # check scientific files
        string = self.config.get('reduction', 'fileid.science')
        id_lst = parse_num_seq(string)
        for item in self.log:
            if item.frameid not in id_lst:
                continue
            infilename  = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.input_surfix))
            mskfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.mask_surfix))
            outfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.output_surfix))
            scafilename = os.path.join(self.paths['midproc'],
                            '%s_sca.fits'%item.fileid)
            infile_lst.append(infilename)
            mskfile_lst.append(mskfilename)
            outfile_lst.append(outfilename)
            scafile_lst.append(scafilename)
            scale_lst.append('linear')
            logger.info('Add scientific file "%s" to background file queue'%
                        infilename)
    
        scan_step = self.config.getint('reduction', 'background.scan_step')
    
        # correct the backgrounds
        for infile,mskfile,outfile,scafile,scale in zip(
                infile_lst,mskfile_lst,outfile_lst,scafile_lst,scale_lst):
            correct_background(infile, mskfile, outfile, scafile,
                               order_lst       = order_lst,
                               scale           = scale,
                               block_mask      = 4,
                               scan_step       = scan_step,
                               xorder          = xorder,
                               yorder          = yorder,
                               maxiter         = maxiter,
                               upper_clipping  = upper_clipping,
                               lower_clipping  = lower_clipping,
                               expand_grid     = expand_grid,
                               fig1            = fig1,
                               fig2            = fig2,
                               report_img_path = self.paths['report_img'],
                               )

        if display:
            # close the figures
            plt.close(fig1)
            plt.close(fig2)

        # update surfix
        self.input_surfix = self.output_surfix

    def extract(self):
        '''
        Extract 1d spectra
        '''
        # find output surfix for fits
        self.output_surfix = self.config.get('reduction', 'extract.surfix')
    
        if self.config.getboolean('reduction', 'extract.skip'):
            logger.info('Skip [extract] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        from ..echelle.extract import sum_extract

        order_loc_file = self.config.get('reduction', 'trace.location_file')
        order_lst, info = load_order_locations(order_loc_file)

        flat_file = self.config.get('reduction', 'flat.file')

        infile_lst, outfile_lst, mskfile_lst = [], [], []
    
        # check combined flat
        if self.config.has_option('reduction', 'fileid.flat'):
            infilename  = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.input_surfix))
            mskfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.mask_surfix))
            outfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.output_surfix))
            infile_lst.append(infilename)
            mskfile_lst.append(mskfilename)
            outfile_lst.append(outfilename)
            logger.info('Add "%s" to extraction queue'%infilename)
    
        if len(self.flat_groups)>0:
            for flatname in sorted(self.flat_groups.keys()):
                infilename  = os.path.join(self.paths['midproc'],
                              '%s%s.fits'%(flatname,self.input_surfix))
                mskfilename = os.path.join(self.paths['midproc'],
                              '%s%s.fits'%(flatname,self.mask_surfix))
                outfilename = os.path.join(self.paths['midproc'],
                             '%s%s.fits'%(flatname,self.output_surfix))
                infile_lst.append(infilename)
                mskfile_lst.append(mskfilename)
                outfile_lst.append(outfilename)
                logger.info('Add "%s" to extraction queue'%infilename)
        
            # add mosaiced flat file
            infilename = flat_file
            mskfilename = '%s%s.fits'%(flat_file[0:-5],self.mask_surfix)
            outfilename = '%s%s.fits'%(flat_file[0:-5],self.output_surfix)
            infile_lst.append(infilename)
            mskfile_lst.append(mskfilename)
            outfile_lst.append(outfilename)
            logger.info('Add "%s" to extraction queue'%infilename)

        # check comparison lamp
        if self.config.has_option('reduction','fileid.thar'):
            string = self.config.get('reduction','fileid.thar')
            id_lst = parse_num_seq(string)
            for item in self.log:
                if item.frameid not in id_lst:
                    continue
                infilename  = os.path.join(self.paths['midproc'],
                              '%s%s.fits'%(item.fileid,
                              self.config.get('reduction','bias.surfix')))
                mskfilename = os.path.join(self.paths['midproc'],
                              '%s%s.fits'%(item.fileid, self.mask_surfix))
                outfilename = os.path.join(self.paths['midproc'],
                              '%s%s.fits'%(item.fileid, self.output_surfix))
                infile_lst.append(infilename)
                mskfile_lst.append(mskfilename)
                outfile_lst.append(outfilename)
                logger.info('Add comparison file "%s" to extraction queue'%
                            infilename)
    
        # check scientific files
        string = self.config.get('reduction','fileid.science')
        id_lst = parse_num_seq(string)
        for item in self.log:
            if item.frameid not in id_lst:
                continue
            infilename  = os.path.join(self.paths['midproc'],
                                '%s%s.fits'%(item.fileid, self.input_surfix))
            mskfilename = os.path.join(self.paths['midproc'],
                                '%s%s.fits'%(item.fileid, self.mask_surfix))
            outfilename = os.path.join(self.paths['midproc'],
                                '%s%s.fits'%(item.fileid, self.output_surfix))
            infile_lst.append(infilename)
            mskfile_lst.append(mskfilename)
            outfile_lst.append(outfilename)
            logger.info('Add scientific file "%s" to extraction queue'%
                         infilename)
    
    
        # check whether to display the figure
        display = self.config.getboolean('reduction','extract.display')
        if display:
            fig = plt.figure(figsize=(10,8), dpi=150)
            ax = fig.gca()
            plt.show(block=False)
        else:
            fig = None
    
        for infile, mskfile, outfile in zip(infile_lst,
                                            mskfile_lst,
                                            outfile_lst):
            sum_extract(infile, mskfile, outfile,
                        order_lst=order_lst, figure=fig)
            # display information in screen
            print('extract {} -> {}'.format(os.path.basename(infile),
                                            os.path.basename(outfile)))

        # close the figure
        if display:
            plt.close(fig)

        # update surfix
        self.input_surfix = self.output_surfix

    def deblaze(self):
        '''
        Correct the blaze funtions of the 1-D spectra
        '''

        # find output surfix for fits
        self.output_surfix = self.config.get('reduction','deblaze.surfix')

        if self.config.getboolean('reduction', 'deblaze.skip'):
            logger.info('Skip [deblaze] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        flat_file = self.config.get('reduction','flat.file')
        flat_ext_file = '%s%s.fits'%(flat_file[0:-5], self.input_surfix)
        flatdata = fits.getdata(flat_ext_file)

        # extent the structrured array
        w = flatdata['points'].max()
        # generate the new numpy dtype for deblazed binary table
        newdescr = []
        # copy the original columns
        for descr in flatdata.dtype.descr:
            newdescr.append(descr)
        # add new columns
        #newdescr.append(('flat','>f4',(w,)))
        newdescr.append(('flux_deblazed','>f4',(w,)))
        # new newdescr can be used as the new dtype

        # find out science files
        string = self.config.get('reduction','fileid.science')
        id_lst = parse_num_seq(string)
        for item in self.log:
            if item.frameid not in id_lst:
                continue
            infilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.input_surfix))
            outfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.output_surfix))
            f = fits.open(infilename)
            head     = f[0].header
            fluxdata = f[1].data
            f.close()
            spec = []
            for o, row in enumerate(fluxdata):
                deblazed_flux = row['flux']/flatdata[o]['flux']
                item = list(row)
                item.append(deblazed_flux)
                item = np.array(tuple(item),dtype=newdescr)
                spec.append(item)
            spec = np.array(spec, dtype=newdescr)

            # save the data to fits 
            pri_hdu = fits.PrimaryHDU(header=head)
            tbl_hdu = fits.BinTableHDU(spec)
            hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
            if os.path.exists(outfilename):
                os.remove(outfilename)
            hdu_lst.writeto(outfilename)

        # update surfix
        self.input_surfix = self.output_surfix

    def wvcalib(self):
        '''
        Wavelength calibration
        '''

        # find output surfix for fits
        self.output_surfix = self.config.get('reduction','wvcalib.surfix')

        if self.config.getboolean('reduction', 'wvcalib.skip'):
            logger.info('Skip [wavcalib] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        from ..echelle.wvcalib import wvcalib, recalib, reference_wv, reference_wv_self

        kwargs = {
                'linelist': self.config.get('reduction', 'wvcalib.linelist'),
                'window_size': self.config.getint('reduction', 'wvcalib.window_size'),
                'xorder':   self.config.getint('reduction', 'wvcalib.xorder'), 
                'yorder':   self.config.getint('reduction', 'wvcalib.yorder'),
                'maxiter':  self.config.getint('reduction', 'wvcalib.maxiter'),
                'clipping': self.config.getfloat('reduction', 'wvcalib.clipping'),
                'snr_threshold': self.config.getfloat('reduction', 'wvcalib.snr_threshold'),
                'fig_width':  self.config.getint('reduction', 'wvcalib.fig_width'),
                'fig_height': self.config.getint('reduction', 'wvcalib.fig_height'),
                'fig_dpi':    self.config.getfloat('reduction', 'wvcalib.fig_dpi'),
                }

        if self.nchannels == 1:
            # single fiber calibration

            # check comparison lamp
            if self.config.has_option('reduction', 'fileid.thar'):
            
                # find comparison file list
                string = self.config.get('reduction', 'fileid.thar')
                id_lst = parse_num_seq(string)
            
                # find linelist
                linelistname = self.config.get('reduction', 'wvcalib.linelist')
            
                extract_surfix = self.config.get('reduction', 'extract.surfix')
            
                for item in self.log:
                    if item.frameid in id_lst:
                        infilename = os.path.join(self.paths['midproc'],
                                     '%s%s.fits'%(item.fileid, extract_surfix))
                        mskfilename = os.path.join(self.paths['midproc'],
                                      '%s%s.fits'%(item.fileid, self.mask_surfix))
                        coeff = wvcalib(infilename, linelistname)
                        break
            
                # find science file list
                string = self.config.get('reduction','fileid.science')
                sci_id_lst = parse_num_seq(string)
            
                for item in self.log:
                    if item.frameid in sci_id_lst:
                        infilename = os.path.join(self.paths['midproc'],
                                     '%s%s.fits'%(item.fileid, extract_surfix))
                        outfilename = os.path.join(self.paths['midproc'],
                                      '%s%s.fits'%(item.fileid, self.output_surfix))
                        reference_wv(infilename, outfilename, coeff)
        else:
            # multifiber calibration

            # loop all channels
            result_lst = {}

            thar_lst = {}
            for ichannel in range(self.nchannels):
                channel = chr(ichannel+65)
                thar_lst[channel] = [item for item in self.log
                                        if len(item.objectname) == self.nchannels and 
                                            item.objectname[ichannel] == 'ThAr']

            calib_lst = {}
            for ich, (channel, item_lst) in enumerate(sorted(thar_lst.items())):
                kwargs['channel'] = channel
                for i, item in enumerate(item_lst):
                    filename = os.path.join(self.paths['midproc'],
                               '%s%s.fits'%(item.fileid, self.input_surfix))
                    if ich == 0 and i == 0:
                        calib = wvcalib(filename, **kwargs)
                        ref_calib = calib
                        spec = fits.getdata(filename)
                        ref_spec = spec[spec['channel']==channel]
                    else:
                        calib = recalib(filename,
                                        channel       = channel,
                                        ref_spec      = ref_spec,
                                        linelist      = kwargs['linelist'],
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
                                        fig_width     = kwargs['fig_width'],
                                        fig_height    = kwargs['fig_height'],
                                        fig_dpi       = kwargs['fig_dpi'],
                                        )
                    if item.frameid not in calib_lst:
                        calib_lst[item.frameid] = {}
                    calib_lst[item.frameid][channel] = calib


            # find file to calibrate
            for item in self.log:
                if item.imagetype == 'sci':
                    infilename = os.path.join(self.paths['midproc'],
                                              '%s%s.fits'%(item.fileid, self.input_surfix))
                    outfilename = os.path.join(self.paths['midproc'],
                                              '%s%s.fits'%(item.fileid, self.output_surfix))
                    refcalib_lst = {}
                    # search for ref thar
                    for ichannel in range(len(item.objectname)):
                        channel = chr(ichannel+65)

                        refcalib_lst[channel] = []

                        for direction in [-1, +1]:
                            frameid = item.frameid
                            while(True):
                                frameid += direction
                                if frameid in calib_lst and channel in calib_lst[frameid]:
                                    refcalib_lst[channel].append(calib_lst[frameid][channel])
                                    print(item.frameid, 'append',channel, frameid)
                                    break
                                elif frameid <= min(calib_lst) or frameid >= max(calib_lst):
                                    break
                                else:
                                    continue
                    reference_wv(infilename, outfilename, refcalib_lst)


            for item in self.log:
                if item.frameid in calib_lst:
                    infilename = os.path.join(self.paths['midproc'],
                                              '%s%s.fits'%(item.fileid, self.input_surfix))
                    outfilename = os.path.join(self.paths['midproc'],
                                              '%s%s.fits'%(item.fileid, self.output_surfix))
                    reference_wv_self(infilename, outfilename, calib_lst[item.frameid])


