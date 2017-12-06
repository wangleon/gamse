import os
import datetime
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..ccdproc      import save_fits
from ..utils.obslog import parse_num_seq
from .reduction     import Reduction

class XinglongHRS(Reduction):

    def __init__(self):
        super(XinglongHRS, self).__init__()

        # copy some keywords from the log to the Reduction instance
        self.nchannels = self.log.nchannels

    def overscan(self):
        '''
        '''

        from scipy.signal import savgol_filter
        from scipy.interpolate import InterpolatedUnivariateSpline

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
        
        # find output surfix for fits
        self.output_surfix = self.config.get('reduction', 'overscan.surfix')

        if self.config.getboolean('reduction', 'overscan.skip'):
            logger.info('Skip [overscan] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        # keywords for mask
        saturation_adu = self.config.getint('reduction','saturation')

        # loop over all files (bias, dark, ThAr, flat...)
        # to correct for the overscan

        count = 0
        for item in self.log:
            logger.info('Correct overscan for item %3d: "%s"'%(
                         item.frameid, item.fileid))

            # read in of the data
            filename = '%s%s.fits'%(item.fileid, self.input_surfix)
            fname = os.path.join(self.paths['data'], filename)
            data, head = fits.getdata(fname, header=True)

            h, w = data.shape
            x1, x2 = w-head['COVER'], w

            # find the overscan level along the y-axis
            ovr_lst1 = data[0:int(h/2),x1+2:x2].mean(dtype=np.float64, axis=1)
            ovr_lst2 = data[int(h/2):h,x1+2:x2].mean(dtype=np.float64, axis=1)

            ovr_lst1_fix = fix_cr(ovr_lst1)
            ovr_lst2_fix = fix_cr(ovr_lst2)

            ovrsmooth1 = savgol_filter(ovr_lst1_fix, window_length=201, polyorder=3)
            ovrsmooth2 = savgol_filter(ovr_lst2_fix, window_length=201, polyorder=3)

            # plot the overscan regions
            if count%5 == 0:
                fig = plt.figure(figsize=(10,6), dpi=150)

            ax1 = fig.add_axes([0.08, 0.83-(count%5)*0.185, 0.42, 0.15])
            ax2 = fig.add_axes([0.55, 0.83-(count%5)*0.185, 0.42, 0.15])

            ax1.plot([0,0],[ovr_lst1_fix.min(), ovr_lst1_fix.max()], 'w-', alpha=0)
            y11, y12 = ax1.get_ylim()
            ax1.plot(np.arange(0, h/2), ovr_lst1, 'r-', alpha=0.3)
            ax1.set_ylim(y11, y12)

            ax2.plot([0,0],[ovr_lst2_fix.min(), ovr_lst2_fix.max()], 'w-', alpha=0)
            y21, y22 = ax2.get_ylim()
            ax2.plot(np.arange(h/2, h), ovr_lst2, 'b-', alpha=0.3)
            ax2.set_ylim(y21, y22)

            ax1.plot(np.arange(0, h/2), ovrsmooth1, 'm', ls='-')
            ax2.plot(np.arange(h/2, h), ovrsmooth2, 'c', ls='-')
            ax1.set_ylabel('ADU')
            ax2.set_ylabel('')
            ax1.set_xlim(0, h/2)
            ax2.set_xlim(h/2, h)
            for ax in [ax1, ax2]:
                x1,x2 = ax.get_xlim()
                y1,y2 = ax.get_ylim()
                ax.text(0.95*x1+0.05*x2, 0.2*y1+0.8*y2, item.fileid, fontsize=9)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.xaxis.set_major_locator(tck.MultipleLocator(500))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            if count%5==4:
                ax1.set_xlabel('Y (pixel)')
                ax2.set_xlabel('Y (pixel)')
                figpath = os.path.join(self.paths['report_img'],
                                       'overscan_%02d.png'%(int(count/5)+1))
                fig.savefig(figpath)
                logger.info('Save image: %s'%figpath)
                plt.close(fig)

            y1 = head['CRVAL2']
            y2 = y1 + head['NAXIS2'] - head['ROVER']
            ymid = int((y1 + y2)/2)
            x1 = head['CRVAL1']
            x2 = x1 + head['NAXIS1'] - head['COVER']
            mask_sat   = (data[y1:y2,x1:x2]>=saturation_adu)
            mask       = np.int16(mask_sat)
            mask_fname = os.path.join(self.paths['midproc'],
                         '%s%s.fits'%(item.fileid, self.mask_surfix))

            save_fits(mask_fname, mask*4)

            # subtract overscan
            new_data = np.zeros((y2-y1, x2-x1), dtype=np.float64)
            ovrdata1 = np.transpose(np.repeat([ovrsmooth1],x2-x1,axis=0))
            ovrdata2 = np.transpose(np.repeat([ovrsmooth2],x2-x1,axis=0))
            new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
            new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2

            # update fits header
            head['HIERARCH EDRS OVERSCAN']        = True
            head['HIERARCH EDRS OVERSCAN METHOD'] = 'smooth'

            # save data
            newfilename = '%s%s.fits'%(item.fileid, self.output_surfix)
            newpath = os.path.join(self.paths['midproc'], newfilename)
            save_fits(newpath, new_data, head)
            print('Correct Overscan {} -> {}'.format(filename, newfilename))

            count += 1

        logger.info('Overscan corrected. Change surfix: %s -> %s'%
                    (self.input_surfix, self.output_surfix))
        self.input_surfix = self.output_surfix

    def bias(self):
        self.output_surfix = self.config.get('reduction','bias.surfix')

        if self.config.getboolean('reduction', 'bias.skip'):
            logger.info('Skip [bias] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        bias_id_lst = parse_num_seq(self.config.get('reduction','fileid.bias'))
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
        head['HIERARCH EDRS BIAS NFILE'] = len(bias_id_lst)

        # get final bias filename from the config file
        bias_file = self.config.get('reduction', 'bias.file')

        if self.config.has_option('reduction', 'bias.smooth_method'):
            # perform smoothing for bias
            smooth_method = self.config.get('reduction', 'bias.smooth_method')
            smooth_method = smooth_method.strip().lower()

            logger.info('Smoothing bias: %s'%smooth_method)

            if smooth_method in ['gauss','gaussian']:
                # perform 2D gaussian smoothing

                smooth_sigma = self.config.getint('reduction',
                                                  'bias.smooth_sigma')
                smooth_mode  = self.config.get('reduction',
                                               'bias.smooth_mode')

                logger.info('Smoothing bias: sigma = %f'%smooth_sigma)
                logger.info('Smoothing bias: mode = %s'%smooth_mode)

                from scipy.ndimage.filters import gaussian_filter
                h, w = bias.shape
                bias_smooth = np.zeros((h, w), dtype=np.float64)
                bias_smooth[0:h/2,:] = gaussian_filter(bias[0:h/2,:],
                                                       smooth_sigma,
                                                       mode=smooth_mode)
                bias_smooth[h/2:h,:] = gaussian_filter(bias[h/2:h,:],
                                                       smooth_sigma,
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
        
        self.plot_bias_variation(all_data, all_head, time_key='DATE-STA')

        # finally all files are corrected for the bias
        for item in self.log:
            if item.frameid not in bias_id_lst:
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


    def trace2(self):
        
        if self.config.getboolean('reduction', 'trace.skip'):
            logger.info('Skip [trace] according to the config file')
            return True

        # find the parameters for order tracing
        trace_file   = self.config.get('reduction', 'trace.file')
        scale        = self.config.get('reduction', 'trace.scale')
        scan_step    = self.config.getint('reduction', 'trace.scan_step')
        minimum      = self.config.getfloat('reduction', 'trace.minimum')
        animation    = self.config.getboolean('reduction', 'trace.animation')
        degree       = self.config.getint('reduction', 'trace.degree')
        clipping     = self.config.getfloat('reduction', 'trace.clipping')
        maxiter      = self.config.getint('reduction', 'trace.maxiter')
        saturation   = self.config.getint('reduction', 'saturation')

        find_orders('midproc/flat_red.fits',
                    scale     = scale,
                    scan_step = scan_step,
                    animation = animation,
                    minimum   = minimum,
                    )


def find_orders(filename, **kwargs):
    import astropy.io.fits as fits
    from scipy.ndimage.filters import median_filter
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt

    from edrs.utils.onedarray import get_local_minima

    scale        = kwargs.pop('scale', 'log')
    animation    = kwargs.pop('animation', True)
    scan_step    = kwargs.pop('scan_step', 50)
    minimum      = kwargs.pop('minimum', 10)

    data, head = fits.getdata(filename, header=True)
    h,w = data.shape

    data0 = data

    #neg_mask = data0 < minimum
    #data0[neg_mask] = minimum
    #data1 = np.log10(data0)
    data1 = data0

    if animation:
        fig = plt.figure(figsize=(16,8), dpi=150)
        ax1 = fig.add_axes([0.05,0.06,0.43,0.86])
        ax2 = fig.add_axes([0.54,0.06,0.43,0.86])
        ax1.imshow(data1,cmap='gray')

        fig.suptitle('Trace for %s'%os.path.basename(filename))
        fig.canvas.draw()
        plt.show(block=False)

    # scan iterval in pixels
    dx = int(scan_step)
    x_lst = np.arange(dx,w,dx)
    
    smooth_window = np.hanning(31)
    smooth_window /= smooth_window.sum()
    
    for ix, xnode in enumerate(x_lst):
        # calculate the median of the nearest 3 columns
        flux0 = np.median(data1[:,xnode-1:xnode+2], axis=1)

        flux1 = savgol_filter(flux0, window_length=5, polyorder=2)
        flux1 = np.convolve(flux1, smooth_window, mode='same')

        xmax, ymax = get_local_minima2(-flux1, 25)
        ymax = -ymax

        minimask = ymax > minimum
        xmax = xmax[minimask]
        ymax = ymax[minimask]

        if animation:
            ax1.plot(np.repeat(xnode,len(xmax)), xmax, 'wo',ms=3,markeredgewidth=1)
            ax1.set_xlim(0, w-1)
            ax1.set_ylim(h-1, 0)
            
            ax2.cla()
           
            if scale == 'linear':
                xsection = flux1
            elif scale == 'log':
                #xsection = np.power(10,flux1)
                xsection = flux1

            ax2.plot(xsection,'k-',alpha=0.3)
            ax2.scatter(xmax, ymax, c='b',s=5)
            ax2.set_yscale('log')
            ax2.set_xlim(0, h-1)
            fig.canvas.draw()

def get_local_minima2(x, window):
    x = np.array(x)
    dif = np.diff(x)
    ind = dif > 0
    tmp = np.logical_xor(ind, np.roll(ind,1))
    idx = np.logical_and(tmp,ind)
    index = np.where(idx)[0]

 
    return index_lst, x[index_lst]


def make_log(rawdata_path):
    '''

    '''
    from ..utils import obslog

    # scan the raw files
    fname_lst = sorted(os.listdir(rawdata_path))
    log = []
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid  = fname[0:-5]
        obsdate = None
        exptime = None
        filepath = os.path.join(rawdata_path,fname)
        data,head = fits.getdata(filepath,header=True)
        naxis1 = head['NAXIS1']
        cover  = head['COVER']
        y1 = head['CRVAL2']
        y2 = y1 + head['NAXIS2'] - head['ROVER']
        x1 = head['CRVAL1']
        x2 = x1 + head['NAXIS1'] - head['COVER']
        data = data[y1:y2,x1:x2]
        obsdate = head['DATE-STA']
        exptime = head['EXPTIME']
        objtname = head['OBJECT']

        # determine the fraction of saturated pixels permillage
        mask_sat = (data>=65535)
        prop = float(mask_sat.sum())/data.size*1e3

        # find the brightness index in the central region
        h,w = data.shape
        data1 = data[int(h*0.3):int(h*0.7),int(w/2)-2:int(w/2)+3]
        bri_index = np.median(data1,axis=1).mean()

        logitem = obslog.LogItem(
                      fileid     = fileid,
                      obsdate    = obsdate,
                      exptime    = exptime,
                      objtname   = objtname,
                      saturation = prop,
                      brightness = bri_index,
                  )
        log.append(logitem)

    log = obslog.sort_log(log, 'obsdate')

    # make info list
    all_info_lst = []
    columns = ['frameid','fileid','objtname','exptime','obsdate','saturation',
               'brightness']
    prev_frameid = -1
    for logitem in log:
        frameid = int(logitem.fileid[8:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(frameid, prev_frameid))
        info_lst = [
                    str(frameid),
                    str(logitem.fileid),
                    str(logitem.objtname),
                    '%8.3f'%logitem.exptime,
                    str(logitem.obsdate),
                    '%.3f'%logitem.saturation,
                    '%.1f'%logitem.brightness,
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
    date = log[0].fileid.split('_')[0]
    outfilename = '%s-%s-%s.log'%(date[0:4],date[4:6],date[6:8])
    outfile = open(outfilename,'w')
    string = '# '+', '.join(columns)
    outfile.write(string+os.linesep)
    print(string)
    for info_lst in all_info_lst:
        string = ' | '.join(info_lst)
        string = ' '+string
        outfile.write(string+os.linesep)
        print(string)
    outfile.close()