import os
import re
import shutil
import datetime
import logging
logger = logging.getLogger(__name__)
import dateutil.parser

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.dates  as mdates

from ...echelle.imageproc import combine_images
from ...echelle.trace import TraceFigureCommon
from ...echelle.background import BackgroundFigureCommon
from ..reduction     import Reduction

def correct_overscan(data, mask=None, direction=None):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.
        mask (:class:`numpy.ndarray`): Input image mask.
        direction (str): CCD direction code.
    
    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.
            * **card_lst** (*list*) – A new card list for FITS header.
            * **overmean** (*float*) – Mean value of overscan pixels.
            * **overstd** (*float*) – Standard deviation of overscan pixels.
    """
    ny, nx = data.shape
    overdata1 = data[:, 0:20]
    overdata2 = data[:, nx-18:nx]
    overdata_tot = np.hstack((overdata1, overdata2))

    # find the overscan level along the y axis
    # 1 for the left region, 2 for the right region
    # calculate the mean of overscan region along x direction
    ovr_lst1 = overdata1.mean(dtype=np.float64, axis=1)
    ovr_lst2 = overdata2.mean(dtype=np.float64, axis=1)
    
    # only used the bluer ~1/2 regions for calculating mean of overscan
    if direction[1]=='b':
        vy1, vy2 = 0, ny//2
    elif direction[1]=='r':
        vy1, vy2 = ny//2, ny
    else:
        print('Unknown direction:', direction)
        raise ValueError

    # find the mean and standard deviation for left & right overscan
    '''
    ovrmean1 = ovr_lst1[vy1:vy2].mean(dtype=np.float64)
    ovrmean2 = ovr_lst2[vy1:vy2].mean(dtype=np.float64)
    ovrstd1  = ovr_lst1[vy1:vy2].std(dtype=np.float64, ddof=1)
    ovrstd2  = ovr_lst2[vy1:vy2].std(dtype=np.float64, ddof=1)
    '''

    ovrmean1 = data[vy1:vy2, 0:20].mean(dtype=np.float64)
    ovrstd1  = data[vy1:vy2, 0:20].std(dtype=np.float64)
    # subtract overscan
    new_data = data[:, 20:2068] - ovrmean1
    
    card_lst = []
    prefix = 'HIERARCH GAMSE OVERSCAN '
    card_lst.append((prefix + 'CORRECTED', True))
    card_lst.append((prefix + 'METHOD',    'mean'))
    card_lst.append((prefix + 'AXIS1',     '1:20'))
    card_lst.append((prefix + 'AXIS2',     '{}:{}'.format(vy1+1,vy2)))
    card_lst.append((prefix + 'MEAN',      ovrmean1))
    card_lst.append((prefix + 'STDEV',     ovrstd1))

    return new_data, card_lst, ovrmean1, ovrstd1

def get_bias(config, logtable):
    """Get bias image.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) – Output bias image.
            * **bias_card_lst** (list) – List of FITS header cards related to
              the bias correction.

    """
    mode = config['reduce'].get('mode')
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        bias = hdu_lst[-1].data
        head = hdu_lst[0].header
        hdu_lst.close()

        reobj = re.compile('GAMSE BIAS[\s\S]*')
        # filter header cards that match the above pattern
        bias_card_lst = [(card.keyword, card.value) for card in head.cards
                            if reobj.match(card.keyword)]

        n_bias = head['GAMSE BIAS NFILE']
        bias_ovrstd_lst = [head['GAMSE BIAS FILE {:03d} OVERSCAN STDEV'.format(i+1)]
                            for i in range(n_bias)]
        ovrstd = np.mean(bias_ovrstd_lst)
        # print info
        message = 'Load bias from image: "{}"'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        bias, bias_card_lst, n_bias, ovrstd = combine_bias(config, logtable)

    return bias, bias_card_lst, n_bias, ovrstd

def combine_bias(config, logtable):
    """Combine bias images.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:
        tuple: A tuple containing:

            * **bias** (:class:`numpy.ndarray`) – Output bias image.
            * **bias_card_lst** (list) – List of FITS header cards related to
              the bias correction.
    
    Combine bias frames found in observing log.
    The resulting array **bias** is combined using sigma-clipping method with
    an uppper clipping value given by "cosmic_clip" in "reduce.bias" section in
    **config**.
    Meanwhile, a card list containing the method, mean value and standard
    deviation to be added to the FITS header is also returned.

    """
    rawdata   = config['data']['rawdata']
    direction = config['data']['direction']
    section = config['reduce.bias']
    bias_file = section['bias_file']

    # read each individual CCD
    bias_data_lst = []
    bias_card_lst = []
    bias_overstd_lst = []

    bias_items = list(filter(lambda item: item['object'].lower()=='bias',
                             logtable))
    # get the number of bias images
    n_bias = len(bias_items)

    if n_bias == 0:
        # there is no bias frames
        return None, [], 0, 0.0

    for ifile, logitem in enumerate(bias_items):

        # now filter the bias frames
        filename = os.path.join(rawdata, logitem['fileid']+'.fits')
        data, head = fits.getdata(filename, header=True)
        if data.ndim == 3:
            data = data[0,:,:]
        mask = get_mask(data)
        data, card_lst, overmean, overstd = correct_overscan(
                                                data, mask, direction)
        bias_overstd_lst.append(overstd)
        # head['BLANK'] is only valid for integer arrays.
        if 'BLANK' in head:
            del head['BLANK']

        # pack the data and fileid list
        bias_data_lst.append(data)

        # append the file information
        prefix = 'HIERARCH GAMSE BIAS FILE {:03d}'.format(ifile+1)
        card = (prefix+' FILEID', logitem['fileid'])
        bias_card_lst.append(card)

        # append the overscan information of each bias frame to
        # bias_card_lst
        for keyword, value in card_lst:
            mobj = re.match('^HIERARCH GAMSE (OVERSCAN[\s\S]*)', keyword)
            if mobj:
                newkey = prefix + ' ' + mobj.group(1)
                bias_card_lst.append((newkey, value))

        # print info
        if ifile == 0:
            print('* Combine Bias Images: "{}"'.format(bias_file))
        message_lst = [
                '  - FileID: {}'.format(logitem['fileid']),
                'exptime = {:<5g}'.format(logitem['exptime']),
                'mean = {:<7.2f}'.format(overmean),
                ]
        print('    '.join(message_lst))

    prefix = 'HIERARCH GAMSE BIAS '
    bias_card_lst.append((prefix + 'NFILE', n_bias))

    # combine bias images
    bias_data_lst = np.array(bias_data_lst)

    combine_mode = 'mean'
    cosmic_clip  = section.getfloat('cosmic_clip')
    maxiter      = section.getint('maxiter')
    maskmode     = (None, 'max')[n_bias>=3]

    bias_combine = combine_images(bias_data_lst,
            mode       = combine_mode,
            upper_clip = cosmic_clip,
            maxiter    = maxiter,
            maskmode   = maskmode,
            )

    bias_card_lst.append((prefix+'COMBINE_MODE', combine_mode))
    bias_card_lst.append((prefix+'COSMIC_CLIP',  cosmic_clip))
    bias_card_lst.append((prefix+'MAXITER',      maxiter))
    bias_card_lst.append((prefix+'MASK_MODE',    str(maskmode)))

    # create the hdu list to be saved
    hdu_lst = fits.HDUList()
    # create new FITS Header for bias
    head = fits.Header()
    for card in bias_card_lst:
        head.append(card)
    hdu_lst.append(fits.PrimaryHDU(data=bias_combine, header=head))

    ############## bias smooth ##################
    if section.getboolean('smooth'):
        # bias needs to be smoothed
        smooth_method = section.get('smooth_method')

        newcard_lst = []
        if smooth_method in ['gauss', 'gaussian']:
            # perform 2D gaussian smoothing
            smooth_sigma = section.getint('smooth_sigma')
            smooth_mode  = section.get('smooth_mode')

            bias_smooth = gaussian_filter(bias_combine,
                            sigma=smooth_sigma, mode=smooth_mode)

            # write information to FITS header
            bias_card_lst.append((prefix+'SMOOTH CORRECTED', True))
            bias_card_lst.append((prefix+'SMOOTH METHOD', 'GAUSSIAN'))
            bias_card_lst.append((prefix+'SMOOTH SIGMA',  smooth_sigma))
            bias_card_lst.append((prefix+'SMOOTH MODE',   smooth_mode))
        else:
            print('Unknown smooth method: ', smooth_method)
            pass

        # pack the cards to bias_card_lst and also hdu_lst
        for card in newcard_lst:
            bias_card_lst.append(card)
            hdu_lst[0].header.append(card)
        hdu_lst.append(fits.ImageHDU(data=bias_smooth))

        # bias is the result array to return
        bias = bias_smooth
    else:
        # bias not smoothed
        card = (prefix+'SMOOTH CORRECTED', False)
        bias_card_lst.append(card)
        hdu_lst[0].header.append(card)

        # bias is the result array to return
        bias = bias_combine

    hdu_lst.writeto(bias_file, overwrite=True)

    message = 'Bias image written to "{}"'.format(bias_file)
    logger.info(message)
    print(message)

    # calculate average overstd values for all bias frames
    bias_meanstd = np.mean(bias_overstd_lst)

    return bias, bias_card_lst, n_bias, bias_meanstd


def get_mask(data):
    """Get the mask of input image.

    Args:
        data (:class:`numpy.ndarray`): Input image data.

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

def print_wrapper(string, item):
    """A wrapper for log printing for FOCES pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype    = item['imgtype']
    objectname = item['object'].strip().lower()

    if imgtype=='cal' and objectname=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='cal':
        if objectname == 'thar':
            # arc lamp, use light yellow (93)
            return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
        else:
            return string
        #elif (item['fiber_A'], item['fiber_B']) in [('ThAr', ''),
        #                                            ('', 'ThAr'),
        #                                            ('ThAr', 'ThAr')]:
        #    # arc lamp, use light yellow (93)
        #    return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
        #else:
        #    return string
    else:
        return string


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

class BackgroudFigure(BackgroundFigureCommon):
    """Figure to plot the background correction.
    """
    def __init__(self):
        BackgroundFigureCommon.__init__(self, figsize=(16, 7), dpi=150)
        _width = 0.37
        _height = _width*16/7
        self.ax1  = self.add_axes([0.06, 0.1, _width, _height])
        self.ax2  = self.add_axes([0.55, 0.1, _width, _height])
        self.ax1c = self.add_axes([0.06+_width+0.01, 0.1, 0.015, _height])
        self.ax2c = self.add_axes([0.55+_width+0.01, 0.1, 0.015, _height])

    def plot_background(self, data, stray):
        # find the minimum and maximum value of plotting
        #s = np.sort(oridata.flatten())
        #vmin = s[int(0.05*data.size)]
        #vmax = s[int(0.95*data.size)]
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)

        cax_data  = self.ax1.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
        cax_stray = self.ax2.imshow(stray, cmap='viridis')
        cs = self.ax2.contour(stray, colors='r', linewidths=0.5)
        self.ax2.clabel(cs, inline=1, fontsize=9, use_clabeltext=True)
        self.colorbar(cax_data, cax=self.ax1c)
        self.colorbar(cax_stray, cax=self.ax2c)
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.yaxis.set_major_locator(tck.MultipleLocator(500))
            ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
