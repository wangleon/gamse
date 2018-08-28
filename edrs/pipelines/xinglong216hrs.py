import os
import datetime
import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils import obslog
from ..utils.config import read_config
from ..echelle.imageproc import (combine_images, array_to_table,
                                 table_to_array, fix_pixels)
from ..echelle.trace import find_apertures, load_aperture_set
from ..echelle.flat  import get_fiber_flat, mosaic_flat_auto, mosaic_images
from ..echelle.extract import extract_aperset

from .reduction          import Reduction

def get_badpixel_mask(shape, bins):
    '''Get the mask of bad pixels and columns.

    Args:
        shape (tuple): Shape of image.
        bins (tuple): CCD bins.
    Returns:
        :class:`numpy.ndarray`: 2D binary mask, where bad pixels are marked with
            *True*, others *False*.

    The bad pixels are found *empirically*.
        
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

def get_mask(data, head):
    '''Get the mask of input image.

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

    '''

    saturation_adu = 65535

    # determine shape of output image (also the shape of science region)
    y1 = head['CRVAL2']
    y2 = y1 + head['NAXIS2'] - head['ROVER']
    x1 = head['CRVAL1']
    x2 = x1 + head['NAXIS1'] - head['COVER']
    newshape = (y2-y1, x2-x1)

    # find the saturation mask
    mask_sat = (data[y1:y2, x1:x2]>=saturation_adu)
    # get bad pixel mask
    bins = (head['RBIN'], head['CBIN'])
    mask_bad = get_badpixel_mask(newshape, bins=bins)

    mask = np.int16(mask_sat)*4 + np.int16(mask_bad)*2

    return mask


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
    # define the cosmic ray fixing function
    def fix_cr(data):
        m = data.mean(dtype=np.float64)
        s = data.std(dtype=np.float64)
        _mask = data > m + 3.*s
        if _mask.sum()>0:
            x = np.arange(data.size)
            f = InterpolatedUnivariateSpline(x[~_mask], data[~_mask], k=3)
            return f(x)
        else:
            return data

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

    # determine shape of output image (also the shape of science region)
    y1 = head['CRVAL2']
    y2 = y1 + head['NAXIS2'] - head['ROVER']
    ymid = (y1 + y2)//2
    x1 = head['CRVAL1']
    x2 = x1 + head['NAXIS1'] - head['COVER']
    newshape = (y2-y1, x2-x1)

    # subtract overscan
    new_data = np.zeros(newshape, dtype=np.float64)
    ovrdata1 = np.repeat([ovrsmooth1],x2-x1,axis=0).T
    ovrdata2 = np.repeat([ovrsmooth2],x2-x1,axis=0).T
    new_data[y1:ymid, x1:x2] = data[y1:ymid,x1:x2] - ovrdata1
    new_data[ymid:y2, x1:x2] = data[ymid:y2,x1:x2] - ovrdata2

    if mask is not None:
        # fix bad pixels
        bad_mask = (mask&2 > 0)
        new_data = fix_pixels(new_data, bad_mask, 'x', 'linear')

    # update fits header
    head['HIERARCH EDRS OVERSCAN']        = True
    head['HIERARCH EDRS OVERSCAN METHOD'] = 'smooth'

    return new_data, head

def reduce():
    '''Reduce the Xinglong 216/HRS spectra.

    Args:
        No args.

    Returns:
        No returns.
    '''

    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    config = read_config('Xinglong216HRS')

    rawdata = config['data']['rawdata']

    ############################# parse bias ###################################
    if os.path.exists('bias.fits'):
        bias = fits.getdata('bias.fits')
    else:
        bias_lst = []
        for item in log:
            if item.objectname[0].strip().lower()=='bias':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                mask = get_mask(data, head)
                data, head = correct_overscan(data, head, mask)
                bias_lst.append(data)

        if len(bias_lst)>0:
            # there is bias frames
            if not config.has_section('bias'):
                print('Warning: Configuration file has no bias section')
                exit()

            bias = combine_images(bias_lst,
                                    mode       = 'mean',
                                    upper_clip = config['bias']['cosmic_clip'],
                                    maxiter    = 5,
                                    )

            # create new FITS Header for bias
            head = fits.Header()
            head['HIERARCH EDRS BIAS NFILE'] = len(bias_id_lst)

            if config.has_option('bias', 'smooth_method'):
                # perform smoothing for bias

                h, w = bias.shape
                smooth_method = config['bias']['smooth_method']
                if smooth_method in ['gauss','gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = int(config['bias']['smooth_sigma'])
                    smooth_mode  = config['bias']['smooth_mode']
                    bias_smooth = np.zeros((h, w), dtype=np.float64)
                    bias_smooth[0:h/2, :] = gaussian_filter(bias[0:h/2, :],
                                                            smooth_sigma,
                                                            mode=smooth_mode)
                    bias_smooth[h/2:h, :] = gaussian_filter(bias[h/2:h, :],
                                                            smooth_sigma,
                                                            mode=smooth_mode)
                    # write information to FITS header
                    head['HIERARCH EDRS BIAS SMOOTH']        = True
                    head['HIERARCH EDRS BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                    head['HIERARCH EDRS BIAS SMOOTH SIGMA']  = smooth_sigma
                    head['HIERARCH EDRS BIAS SMOOTH MODE']   = smooth_mode
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth

            fits.writeto('bias.fits', bias, header=head, overwrite=True)
        else:
            # no bias in this dataset
            bias = 0

    ######################### find flat groups##################################
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
        flat_filename = '%s.fits'%flatname
        mask_filename = '%s_msk.fits'%flatname
        aperset_filename = 'trace_%s.trc'%flatname
        aperset_regname  = 'trace_%s.reg'%flatname

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
                mask = get_mask(data, head)
                sat_mask = (mask&4>0)
                bad_mask = (mask&2>0)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += sat_mask
                data, head = correct_overscan(data, head, mask)
                data = data - bias
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

            aperset = find_apertures(flat_data, mask_array,
                        scan_step  = config['trace'].getint('scan_step'),
                        minimum    = config['trace'].getfloat('minimum'),
                        seperation = config['trace'].getfloat('seperation'),
                        sep_der    = config['trace'].getfloat('sep_der'),
                        filling    = config['trace'].getfloat('filling'),
                        degree     = config['trace'].getint('degree'),
                        display    = config['trace'].getboolean('display'),
                        filename   = flat_filename,
                        fig_file   = 'trace_%s.png'%flatname,
                        )
            aperset.save_txt('trace_%s.trc'%flatname)
            aperset.save_reg('trace_%s.reg'%flatname)

        # append the flat data and mask
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = mask_array
        aperset_lst[flatname]   = aperset

    ########################### Get flat fielding ##############################
    flatmap_lst = {}
    for flatname in sorted(flat_groups.keys()):
        flatmap_filename = '%s_rsp.fits'%flatname
        if os.path.exists(flatmap_filename):
            flatmap = fits.getdata(flatmap_filename)
        else:
            flatmap = get_fiber_flat(
                        data        = flat_data_lst[flatname],
                        mask        = flat_mask_lst[flatname],
                        apertureset = aperset_lst[flatname],
                        slit_step   = config['flat'].getint('slit_step'),
                        nflat       = len(flat_groups[flatname]),
                        q_threshold = config['flat'].getfloat('q_threshold'),
                        param_deg   = config['flat'].getint('param_deg'),
                        fig_aperpar = None,
                        fig_overlap = None,
                        fig_slit    = '%s_slit.png'%flatname,
                        slit_file   = None,
                        )
        
            # save flat result to fits file
            fits.writeto(flatmap_filename, flatmap, overwrite=True)

        # append the flatmap
        flatmap_lst[flatname] = flatmap

    ############################# Mosaic Flats #################################
    if len(flat_groups) == 1:
        flatname = flat_groups.keys()[0]
        shutil.copyfile('%s.fits'%flatname, 'flat.fits')
        shutil.copyfile('%s_msk.fits'%flatname, 'flat_msk.fits')
        shutil.copyfile('trace_%s.trc', 'trace.trc')
        shutil.copyfile('trace_%s.reg', 'trace.reg')
        shutil.copyfile('%s_rsp.fits'%flatname, 'flat_rsp.fits')
        flat_map = fits.getdata('flat_rsp.fits')
    else:
        # mosaic apertures
        mosaic_aperset = mosaic_flat_auto(
                aperture_set_lst = aperset_lst,
                max_count        = config['flat'].getfloat('mosaic_maxcount'),
                )
        # mosaic original flat images
        flat_data = mosaic_images(flat_data_lst, mosaic_aperset)
        fits.writeto('flat.fits', flat_data, overwrite=True)

        # mosaic flat mask images
        mask_data = mosaic_images(flat_mask_lst, mosaic_aperset)
        mask_table = array_to_table(mask_data)
        fits.writeto('flat_msk.fits', mask_table, overwrite=True)

        # mosaic sensitivity map
        flat_map = mosaic_images(flatmap_lst, mosaic_aperset)
        fits.writeto('flat_rsp.fits', flat_map, overwrite=True)

        mosaic_aperset.save_txt('trace.trc')
        mosaic_aperset.save_reg('trace.reg')

    ############################## Extract ThAr ################################

    # find a file and get the data shape
    filename = os.path.join(rawdata, '%s.fits'%log.item_list[0].fileid)
    data = fits.getdata(filename)
    h, w = data.shape

    # define dtype of 1-d spectra
    names = [
            ('aperture',   np.int16),
            ('order',      np.int16),
            ('points',     np.int16),
            ('wavelength', (np.float64, w)),
            ('flux',       (np.float32, w)),
            ]
    _names, _formats = list(zip(*types))
    spectype = np.dtype({'names': _names, 'formats'})

    calib_lst = {}
    count_thar = 0
    for item in log:
        if item.objectname[0]=='ThAr':
            count_thar += 1
            filename = os.path.join(rawdaa, '%s.fits'%item.fileid)
            data, head = fits.getdata(filename, header=True)
            mask = get_mask(data, head)
            data, head = correct_overscan(data, head, mask)
            data = data - bias
            spectra1d = extract_aperset(data, mask,
                        apertureset = mosaic_aperset,
                        lower_limit = 5,
                        upper_limit = 5,
                        )
            head = mosaic_aperset.to_fitsheader(head, channel=None)
            spec = [(aper, 0, item['flux_sum'].size,
                     np.zeros_lke(item['flux_sum'].size, dtype=np.float64),
                     iteme['flux_sum'])
                     for aper, item in sorted(spectra1d.items())]
            spec = np.array(spec, dtype=spectype)



    
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

           **skip**,    *bool*,   Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,  *string*, Suffix of the corrected files.
           **plot**,    *bool*,   Plot the overscan levels if *yes*.
           **var_fig**, *string*, Filename of the overscan variation figure.

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
            head['HIERARCH EDRS OVERSCAN']        = True
            head['HIERARCH EDRS OVERSCAN METHOD'] = 'smooth'

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

           **skip**,        *bool*,   Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,      *string*, Suffix of the corrected files.
           **cosmic_clip**, *float*,  Upper clipping threshold to remove cosmic-rays.
           **file**,        *string*, Name of bias file.

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
        head['HIERARCH EDRS BIAS NFILE'] = len(bias_id_lst)

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
        return True

def make_log(path):
    '''
    Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (string): Path to the raw FITS files.

    '''

    # scan the raw files
    fname_lst = sorted(os.listdir(path))
    log = obslog.Log()
    for fname in fname_lst:
        if fname[-5:] != '.fits':
            continue
        fileid  = fname[0:-5]
        obsdate = None
        exptime = None
        filepath = os.path.join(path,fname)
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
        objectname = head['OBJECT']
        if objectname.lower().strip() in ['bias', 'flat', 'dark', 'i2', 'thar']:
            imagetype = 'cal'
        else:
            imagetype = 'sci'

        # determine the fraction of saturated pixels permillage
        mask_sat = (data>=65535)
        prop = float(mask_sat.sum())/data.size*1e3

        # find the brightness index in the central region
        h, w = data.shape
        data1 = data[int(h*0.3):int(h*0.7),int(w/2)-2:int(w/2)+3]
        bri_index = np.median(data1,axis=1).mean()

        item = obslog.LogItem(
                   fileid     = fileid,
                   obsdate    = obsdate,
                   exptime    = exptime,
                   imagetype  = imagetype,
                   i2         = 0,
                   objectname = objectname,
                   saturation = prop,
                   brightness = bri_index,
                   )
        log.add_item(item)

    log.sort('obsdate')

    # make info list
    all_info_lst = []
    columns = ['frameid (i)', 'fileid (s)', 'imagetype (s)', 'objectname (s)',
                'i2 (i)', 'exptime (f)', 'obsdate (s)', 'saturation (f)',
                'brightness (f)']
    prev_frameid = -1
    for logitem in log:
        frameid = int(logitem.fileid[8:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(frameid, prev_frameid))
        info_lst = [
                    str(frameid),
                    str(logitem.fileid),
                    logitem.imagetype,
                    str(logitem.objectname),
                    str(logitem.i2),
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
    #date = log[0].fileid.split('_')[0]
    #outfilename = '%s-%s-%s.log'%(date[0:4],date[4:6],date[6:8])
    #outfile = open(outfilename,'w')
    string = '% columns = '+', '.join(columns)
    #outfile.write(string+os.linesep)
    print(string)
    for info_lst in all_info_lst:
        string = ' | '.join(info_lst)
        string = ' '+string
        #outfile.write(string+os.linesep)
        print(string)
    #outfile.close()
