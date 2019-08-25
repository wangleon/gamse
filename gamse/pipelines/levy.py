import os
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time  import Time
import scipy.signal as sg
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from ..echelle.imageproc import combine_images, table_to_array, array_to_table
from ..echelle.trace import find_apertures, load_aperture_set, TraceFigureCommon
from ..echelle.flat import get_slit_flat
from ..echelle.extract import extract_aperset
from ..echelle.wlcalib import (wlcalib, recalib, select_calib_from_database, 
                               #self_reference_singlefiber,
                               #wl_reference,
                               get_time_weight)
from ..echelle.background import find_background
from ..utils.config import read_config
from ..utils.obslog import read_obslog
from .common import FormattedInfo

def correct_overscan(data, head):
    """Correct the overscan of CCD image.

    Args:
        data (:class:`numpy.dtype`): Input data image.
        head (:class:`astropy.io.fits.Header`): Input FITS header.

    Returns:
        tuple: A tuple containing:

            * **corrdata** (:class:`numpy.dtype`) – Output image with overscan 
              corrected.
            * **head** (:class:`astropy.io.fits.Header`) – Updated FITS header.
            * **overmean** (*float*) – Average of overscan values.
    """
    if data.shape==(4608, 2080):
        overmean = data[:,2049:2088].mean(axis=1)
        oversmooth = sg.savgol_filter(overmean, window_length=1201, polyorder=3)
        #coeff = np.polyfit(np.arange(overmean.size), overmean, deg=7)
        #oversmooth2 = np.polyval(coeff, np.arange(overmean.size))
        res = (overmean - oversmooth).std()
        #fig = plt.figure(dpi=150)
        #ax = fig.gca()
        #ax.plot(overmean)
        #ax.plot(oversmooth)
        #ax.plot(oversmooth2)
        #plt.show()
        #plt.close(fig)
        overdata = np.tile(oversmooth, (2048, 1)).T
        corrdata = data[:,0:2048] - overdata
        overmean = overdata.mean()

        # update fits header
        head['HIERARCH GAMSE OVERSCAN']        = True
        head['HIERARCH GAMSE OVERSCAN METHOD'] = 'smooth'
        head['HIERARCH GAMSE OVERSCAN AXIS-1'] = '2049:2088'
        head['HIERARCH GAMSE OVERSCAN AXIS-2'] = '0:4608'
        head['HIERARCH GAMSE OVERSCAN MEAN']   = overmean

        return corrdata, head, overmean

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
    """Reduce the APF/Levy spectra.
    """

    # read obs log
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
    config_file = os.path.join(config_path, 'Levy.cfg')
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
    section = config['reduce']
    midproc     = section.get('midproc')
    result      = section.get('result')
    report      = section.get('report')
    mode        = section.get('mode')
    fig_format  = section.get('fig_format')
    oned_suffix = section.get('oned_suffix')

    # create folders if not exist
    if not os.path.exists(report):  os.mkdir(report)
    if not os.path.exists(result):  os.mkdir(result)
    if not os.path.exists(midproc): os.mkdir(midproc)

    ################################ parse bias ################################
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        bias = fits.getdata(bias_file)
        message = 'Load bias from file: {}'.format(bias_file)
        logger.info(message)
        print(message)
    else:
        bias_data_lst = []

        # prepare print info
        columns = [
                ('fileid',   '{0:10s}', '{0.fileid:10s}'),
                ('exptime',  '{1:7s}',  '{0.exptime:7g}'),
                ('obsdate',  '{2:25s}', '{0.obsdate:25s}'),
                ('overscan', '{3:8s}',  '{1:8.2f}'),
                ('mean',     '{4:8s}',  '{2:8.2f}'),
                ]
        title, fmt_title, fmt_item = zip(*columns)
        fmt_title = ' '.join(fmt_title)
        fmt_item  = ' '.join(fmt_item)

        for logitem in logtable:
            if logitem['object'].strip()=='Bias' \
                and abs(logitem['exptime'])<1e-3:
                filename = os.path.join(rawdata,
                            '{}.fits'.format(logitem['fileid']))
                data, head = fits.getdata(filename, header=True)
                # correct overscan here
                data, head, overmean = correct_overscan(data, head)

                # print info
                if len(bias_data_lst) == 0:
                    print('* Combine Bias Images: {}'.format(bias_file))
                    print(' '*2 + fmt_title.format(*title))
                print(' '*2 + fmt_item.format(item, overmean, data.mean()))

                bias_data_lst.append(data)

        n_bias = len(bias_data_lst)         # number of bias images
        has_bias = n_bias > 0

        if has_bias:
            # there is bias frames

            # combine bias images
            bias_data_lst = np.array(bias_data_lst)

            section = config['reduce.bias']
            bias = combine_images(bias_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('cosmic_clip'),
                    maxiter    = section.getint('maxiter'),
                    mask       = (None, 'max')[n_bias>=3],
                    )

            # create new FITS Header for bias
            head = fits.Header()
            head['HIERARCH GAMSE BIAS NFILE'] = n_bias

            ############## bias smooth ##################
            if section.getboolean('smooth'):
                # bias needs to be smoothed
                smooth_method = section.get('smooth_method')

                if smooth_method in ['gauss','gaussian']:
                    # perform 2D gaussian smoothing
                    smooth_sigma = section.getint('smooth_sigma')
                    smooth_mode  = section.get('smooth_mode')

                    bias_smooth = gaussian_filter(bias,
                                    sigma=smooth_sigma, mode=smooth_mode)

                    # write information to FITS header
                    head['HIERARCH GAMSE BIAS SMOOTH']        = True
                    head['HIERARCH GAMSE BIAS SMOOTH METHOD'] = 'GAUSSIAN'
                    head['HIERARCH GAMSE BIAS SMOOTH SIGMA']  = smooth_sigma
                    head['HIERARCH GAMSE BIAS SMOOTH MODE']   = smooth_mode
                else:
                    print('Unknown smooth method: ', smooth_method)
                    pass

                bias = bias_smooth
            else:
                # bias not smoothed
                head['HIERARCH GAMSE BIAS SMOOTH'] = False

            fits.writeto(bias_file, bias, header=head, overwrite=True)
            logger.info('Bias image written to "%s"'%bias_file)

        else:
            # no bias found
            pass

    ########################### find & trace the orders ######################
    section = config['reduce.trace']
    trace_file = section['trace_file']

    if mode=='debug' and os.path.exists(trace_file):
        # load trace image from existing file
        has_trace = True
        trace = fits.getdata(trace_file)
    else:
        # combine trace file from narrow flats
        trace_data_lst = []

        # prepare print info
        columns = [
                ('fileid',   '{0:10s}', '{0.fileid:10s}'),
                ('exptime',  '{1:7s}',  '{0.exptime:7g}'),
                ('obsdate',  '{2:25s}', '{0.obsdate:25s}'),
                ('overscan', '{3:8s}',  '{1:8.2f}'),
                ('mean',     '{4:8s}',  '{2:8.2f}'),
                ]
        title, fmt_title, fmt_item = zip(*columns)
        fmt_title = ' '.join(fmt_title)
        fmt_item  = ' '.join(fmt_item)

        for logitem in logtable:
            if logitem['object'].strip()=='NarrowFlat':
                filename = os.path.join(rawdata,
                            '{}.fits'.format(item['fileid']))
                data, head = fits.getdata(filename, header=True)
                data, head, overmean = correct_overscan(data, head)
                if has_bias:
                    data = data - bias

                # print info
                if len(trace_data_lst) == 0:
                    print('* Combine Images for Order Tracing: %s'%trace_file)
                    print(' '*2 + fmt_title.format(*title))
                print(' '*2 + fmt_item.format(item, overmean, data.mean()))

                trace_data_lst.append(data)

        n_trace = len(trace_data_lst)  # number of trace images
        has_trace = n_trace > 0

        if has_trace:
            # there is trace frames

            # combine trace images
            trace_data_lst = np.array(trace_data_lst)

            trace = combine_images(trace_data_lst,
                    mode       = 'mean',
                    upper_clip = section.getfloat('upper_clip'),
                    maxiter    = section.getint('maxiter'),
                    mask       = (None, 'max')[n_trace>=3],
                    )
            trace = trace.T
            fits.writeto(trace_file, trace, overwrite=True)

        else:
            # no trace image found
            pass

    # find the name of .trc file
    trc_file = '.'.join(trace_file.split('.')[:-1])+'.trc'
    trc_reg  = '.'.join(trace_file.split('.')[:-1])+'.reg'
    trace_filename = os.path.basename(trace_file)
    trace_fileid = '.'.join(trace_filename.split('.')[:-1])

    if os.path.exists(trc_file):
        # load apertures from existing file
        aperset = load_aperture_set(trc_file)
    else:
        mask = np.zeros_like(trace, dtype=np.int8)

        # create the trace figure
        tracefig = TraceFigure()

        aperset = find_apertures(trace, mask,
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
        figfile = os.path.join(report,
                    'trace_{}.{}'.format(flatname, fig_format))
        tracefig.savefig(figfile)

        aperset.save_txt(trc_file)
        aperset.save_reg(trc_reg)

    ######################### find flat groups #################################
    ########################### Combine flat images ############################
    flat_groups = {}
    for logitem in logtable:
        if logitem['objectname']=='WideFlat':
            flatname = 'flat_{%d}'.format(item['exptime'])
            if flatname not in flat_groups:
                flat_groups[flatname] = []
            flat_groups[flatname].append(logitem['fileid'])
    # print how many flats in each flat name
    for flatname in flat_groups:
        n = len(flat_groups[flatname])
        print('{} images in {}'.format(n, flatname))

    flat_data_lst = {}
    flat_mask_lst = {}
    for flatname, fileids in flat_groups.items():
        flat_filename = '{}.fits'.format(flatname)
        mask_filename = '%s_msk.fits'%flatname
        if os.path.exists(flat_filename) and os.path.exists(mask_filename):
            flat_data = fits.getdata(flat_filename)
            mask_table = fits.getdata(mask_filename)
            mask_array = table_to_array(mask_table, flat_data.shape)
        else:
            data_lst = []
            for ifile, fileid in enumerate(fileids):
                filename = os.path.join(rawdata, '%s.fits'%fileid)
                data, head = fits.getdata(filename, header=True)
                mask = (data[:,0:2048]==65535)
                if ifile==0:
                    allmask = np.zeros_like(mask, dtype=np.int16)
                allmask += mask
                data, head = correct_overscan(data, head)
                data = data - bias
                data_lst.append(data)
            nflat = len(data_lst)
            print('combine images for', flatname)
            flat_data = combine_images(data_lst, mode='mean',
                        upper_clip=10, maxiter=5)
            fits.writeto(flat_filename, flat_data, overwrite=True)
            
            sat_mask = allmask>nflat/2.
            mask_array = np.int16(sat_mask)*4
            mask_table = array_to_table(mask_array)
            fits.writeto(mask_filename, mask_table, overwrite=True)
        flat_data_lst[flatname] = flat_data
        flat_mask_lst[flatname] = mask_array

    ######################### Extract flat spectrum ############################
    flatmap_lst = {}
    flat_spectra1d_lst = {}
    for flatname in flat_groups:
        resp_filename = '%s_resp.fits'%flatname
        data = flat_data_lst[flatname]
        mask = flat_mask_lst[flatname]
        spectra1d = extract_aperset(data.T, mask.T,
                        apertureset = aperset,
                        lower_limit = 5,
                        upper_limit = 5,
                        )
        flat_spectra1d_lst[flatname] = spectra1d

        if os.path.exists(resp_filename):
            flatmap = fits.getdata(resp_filename)
        else:
            flatmap = get_slit_flat(data.T, mask.T,
                            apertureset = aperset,
                            spectra1d   = spectra1d,
                            lower_limit = 6,
                            upper_limit = 5,
                            deg         = 7,
                            q_threshold = 20**2,
                            )
            fits.writeto(resp_filename, flatmap, overwrite=True)
        flatmap_lst[flatname] = flatmap


    ############################ Mosaic flats ##################################
    mosaic_resp_filename = 'flat_resp.fits'
    if os.path.exists(mosaic_resp_filename):
        flatmap = fits.getdata(mosaic_resp_filename)
    else:
        mosaic_mask_lst = {flatname:np.zeros_like(flat_data_lst[flatname].T,dtype=np.bool)
                           for flatname in flat_groups}
        maxcount = 55000
        h, w = flat_data_lst[list(flat_groups.keys())[0]].T.shape
        yy, xx = np.mgrid[:h:,:w:]
        for iaper, (aper, aper_loc) in enumerate(sorted(aperset.items())):
    
            # find the maximum count and its belonging flatname of this aperture
            cmax = -999
            maxflatname = None
            for flatname in flat_groups:
                nsat = flat_spectra1d_lst[flatname][aper]['mask_sat'].sum()
                cmaxi = np.sort(flat_spectra1d_lst[flatname][aper]['flux_mean'])[-10]
                if nsat > 0 or cmaxi > maxcount:
                    continue
                if cmaxi > cmax:
                    cmax = cmaxi
                    maxflatname = flatname
    
            print(aper, maxflatname, cmax)
            #domain = aper_loc.position.domain
            #d1, d2 = int(domain[0]), int(domain[1])+1
            #newx = np.arange(d1, d2)
            newx = np.arange(aper_loc.shape[1])
            position = aper_loc.position(newx)
            if iaper==0:
                mosaic_mask_lst[maxflatname][:,:] = True
            else:
                boundary = (position + prev_position)/2
                _m = yy > boundary
                for flatname in flat_groups:
                    if flatname == maxflatname:
                        mosaic_mask_lst[flatname][_m] = True
                    else:
                        mosaic_mask_lst[flatname][_m] = False
            prev_position = position
    
        flatname0 = list(flat_groups.keys())[0]
        flatmap0 = flatmap_lst[flatname0]
        flatmap = np.zeros_like(flatmap0, dtype=flatmap0.dtype)
        for flatname, maskdata in mosaic_mask_lst.items():
            #fits.writeto('mask_%s.fits'%flatname, np.int16(maskdata), overwrite=True)
            flatmap += flatmap_lst[flatname]*maskdata
        fits.writeto(mosaic_resp_filename, flatmap, overwrite=True)

    ############################# extract ThAr #################################
    h, w = bias.shape
    spectype = np.dtype({
                'names':  ('aperture', 'order', 'points', 'wavelength', 'flux'),
                'formats':('i',       'i',     'i',      '(%d,)f8'%h,  '(%d,)f'%h),
                })

    if True:
        calib_lst = {}
        count_thar = 0
        for item in log:
            if item.objectname[0]=='ThAr':
                count_thar += 1
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                data, head = fits.getdata(filename, header=True)
                mask = np.int16(data == 65535)*4
                data, head, overmean = correct_overscan(data, head)
                spectra1d = extract_aperset(data.T, mask.T,
                                apertureset = aperset,
                                lower_limit = 5,
                                upper_limit = 5,
                                )
                head = aperset.to_fitsheader(head, channel=None)
    
                spec = [(aper, 0, item['flux_sum'].size,
                        np.zeros_like(item['flux_sum'].size, dtype=np.float64),
                        item['flux_sum'])
                        for aper, item in sorted(spectra1d.items())]
                spec = np.array(spec, dtype=spectype)
                
                if count_thar == 1:
                    ref_spec, ref_calib, ref_aperset = select_calib_from_database('Levy',
                                                        'DATE-OBS',
                                                        head['DATE-OBS'],
                                                        channel=None)
    
                if ref_spec is None or ref_calib is None:
                    calib = wvcalib(spec,
                                    filename      = '%s.fits'%item.fileid,
                                    identfilename = 'a.idt',
                                    figfilename   = 'wvcalib_%s.png'%item.fileid,
                                    channel       = None,
                                    linelist      = 'thar.dat',
                                    window_size   = 13,
                                    xorder        = 5,
                                    yorder        = 4,
                                    maxiter       = 10,
                                    clipping      = 3,
                                    snr_threshold = 10,
                                    )
                    aper_offset = 0
                else:
                    aper_offset = ref_aperset.find_aper_offset(aperset)
                    calib = recalib(spec,
                                    filename      = '%s.fits'%item.fileid,
                                    figfilename   = 'wvcalib_%s.png'%item.fileid,
                                    ref_spec      = ref_spec,
                                    channel       = None,
                                    linelist      = 'thar.dat',
                                    identfilename = '',
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
    
                if count_thar == 1:
                    ref_calib = calib
                    ref_spec  = spec
                    aper_offset = 0
    
                hdu_lst = self_reference_singlefiber(spec, head, calib)
                hdu_lst.writeto('%s_wlc.fits'%item.fileid, overwrite=True)

                # add more infos in calib
                calib['fileid']   = item.fileid
                calib['date-obs'] = head['DATE-OBS']
                calib['exptime']  = head['EXPTIME']
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

    ###################### Extract science spectra #############################
    for item in log:
        if item.imagetype=='sci':
            filename = os.path.join(rawdata, '%s.fits'%item.fileid)
            data, head = fits.getdata(filename, header=True)
            mask = np.int16(data == 65535)*4

            data, head, overmean = correct_overscan(data, head)

            # write order locations to header
            head = aperset.to_fitsheader(head, channel=None)

            # flat fielding correction
            data = data.T/flatmap
            mask = mask.T

            # background correction
            stray = find_background(data, mask,
                        channels        = ['A'],
                        apertureset_lst = {'A': aperset},
                        scale           = 'log',
                        block_mask      = 4,
                        scan_step       = 200,
                        xorder          = 2,
                        yorder          = 3,
                        maxiter         = 5,
                        upper_clip      = 3,
                        lower_clip      = 3,
                        extend          = True,
                        display         = True,
                        fig_file        = 'background_%s.png'%item.fileid,
                        )
            data = data - stray
            # 1d spectra extraction
            spectra1d = extract_aperset(data, mask,
                        apertureset = aperset,
                        lower_limit = 6,
                        upper_limit = 5,
                        )
            spec = [(aper, 0, item['flux_sum'].size,
                    np.zeros_like(item['flux_sum'].size, dtype=np.float64),
                    item['flux_sum'])
                    for aper, item in sorted(spectra1d.items())]
            spec = np.array(spec, dtype=spectype)

            weight_lst = get_time_weight(ref_datetime_lst, head['DATE-OBS'])
            spec, head = wv_reference_singlefiber(spec, head,
                            ref_calib_lst, weight_lst)

            # pack and save wavelength referenced spectra
            pri_hdu = fits.PrimaryHDU(header=head)
            tbl_hdu = fits.BinTableHDU(spec)
            hdu_lst = fits.HDUList([pri_hdu, tbl_hdu])
            hdu_lst.writeto('%s_wlc.fits'%item.fileid, overwrite=True)

all_columns = [
        ('frameid', 'int',   '{:^7s}',  '{0[frameid]:7d}'),
        ('fileid',  'str',   '{:^12s}', '{0[fileid]:12s}'),
        ('imgtype', 'str',   '{:^7s}',  '{0[imgtype]:^7s}'),
        ('object',  'str',   '{:^12s}', '{0[object]:12s}'),
        ('exptime', 'float', '{:^7s}',  '{0[exptime]:7g}'),
        ('obsdate', 'time',  '{:^23s}', '{0[obsdate]:}'),
        ('nsat',    'int',   '{:^7s}',  '{0[nsat]:7d}'),
        ('q95',     'int',   '{:^6s}',  '{0[q95]:6d}'),
        ]

def make_obslog(path):
    """Print the observing log.

    Args:
        path (str): Path to the raw FITS files.
    """
    cal_objects = ['bias', 'wideflat', 'narrowflat', 'flat', 'dark', 'iodine',
                    'thar']

    # prepare logtable
    logtable = Table(dtype=[
        ('frameid', 'i2'),  ('fileid', 'S12'),  ('imgtype', 'S3'),
        ('object',  'S12'), ('i2',     'bool'), ('exptime', 'f4'),
        ('obsdate', Time),  ('nsat',   'i4'),   ('q95',     'i4'),
        ])

    # prepare infomation to print
    pinfo = FormattedInfo(all_columns,
            ['frameid', 'fileid', 'imgtype', 'object', 'exptime', 'obsdate',
             'nsat', 'q95'])

    # print header of logtable
    print(pinfo.get_separator())
    print(pinfo.get_title())
    print(pinfo.get_separator())

    # start scanning the raw files
    for fname in sorted(os.listdir(path)):
        if fname[-5:] != '.fits':
            continue
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)

        fileid     = fname[0:-5]
        obstype    = head['OBSTYPE']
        exptime    = head['EXPTIME']
        objectname = head['OBJECT']
        obsdate    = Time(head['DATE-OBS'])
        i2cell     = {'In': 1, 'Out': 0}[head['ICELNAM']]

        imgtype = ('sci', 'cal')[objectname.lower().strip() in cal_objects]

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        item = [0, fileid, imgtype, obstype, objectname, i2cell, exptime,
                obsdate, saturation, quantile95]
        logtable.add_row(item)

        item = logtable[-1]

        # print log item with colors

    logtable.sort('obsdate')

    # allocate frameid
    prev_frameid = -1
    for item in logtable:
        frameid = int(item['fileid'][-4:])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(
                    frameid, prev_frameid))

        item['frameid'] = frameid

        prev_frameid = frameid

    # determine filename of logtable.
    # use the obsdate of the first frame
    obsdate = logtable[0]['obsdate'][0:10]
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
    logtable.write(outfilename, format='ascii.fixed_width_two_line')

    return True
