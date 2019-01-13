import os
import re
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import astropy.io.fits as fits

from ..echelle.imageproc import combine_images
from ..utils import obslog

def make_log(path):
    '''
    Scan the raw data, and generated a log file containing the detail
    information for each frame.

    An ascii file will be generated after running. The name of the ascii file is
    `YYYY-MM-DD.log`.

    Args:
        path (str): Path to the raw FITS files.

    '''

    # scan the raw files
    fname_lst = sorted(os.listdir(path))
    log = obslog.Log()
    for fname in fname_lst:
        if not re.match('HI\.\d{8}\.\d{5}\.fits', fname):
            continue
        fileid = fname[0:17]
        filepath = os.path.join(path, fname)
        hdu_lst = fits.open(filepath)
        head0 = hdu_lst[0].header

        # get obsdate in 'YYYY-MM-DDTHH:MM:SS' format
        date = head0.get('DATE-OBS')
        utc  = head0.get('UTC', head0.get('UT'))
        obsdate = '%sT%s'%(date, utc)

        exptime    = head0.get('ELAPTIME')
        i2in       = head0.get('IODIN', False)
        i2out      = head0.get('IODOUT', True)
        objectname = head0.get('TARGNAME', '')
        _type      = head0.get('IMAGETYP')
        if _type.strip() == 'object':
            imagetype = 'sci'
        else:
            imagetype = 'cal'
            objectname = _type
        # get deck and filter information
        deckname = head0.get('DECKNAME', '')
        filter1  = head0.get('FIL1NAME', '')
        filter2  = head0.get('FIL2NAME', '')

        data1 = hdu_lst[1].data
        data2 = hdu_lst[2].data
        data3 = hdu_lst[3].data

        # determine the fraction of saturated pixels permillage
        mask_sat1 = (data1==0)
        mask_sat2 = (data2==0)
        mask_sat3 = (data3==0)
        prop = (mask_sat1.sum() + mask_sat2.sum() + mask_sat3.sum())/(
                data1.size + data2.size + data3.size)*1e3

        # find the brightness index in the central region
        h, w = data2.shape
        d = data2[h//2-2:h//2+3, int(w*0.2):int(w*0.8)]
        bri_index = np.median(d, axis=1).mean()

        hdu_lst.close()

        item = obslog.LogItem(
                   fileid     = fileid,
                   obsdate    = obsdate,
                   exptime    = exptime,
                   imagetype  = imagetype,
                   i2         = i2in,
                   objectname = objectname,
                   deckname   = deckname,
                   filter1    = filter1,
                   filter2    = filter2,
                   saturation = prop,
                   brightness = bri_index,
                   )
        log.add_item(item)

    log.sort('obsdate')

    # make info list
    all_info_lst = []
    column_lst = [('frameid',    'i'), ('fileid',     's'), ('imagetype',  's'),
                  ('objectname', 's'), ('i2',         'i'), ('exptime',    'f'),
                  ('obsdate',    's'), ('deckname',   's'), ('filter1',    's'),
                  ('filter2',    's'), ('saturation', 'f'), ('brightness', 'f'),
                 ]
    columns = ['%s (%s)'%(_name, _type) for _name, _type in column_lst]

    prev_frameid = -1
    for logitem in log:
        frameid = prev_frameid + 1
        info_lst = [
                    str(frameid),
                    str(logitem.fileid),
                    logitem.imagetype,
                    str(logitem.objectname),
                    '%1d'%logitem.i2,
                    '%g'%logitem.exptime,
                    str(logitem.obsdate),
                    '%2s'%logitem.deckname,
                    '%5s'%logitem.filter1,
                    '%5s'%logitem.filter2,
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
            if columns[i] in ['fileid (s)','objectname (s)']:
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


def reduce():
    '''
    2D to 1D pipeline for Keck/HIRES
    '''
    # read obs log
    obslogfile = obslog.find_log(os.curdir)
    log = obslog.read_log(obslogfile)

    # load config files
    config_file_lst = []
    # find built-in config file
    config_path = os.path.join(os.path.dirname(__file__), '../data/config')
    config_file = os.path.join(config_path, 'HIRES.cfg')
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
    rawdata = config['data'].get('rawdata')
    section = config['reduce']
    midproc = section.get('midproc')
    result  = section.get('result')
    report  = section.get('report')
    mode    = section.get('mode')

    # create folders if not exist
    if not os.path.exists(report):
        os.mkdir(report)
    if not os.path.exists(result):
        os.mkdir(result)
    if not os.path.exists(midproc):
        os.mkdir(midproc)

    nccd = 3

    ################################ parse bias ################################
    section = config['reduce.bias']
    bias_file = section['bias_file']

    if os.path.exists(bias_file):
        has_bias = True
        # load bias data from existing file
        hdu_lst = fits.open(bias_file)
        biasdata_lst = [hdu_lst[iccd+1].data for iccd in range(nccd)]
        hdu_lst.close()
        logger.info('Load bias from image: %s'%bias_file)
    else:
        # read each individual CCD
        bias_lst = [[] for iccd in range(nccd)]
        for item in log:
            if item.objectname[0].strip().lower()=='bias':
                filename = os.path.join(rawdata, '%s.fits'%item.fileid)
                hdu_lst = fits.open(filename)
                for iccd in range(nccd):
                    data = hdu_lst[iccd+1].data
                    bias_lst[iccd].append(np.float64(data))
                hdu_lst.close()

        has_bias = len(bias_lst[0])>0

        if has_bias:
            # there is bias frames
            head_lst     = [] # each bias has a head
            biasdata_lst = []
            hdu_lst = fits.HDUList([fits.PrimaryHDU()])

            # scan for each ccd
            for iccd in range(nccd):
                ### 3 CCDs loop begins here ###
                head = fits.Header()
                head['HIERARCH EDRS BIAS NFILE'] = len(bias_lst[iccd])
                bias = combine_images(bias_lst[iccd],
                        mode       = 'mean',
                        upper_clip = section.getfloat('cosmic_clip'),
                        maxiter    = section.getint('maxiter'),
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

                biasdata_lst.append(bias)
                hdu_lst.append(fits.ImageHDU(data=bias, header=head))
                ### 3 CCDs loop ends here ##

            hdu_lst.writeto(bias_file, overwrite=True)

    ########################## find flat groups #########################

    print('*'*10 + 'Parsing Flat Fieldings' + '*'*10)
    # initialize flat_groups
    flat_groups = {}
    # flat_groups = {'flat_M': [fileid1, fileid2, ...],
    #                'flat_N': [fileid1, fileid2, ...]}
