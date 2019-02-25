import os
import re
import time
import shutil
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from mpl_toolkits.mplot3d import Axes3D

from ..utils.config       import read_config
from ..utils.obslog       import read_log, parse_num_seq, find_log
from ..echelle.imageproc  import table_to_array, array_to_table
from ..echelle.trace      import find_apertures, load_aperture_set
from ..echelle.flat       import mosaic_flat_auto, mosaic_images, get_fiber_flat
from ..echelle.background import find_background
from ..echelle.extract    import sum_extract
from ..echelle.wlcalib    import wlcalib, recalib, reference_wl

class Reduction(object):
    '''General echelle reduction.

    Attributes:
        config (:class:`configparser.ConfigParser`): Config of the reduction.
        log (:class:`stella.utils.obslog.Log`): Observing log.
        paths (tuple): A tuple containing following paths:

            * *'rawdata'*: Path to raw images.
            * *'midproc'*: Path to mid-process files.
            * *'report'*: Path to report file.
            * *'report'*: Path to images used in report.

        input_suffix (string): Surfix of filenames before each step.
        output_suffix (string): Surfix of filenames after each step.
        mask_suffix (string): Surfix of mask filenames.

    '''

    def __init__(self, instrument):

        self.instrument = instrument

        # read config file
        self.load_config()

        # read log file
        self.load_log()

    def reduce(self):
        '''
        Main loop of the reduction procedure.
        '''
        # initiliaze file suffix
        self.input_suffix  = ''
        self.output_suffix = ''
        self.mask_suffix = self.config.get('reduction', 'mask_suffix')
        # read steps from config file
        steps_string = self.config.get('reduction', 'steps')

        report_filename = os.path.join(self.paths['report'], 'index.html')
        self.report_file = open(report_filename, 'w')

        # write report header
        text = [
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'+
            ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<head>',
            '    <meta http-equiv="content-type" content="text/html; charset=utf-8" />',
            '    <title>Reduction Report</title>',
            '</head>',
            '<body>',
            '    <h1>Reduction Report</h1>']
        self.report_file.write(os.linesep.join(text)+os.linesep)

        # main loop
        step_lst = [v.strip() for v in steps_string.split(',') if v.strip()!='']
        logger.info('Reduction steps = %s'%(','.join(step_lst)))
        for step in step_lst:
            if hasattr(self, step):
                getattr(self, step)()
            else:
                logger.error('Unknown step: %s'%step)

        # write message footer
        text = ['</body>', '</html>']
        self.report_file.write(os.linesep.join(text))

    def load_config(self):
        '''
        Load config file and check the paths.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 20, 10, 50

           **obslog_file**, *string*, (*optional*) Name of the observing log file.
           **rawdata**,     *string*, Path to raw images.
           **midproc**,     *string*, Path to midprocess files.
           **report**,      *string*, Path to reduction report.
           **result**,      *string*, Path to result files

        '''
        self.config = read_config(instrument=self.instrument)

        # get a dict of paths
        #for option in self.config.options('path'):
        #    self.paths[option] = self.config.get('path', option)
        section = self.config['data']
        self.paths = {
                'rawdata': section['rawdata'],
                'midproc': section['midproc'],
                'report':  section['report'],
                'result':  section['result'],
        }

        # check if data path exist
        _rawdata = self.paths['rawdata']
        if not os.path.exists(_rawdata):
            logger.error('data path: "%s" does not exist'%_rawdata)
            exit()

        # Check whether the necessary paths exist
        for _pathname in ['midproc', 'report', 'result']:
            _path = self.paths[_pathname]
            if not os.path.exists(_path):
                os.mkdir(_path)
                logger.info('Create a new directory (%s path: "%s")'%(
                    _pathname, _path)
                    )

    def _find_flat_groups(self):
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
                if len(g)>0 and g[0].lower().strip()=='flat':
                    # the object name of the channel matches "flat ???"

                    # check the lengthes of names for other channels
                    # if this list has no elements (only one fiber) or has no
                    # names, this frame is a single-channel flat
                    other_lst = [len(v) for i, v in enumerate(item.objectname)
                                 if i != ichannel]
                    if len(other_lst) == 0 or max(other_lst)==0:
                        # this frame is a single channel flat

                        # find a proper name for this flat
                        if name.lower().strip()=='flat':
                            # no special names given, use "flat_A_15.000"
                            flatname = 'flat_%s_%.3f'%(channel, item.exptime)
                        else:
                            # flatname is given. replace space with "_"
                            # remove "flat" before the objectname. e.g.,
                            # "Flat Red" becomes "Red" 
                            char = name[4:].strip()
                            # add a channel string
                            flatname = 'flat_%s_%s'%(channel, char.replace(' ','_'))

                        # add flatname to flat_groups
                        if flatname not in flat_groups[channel]:
                            flat_groups[channel][flatname] = []
                        flat_groups[channel][flatname].append(item)

                    else:
                        # this frame is not a single chanel flat. Skip
                        pass

        # put the flat_groups into class attributes
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
        self._find_flat_groups()


    def _find_bias(self):
        '''Find bias frames.

        Scan the log file and find items with "objectname" containing "bias".

        Returns:
            list: A list containing the IDs of bias frames.
        '''
        # first try to search the bias in config file
        if self.config.has_section('objects') and \
           self.config.has_option('objects', 'bias'):
            logger.info('Find bias fileids in config file')
            return parse_num_seq(self.config.get('objects', 'bias'))

        # find bias in log
        bias_id_lst = [item.frameid for item in self.log
                                    for name in item.objectname
                                    if name.lower().strip() == 'bias'
                      ]
        if len(bias_id_lst) > 0:
            logger.info('Find bias fileids in log')
            return bias_id_lst

    def _find_trace(self):
        '''Scan the log file and find trace items in any channel.

        Returns:
            dict: A dict containing the :class:`LogItem` instances of trace
                frames in each channel.
        '''
        message = ['Finding Traces', 'channel frameid fileid']
        find_trace = False

        trace_lst = {}
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            
            for item in self.log:
                name = item.objectname[ichannel]
                g =  name.split()
                if len(g)>0 and g[0].lower().strip() == 'trace':
                    # object name matches "trace ???"
                    
                    # check the lengthes of names for other channels
                    # if this list has no elements (only one fiber) or has no
                    # names, this frame is a single-channel trace
                    other_lst = [len(v) for i, v in enumerate(item.objectname)
                                 if i != ichannel]
                    if len(other_lst) == 0 or max(other_lst)==0:
                        # this frame is a single channle trace

                        # this frame is a single channle trace image
                        if channel not in trace_lst:
                            trace_lst[channel] = []
                        trace_lst[channel].append(item)

                        find_trace = True
                        message.append('%s %3d %s'%(channel, item.frameid, item.fileid))
                    else:
                        # this frame is not a single chanel trace. Skip
                        pass

        # write messages into running log
        if find_trace:
            logger.info((os.linesep+' '*3).join(message))
        else:
            logger.info('No trace file found')

        return trace_lst

    def find_science(self):
        '''Find science items.

        Returns:
            list: A list containing the :class:`LogItem` instances for
                scientific objects.
        '''
        item_lst = [item for item in self.log if item.imagetype=='sci']
        return item_lst

    def find_comp(self):
        '''Scan the log file and find comparison items.

        Returns:
            list: A list containing the :class:`LogItem` instances for
                comparison lamps.
        '''
        item_lst = []
        for item in self.log:
            if item.imagetype=='cal':
                for ich in range(self.nchannels):
                    name = item.objectname[ich]
                    if name.lower().strip()=='thar':
                        item_lst.append(item)
                        break
        return item_lst

    def combine_flat(self, item_list, flatname, exptime_key='EXPOSURE'):
        '''
        Combine flat fielding frames.

        If there's only one file to be combined, just copy it to the destinated
        filename.

        Args:
            item_list (list): List of flat items.
            flatname (string): Name of the input flat set.
        Returns:
            No returns.
        '''
        # path alias
        midproc = self.paths['midproc']

        nfile = len(item_list)

        # find the name of the output flat file
        out_flatfile = '%s.fits'%flatname
        out_flatpath = os.path.join(midproc, out_flatfile)

        # find the name of the output mask file
        out_maskfile = '%s%s.fits'%(flatname, self.mask_suffix)
        out_maskpath = os.path.join(midproc, out_maskfile)

        if nfile == 0:
            logger.info('No file to be combined')
            return False
        elif nfile == 1:
            # only one file. Do not need combine. Directly copy to output file
            item = item_list[0]

            # deal with the fits file
            basename = '%s%s.fits'%(item.fileid, self.input_suffix)
            filename = os.path.join(midproc, basename)
            shutil.copyfile(filename, out_flatpath)
            logger.info('Copy "%s" to flat image: "%s"'%(filename, out_flatpath))

            # deal with the mask file
            maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
            maskpath = os.path.join(self.paths['midproc'], maskname)
            shutil.copyfile(maskpath, out_maskpath)
            logger.info('Copy mask of "%s" to flat mask: "%s"'%(maskpath, out_maskpath))
        else:
            # more than one file to be combined
    
            # create a header object for combined flat
            newhead = fits.Header()
            newhead['HIERARCH EDRS FLAT TYPE']  = 'mean'
            newhead['HIERARCH EDRS FLAT NFILE'] = nfile
        
            # combine flat
            # initialize the final data array and the total exposure time.
            all_data = []
            total_exptime = 0.
            for i, item in enumerate(item_list):
                # load image data
                filename = '%s%s.fits'%(item.fileid, self.input_suffix)
                filepath = os.path.join(midproc, filename)
                data, head = fits.getdata(filepath, header=True)
                if i==0:
                    # all_maks can only be initialized here because its shape is
                    # unkown before reading the first data
                    all_mask = np.zeros_like(data, dtype=np.int16)

                all_data.append(data)
                total_exptime += head[exptime_key]
            
                # load mask data
                maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
                maskpath = os.path.join(midproc, maskname)
                mtable = fits.getdata(maskpath)
                if mtable.size==0:
                    mdata = np.zeros_like(data, dtype=np.int16)
                else:
                    mdata = table_to_array(mtable, data.shape)
                # get new mask
                all_mask = (all_mask|mdata)
            
                # save the filename of each combined flat in to header
                newhead['HIERARCH EDRS FLAT FILE %d'%i] = filename
        
            # clipping algorithm
            all_data = np.array(all_data)
            nz, ny, nx = all_data.shape
            #mask = (all_data == all_data.max(axis=0))
            mask = (np.mgrid[:nz, :ny, :nx][0]==all_data.argmax(axis=0))
            maxiter = 10
            for nite in range(maxiter):
                mdata = np.ma.masked_array(all_data, mask=mask)
                m = mdata.mean(axis=0,dtype=np.float64).data
                s = mdata.std(axis=0, dtype=np.float64).data
                new_mask = np.ones_like(mask, dtype=np.bool)
                for i in np.arange(all_data.shape[0]):
                    data = all_data[i,:,:]
                    mask1 = data > m + 10.*s
                    new_mask[i,:,:] = mask1
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask

            mdata = np.ma.masked_array(all_data, mask=mask)
            data_mean = mdata.mean(axis=0, dtype=np.float64).data

            mexptime  = total_exptime/float(nfile)
            newhead['HIERARCH EDRS FLAT MEANEXPTIME'] = mexptime
            newhead[exptime_key] = mexptime
        
            fits.writeto(out_flatpath, data_mean, newhead, overwrite=True)
            logger.info('Save combined flat image: "%s"'%out_flatpath)
            print('save %s'%out_flatfile)
            
            # save the mask for each individual flat frame
            mtable = array_to_table(all_mask)
            fits.writeto(out_maskpath, mtable, overwrite=True)
            logger.info('Save mask image for combined flat: "%s"'%out_maskpath)

    def combine_trace(self):
        '''
        Combine trace frames.
        '''

        # path alias
        midproc = self.paths['midproc']

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
            filename = '%s%s.fits'%(item.fileid, self.input_suffix)
            filepath = os.path.join(midproc, filename)
            data, head = fits.getdata(filepath, header=True)

            # read mask
            maskname = '%s%s.fits'%(item.fileid, self.mask_suffix)
            maskpath = os.path.join(midproc, maskname)
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
        fits.writeto(trace_file, data_sum, newhead, overwrite=True)
        logger.info('Save trace image: "%s"'%trace_file)

    def flat2(self):
        '''
        Flat fielding correction
        '''

        if self.config.getboolean('reduction', 'flat.skip'):
            logger.info('Skip [flat] according to the config file')
            #self.input_suffix = self.output_suffix
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
                        mask_suffix = self.mask_suffix
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
           :header: Option, Type, Description
           :widths: 20, 10, 80

           **skip**,       *bool*,    Skip this step if *yes* and **mode** = *'debug'*.
           **file**,       *string*,  Name of the trace file.
           **scan_step**,  *integer*, Steps of pixels used to scan along the main dispersion direction.
           **minimum**,    *float*,   Minimum value to filter the input image.
           **seperation**, *float*,   Estimated order seperations (in pixel) at *y* = 0 along the cross-dispersion.
           **sep_der**,    *float*,   Estimated first derivative of seperations per 1000 pixels along the *y* axis.
           **filling**,    *float*,   Fraction of detected pixels to total step of scanning.
           **display**,    *bool*,    Display a figure on screen if *yes*.
           **degree**,     *integer*, Degree of polynomial used to describe the positions of orders.
        '''
        if self.config.getboolean('trace', 'skip'):
            logger.info('Skip [trace] according to the config file')
            return True


        # find the parameters for order tracing
        kwargs = {
            'minimum'   : self.config.getfloat('trace', 'minimum'),
            'scan_step' : self.config.getint('trace', 'scan_step'),
            'seperation': self.config.getfloat('trace', 'seperation'),
            'sep_der'   : self.config.getfloat('trace', 'sep_der'),
            'filling'   : self.config.getfloat('trace', 'filling'),
            'display'   : self.config.getboolean('trace', 'display'),
            'degree'    : self.config.getint('trace', 'degree'),
            }

        trace_lst = self._find_trace()

        aperture_set_lst = {}

        # path alias
        midproc = self.paths['midproc']
        report  = self.paths['report']

        for ichannel in range(self.nchannels):
            channel = chr(ichannel + 65)
            print(ichannel, channel)
            # initialize aperture_set_lst, containing the order locations from
            # different tracing files (either trace or flat image)
            aperture_set_lst[channel] = {}

            if channel in trace_lst:
                if len(trace_lst[channel]) > 1:
                    filename_lst = [os.path.join(midproc, '%s%s.fits'%(item.fileid, self.input_suffix))
                                    for item in trace_lst[channel]]
                    tracename = 'trace_%s'%channel
                    dst_filename = os.path.join(midproc,'%s.fits'%tracename)

                    # combine the trace files
                    combine_fits(filename_lst, dst_filename, mode='sum')

                    # read the combined image and its mask
                    data = fits.getdata(dst_filename)
                    mask = np.zeros_like(data, dtype=np.bool)
                    for item in trace_lst[channel]:
                        mask_file = os.path.join(midproc, '%s%s.fits'%(item.fileid, self.mask_suffix))
                        mask_table = fits.getdata(mask_file)
                        imask = table_to_array(mask_table, data.shape)
                        imask = (imask&4 == 4)
                        mask = (mask|imask)
                else:
                    item = trace_lst[channel][0]
                    tracename = item.fileid
                    filename = os.path.join(midproc, '%s%s.fits'%(item.fileid, self.input_suffix))
                    data = fits.getdata(filename)
                    mask_file = os.path.join(midproc, '%s%s.fits'%(item.fileid, self.mask_suffix))
                    mask_table = fits.getdata(mask_file)
                    mask = table_to_array(mask_table, data.shape)
                    mask = (mask&4 == 4)

                trc_file = os.path.join(midproc, '%s.trc'%tracename)
                reg_file = os.path.join(midproc, '%s_trc.reg'%tracename)
                fig_file = os.path.join(report,  'trace_%s.png'%tracename)

                kwargs.update({'mask'       : mask,
                               'filename'   : trace_file,
                               'trace_file' : trc_file,
                               'reg_file'   : reg_file,
                               'fig_file'   : fig_file,
                               })
                aperture_set = find_apertures(data, **kwargs)

                logger.info('Found %d orders in "%s.fits"'%(len(aperture_set), trace_file))

                aperture_set_lst[channel][tracename] = aperture_set

            else:
                # no trace file for this channel. use flat instead.
                logger.info('Cannot find trace images for channel %s. Use flat images instead'%channel)

                for flatname, item_lst in sorted(self.flat_groups[channel].items()):
                    print(flatname)
                    logger.info('Begin processing flat component: %s'%flatname)

                    flatpath = os.path.join(midproc, '%s.fits'%flatname)

                    # combine flats
                    self.combine_flat(item_lst, flatname)

                    data = fits.getdata(flatpath)
                    mask_file = os.path.join(midproc, '%s%s.fits'%(flatname, self.mask_suffix))
                    mask_table = fits.getdata(mask_file)
                    if mask_table.size==0:
                        mask = np.zeros_like(data, dtype=np.int16)
                    else:
                        mask = table_to_array(mask_table, data.shape)
                    mask = (mask&4 == 4)

                    # determine the result file and figure file
                    trc_file = os.path.join(midproc, '%s.trc'%flatname)
                    reg_file = os.path.join(midproc, '%s_trc.reg'%flatname)
                    fig_file = os.path.join(report,  'trace_%s.png'%flatname)

                    # find the apertures

                    kwargs.update({'mask'       : mask,
                                   'filename'   : flatpath,
                                   'trace_file' : trc_file,
                                   'reg_file'   : reg_file,
                                   'fig_file'   : fig_file,
                                   })

                    aperture_set = find_apertures(data, **kwargs)

                    logger.info('Found %d apertures in "%s.fits"'%(len(aperture_set), flatname))

                    aperture_set_lst[channel][flatname] = aperture_set

        self.aperture_set_lst = aperture_set_lst

    def flat(self):
        '''
        Flat fielding correction.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 25, 10, 70

           **skip**,            *bool*,    Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,          *string*,  Surfix of the flat correceted files.
           **cosmic_clip**,     *float*,   Upper clipping threshold to remove cosmis-rays.
           **slit_step**,       *integer*, Step of slit function scanning.
           **q_threshold**,     *float*,   Threshold of *Q*-factor.
           **param_deg**,       *integer*, Degree of polynomial in parameters fitting.
           **file**,            *string*,  
           **mosaic_maxcount**, *integer*, Maximum count of the flat mosaic.
        '''
        section = self.config['flat']

        # find output suffix for fits
        self.output_suffix = section.get('suffix')

        if section.getboolean('skip'):
            logger.info('Skip [flat] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        slit_step   = section.getint('slit_step')
        q_threshold = section.getfloat('q_threshold')
        param_deg   = section.getint('param_deg')

        # path alias
        midproc = self.paths['midproc']
        report  = self.paths['report']

        mosaic_aperset_lst = {}
        # e.g. {'A': ApertureSet instance, 'B': ApertureSet instance,...}

        reg_color_lst = ['green', 'yellow', 'red', 'blue']

        # temporarily added for debug purpose
        #--------------------------------------
        aperture_set_lst = {}
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            aperture_set_lst[channel] = {}
            for flatname in sorted(self.flat_groups[channel]):
                trc_file = os.path.join(midproc, '%s.trc'%flatname)
                aperture_set = load_aperture_set(trc_file)
                aperture_set_lst[channel][flatname] = aperture_set
        self.aperture_set_lst = aperture_set_lst
        #--------------------------------------

        channel_flatmap_lst = {}
        # loop for each channels
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)

            channel_name = 'flat_%s'%channel

            flat_file = os.path.join(midproc, '%s.fits'%channel_name)
            mask_file = os.path.join(midproc, '%s%s.fits'%(channel_name, self.mask_suffix))
            resp_file = os.path.join(midproc, '%s_rsp.fits'%channel_name)
            trc_file  = os.path.join(midproc, '%s.trc'%channel_name)
            reg_file  = os.path.join(midproc, '%s_trc.reg'%channel_name)

            flat_group = self.flat_groups[channel]
            aperset_lst = self.aperture_set_lst[channel]

            # loop over all kinds of flats in this channel, and get the map of
            # pixel-to-pixel respondings.

            # prepare for data list to aviod duplicate reading.
            data_lst = {}
            mask_lst = {}
            resp_lst = {}

            for flatname in sorted(flat_group):
                # make a sub directory
                subdir = os.path.join(report, flatname)
                if not os.path.exists(subdir):
                    os.mkdir(subdir)

                print(channel, flatname)

                # read flat fielding image
                infile = os.path.join(midproc, '%s.fits'%flatname)
                data = fits.getdata(infile)
                # pack the data to data_lst
                data_lst[flatname] = data

                # read flat fielding mask
                mask_file = os.path.join(midproc,
                                '%s%s.fits'%(flatname, self.mask_suffix))
                mask_table = fits.getdata(mask_file)
                if mask_table.size==0:
                    mask = np.zeros_like(data, dtype=np.int16)
                else:
                    mask = table_to_array(mask_table, data.shape)
                # pack the mask to mask_lst
                mask_lst[flatname] = mask

                aperset = aperset_lst[flatname]
                fig_aperpar = os.path.join(subdir,  'flat_aperpar_%02d.png')
                fig_overlap = os.path.join(subdir,  'flat_overlap_%04d.png')
                fig_slit    = os.path.join(report,  '%s_slit.png'%flatname)
                slit_file   = os.path.join(midproc, '%s.slt'%flatname)

                nflat = len(flat_group[flatname])

                flatmap = get_fiber_flat(data, mask,
                            apertureset = aperset,
                            slit_step   = slit_step,
                            nflat       = nflat,
                            q_threshold = q_threshold,
                            param_deg   = param_deg,
                            fig_aperpar = fig_aperpar,
                            fig_overlap = fig_overlap,
                            fig_slit    = fig_slit,
                            slit_file   = slit_file,
                            )

                # save flat result to fits file
                outfile = os.path.join(midproc, '%s_rsp.fits'%flatname)
                fits.writeto(outfile, flatmap, overwrite=True)
                resp_lst[flatname] = flatmap

                # write to running log
                _string = 'Channel %s, %s: Flat map saved as "%s"'
                _message = _string%(channel, flatname, outfile)
                logger.info(_message)
            # sensitivity map for each color ends here

            # mosaic different colors of flats
            if len(flat_group) == 1:
                # only 1 type of flat, no mosaic. just copy the file (if needed)
                flatname = flat_group.keys()[0]
                if flatname != channel_name:
                    # if names are different, copy the flat, trc and reg files
                    inflt_file = os.path.join(midproc, '%s.fits'%flatname)
                    inmsk_file = os.path.join(midproc, '%s%s.fits'%(flatname, self.mask_suffix))
                    intrc_file = os.path.join(midproc, '%s.trc'%flatname)
                    inreg_file = os.path.join(midproc, '%s_trc.reg'%flatname)
                    shutil.copyfile(inflt_file, flat_file)
                    shutil.copyfile(inmsk_file, mask_file)
                    shutil.copyfile(intrc_file, trc_file)
                    shutil.copyfile(inreg_file, reg_file)
                    # write to running log
                    _message = [
                            'Copy "%s" to "%s"'%(inflt_file, flat_file),
                            'Copy "%s" to "%s"'%(inmsk_file, mask_file),
                            'Copy "%s" to "%s"'%(intrc_file, trc_file),
                            'Copy "%s" to "%s"'%(inreg_file, reg_file),
                            ]
                    logger.info(os.linesep.join(_message))
                else:
                    # if names are the same, do nothing
                    _message = 'Flatname = channel name. nothing to do.'
                    logger.info(_message)

                # pack the image to channel_flatmap_lst
                channel_flatmap_lst[channel] = fits.getdata(flat_file)

            elif len(flat_group) > 1:
                # multiple kinds of flats. do mosaic

                # get filename from config file
                max_count = section.getfloat('mosaic_maxcount')

                # mosaic apertures
                mosaic_aperset = mosaic_flat_auto(
                                    aperture_set_lst = aperset_lst,
                                    max_count        = max_count,
                                    )

                # mosaic original flat images
                flat_data = mosaic_images(data_lst, mosaic_aperset)
                fits.writeto(flat_file, flat_data, overwrite=True)

                # mosaic flat mask images
                mask_data = mosaic_images(mask_lst, mosaic_aperset)
                mask_table = array_to_table(mask_data)
                fits.writeto(mask_file, mask_table, overwrite=True)

                # mosaic sensitivity map
                resp_data = mosaic_images(resp_lst, mosaic_aperset)
                fits.writeto(resp_file, resp_data, overwrite=True)

                # align different channels
                if ichannel > 0:
                    # align this channel relative to channel A
                    ref_aperset = mosaic_aperset_lst['A']
                    offset = mosaic_aperset.find_aper_offset(ref_aperset)
                    _fmt = 'Offset = {} for channel {} relative to channel A'
                    logger.info(_fmt.format(offset, channel))
                    mosaic_aperset.shift_aperture(offset)
                mosaic_aperset_lst[channel] = mosaic_aperset

                # save mosaiced aperset to .txt and .reg files
                mosaic_aperset.save_txt(trc_file)
                mosaic_aperset.save_reg(reg_file,
                                        channel = channel,
                                        color   = reg_color_lst[ichannel%4]
                                        )
            else:
                print('Unknown flat_groups')
                raise ValueError
            # flat mosaic ended here

            
            # pack the mosaiced flat map
            channel_flatmap_lst[channel] = resp_data


        # channel loop ends here

        exit()
        sci_item_lst = self.find_science()
        for item in sci_item_lst:
            input_file = os.path.join(midproc,
                            '%s%s.fits'%(item.fileid, self.input_suffix))
            output_file = os.path.join(midproc,
                            '%s%s.fits'%(item.fileid, self.output_suffix))
            shutil.copy(input_file, output_file)
            logger.info('Correct Flat: "{}"->"{}"'.format(input_file, output_file))

        self.input_suffix = self.output_suffix


    def background(self):
        '''
        Subtract the background for 2D images.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 25, 10, 80

           **skip**,       *bool*,    Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,     *string*,  Suffix of the background correceted files.
           **display**,    *bool*,    Display a graphics if *yes*.
           **scan_step**,  *integer*, Steps of pixels used to scan along the main dispersion direction.
           **xorder**,     *integer*, Degree of 2D polynomial along *x*-axis (dispersion direction).
           **yorder**,     *integer*, Degree of 2D polynomial along *y*-axis (cross-dispersion direction).
           **maxiter**,    *integer*, Maximum number of iteration of 2D polynomial fitting.
           **upper_clip**, *float*,   Upper sigma clipping threshold.
           **lower_clip**, *float*,   Lower sigma clipping threshold.
           **extend**,     *bool*,    Extend the grid to the whole image if *True*.

        '''

        section = self.config['background']
        # find output suffix for fits
        self.output_suffix = section.get('suffix')

        if section.getboolean('skip'):
            logger.info('Skip [background] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        # path alias
        midproc = self.paths['midproc']
        report  = self.paths['report']

        # read config parameters
        display    = section.getboolean('display')
        scan_step  = section.getint('scan_step')
        xorder     = section.getint('xorder')
        yorder     = section.getint('yorder')
        maxiter    = section.getint('maxiter')
        upper_clip = section.getfloat('upper_clip')
        lower_clip = section.getfloat('lower_clip')
        extend     = section.getboolean('extend')

        # load aperture set for different channels
        aperset_lst = {}
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            trcfilename = 'flat_%s.trc'%channel
            trcfile = os.path.join(midproc, trcfilename)
            aperset = load_aperture_set(trcfile)
            aperset_lst[channel] = aperset

        # prepare the file queue
        infile_lst  = []
        mskfile_lst = []
        outfile_lst = []
        scafile_lst = []
        scale_lst   = [] # different files use differenet scales
        channel_lst = []
        figfile_lst = []
        regfile_lst = []

        sci_item_lst = self.find_science()
        for item in sci_item_lst:
            infilename  = '%s%s.fits'%(item.fileid, self.input_suffix)
            mskfilename = '%s%s.fits'%(item.fileid, self.mask_suffix)
            outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)
            scafilename = '%s%s.fits'%(item.fileid, '_sca')
            regfilename = '%s%s.reg'%(item.fileid, self.output_suffix)
            imgfilename = 'bkg-%s.png'%item.fileid

            infile_lst.append(os.path.join(midproc, infilename))
            mskfile_lst.append(os.path.join(midproc, mskfilename))
            outfile_lst.append(os.path.join(midproc, outfilename))
            scafile_lst.append(os.path.join(midproc, scafilename))
            regfile_lst.append(os.path.join(midproc, regfilename))
            figfile_lst.append(os.path.join(report, imgfilename))

            channels = [chr(ich+65) for ich, objectname in enumerate(item.objectname)
                            if len(objectname)>0]
            channel_lst.append(channels)
            scale_lst.append('log')

        # correct the backgrounds
        for i in range(len(infile_lst)):
            infile  = infile_lst[i]
            mskfile = mskfile_lst[i]
            outfile = outfile_lst[i]
            scafile = scafile_lst[i]
            regfile = regfile_lst[i]
            figfile = figfile_lst[i]
            channel = channel_lst[i]
            scale   = scale_lst[i]

            print('Correct background: %s -> %s'%(infile, outfile))
            correct_background(infile, mskfile, outfile, scafile,
                               channels        = channel,
                               apertureset_lst = aperset_lst,
                               scale           = scale,
                               block_mask      = 4,
                               scan_step       = scan_step,
                               xorder          = xorder,
                               yorder          = yorder,
                               maxiter         = maxiter,
                               upper_clip      = upper_clip,
                               lower_clip      = lower_clip,
                               extend          = extend,
                               display         = display,
                               fig_file        = figfile,
                               reg_file        = regfile,
                               )

        # update suffix
        self.input_suffix = self.output_suffix
        return True

        if False:    
            # prepare the file queue
            infile_lst, mskfile_lst, outfile_lst, scafile_lst = [], [], [], []
            # different files use differenet scales
            scale_lst = []
    
            # check combined flat
            if self.config.has_option('reduction', 'fileid.flat'):
                # there only 1 flats in the dataset
                flat_file = self.config.get('reduction', 'flat.file')
                msk_file  = '%s%s.fits'%(flat_file[0:-5], self.mask_suffix)
                out_file  = '%s%s.fits'%(flat_file[0:-5], self.output_suffix)
                sca_file  = '%s_sca.fits'%(flat_file[0:-5])
                infile_lst.append(flat_file)
                mskfile_lst.append(msk_file)
                outfile_lst.append(out_file)
                scafile_lst.append(sca_file)
                scale_lst.append('linear')
                logger.info('Add "%s" to the background file queue'%flat_file)
        
            if len(self.flat_groups)>0:
                for flatname in sorted(self.flat_groups.keys()):
                    infilename  = os.path.join(self.paths['midproc'], '%s.fits'%flatname)
                    mskfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(flatname, self.mask_suffix))
                    outfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(flatname, self.output_suffix))
                    scafilename = os.path.join(self.paths['midproc'], '%s_sca.fits'%flatname)
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
                infilename  = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.input_suffix))
                mskfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.mask_suffix))
                outfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.output_suffix))
                scafilename = os.path.join(self.paths['midproc'], '%s_sca.fits'%item.fileid)
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
                                   report_img_path = self.paths['report'],
                                   )
            
            if display:
                # close the figures
                plt.close(fig1)
                plt.close(fig2)
            

    def extract(self):
        '''
        Extract 1d spectra.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 25, 10, 60

           **skip**,        *bool*,    Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,      *string*,  Suffix of the extracted files.
           **upper_limit**, *float*,   Upper limit of extracted aperture.
           **lower_limit**, *float*,   Lower limit of extracted aperture.

        '''
        section = self.config['extract']

        # find output suffix for fits
        self.output_suffix = section.get('suffix')
    
        if section.getboolean('skip'):
            logger.info('Skip [extract] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        # path alias
        midproc = self.paths['midproc']

        upper_limit = section.getfloat('upper_limit')
        lower_limit = section.getfloat('lower_limit')

        # load aperture set for different channels
        aperset_lst = {}
        for ichannel in range(self.nchannels):
            channel = chr(ichannel+65)
            trcfilename = 'flat_%s.trc'%channel
            trcfile = os.path.join(midproc, trcfilename)
            aperset = load_aperture_set(trcfile)
            aperset_lst[channel] = aperset

        # prepare the file queue
        infile_lst  = []
        mskfile_lst = []
        outfile_lst = []
        channel_lst = []

        # add comparison lamps to the queue
        comp_item_lst = self.find_comp()
        for item in comp_item_lst:

            infilename  = '%s%s.fits'%(item.fileid, '_ovr')
            mskfilename = '%s%s.fits'%(item.fileid, self.mask_suffix)
            outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)

            infile_lst.append(os.path.join(midproc, infilename))
            mskfile_lst.append(os.path.join(midproc, mskfilename))
            outfile_lst.append(os.path.join(midproc, outfilename))

            channels = [chr(ich+65) for ich, objectname in enumerate(item.objectname)
                            if len(objectname)>0]
            channel_lst.append(channels)

        # add scientific frames to the queue
        sci_item_lst = self.find_science()
        for item in sci_item_lst:

            infilename  = '%s%s.fits'%(item.fileid, self.input_suffix)
            mskfilename = '%s%s.fits'%(item.fileid, self.mask_suffix)
            outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)

            infile_lst.append(os.path.join(midproc, infilename))
            mskfile_lst.append(os.path.join(midproc, mskfilename))
            outfile_lst.append(os.path.join(midproc, outfilename))

            channels = [chr(ich+65) for ich, objectname in enumerate(item.objectname)
                            if len(objectname)>0]
            channel_lst.append(channels)

        for i in range(len(infile_lst)):
            infile = infile_lst[i]
            mskfile = mskfile_lst[i]
            outfile = outfile_lst[i]
            channel = channel_lst[i]

            print('Extract: %s -> %s'%(infile, outfile))
            sum_extract(infile, mskfile, outfile,
                        channels        = channel,
                        apertureset_lst = aperset_lst,
                        upper_limit     = upper_limit,
                        lower_limit     = lower_limit,
                        )
        return True

        #--------------

        order_loc_file = self.config.get('reduction', 'trace.location_file')
        order_lst, info = load_order_locations(order_loc_file)

        flat_file = self.config.get('reduction', 'flat.file')

        infile_lst, outfile_lst, mskfile_lst = [], [], []
    
        # check combined flat
        if self.config.has_option('reduction', 'fileid.flat'):
            infilename  = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.input_suffix))
            mskfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.mask_suffix))
            outfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(flat_file[0:-5],self.output_suffix))
            infile_lst.append(infilename)
            mskfile_lst.append(mskfilename)
            outfile_lst.append(outfilename)
            logger.info('Add "%s" to extraction queue'%infilename)
    
        if len(self.flat_groups)>0:
            for flatname in sorted(self.flat_groups.keys()):
                infilename  = os.path.join(self.paths['midproc'], '%s%s.fits'%(flatname,self.input_suffix))
                mskfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(flatname,self.mask_suffix))
                outfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(flatname,self.output_suffix))
                infile_lst.append(infilename)
                mskfile_lst.append(mskfilename)
                outfile_lst.append(outfilename)
                logger.info('Add "%s" to extraction queue'%infilename)
        
            # add mosaiced flat file
            infilename = flat_file
            mskfilename = '%s%s.fits'%(flat_file[0:-5],self.mask_suffix)
            outfilename = '%s%s.fits'%(flat_file[0:-5],self.output_suffix)
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
                infilename  = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid,
                              self.config.get('reduction','bias.suffix')))
                mskfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.mask_suffix))
                outfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.output_suffix))
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
            infilename  = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.input_suffix))
            mskfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.mask_suffix))
            outfilename = os.path.join(self.paths['midproc'], '%s%s.fits'%(item.fileid, self.output_suffix))
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

        # update suffix
        self.input_suffix = self.output_suffix

    def deblaze(self):
        '''
        Correct the blaze funtions of the 1-D spectra
        '''

        # find output suffix for fits
        self.output_suffix = self.config.get('reduction','deblaze.suffix')

        if self.config.getboolean('reduction', 'deblaze.skip'):
            logger.info('Skip [deblaze] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        flat_file = self.config.get('reduction','flat.file')
        flat_ext_file = '%s%s.fits'%(flat_file[0:-5], self.input_suffix)
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
                            '%s%s.fits'%(item.fileid, self.input_suffix))
            outfilename = os.path.join(self.paths['midproc'],
                            '%s%s.fits'%(item.fileid, self.output_suffix))
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

        # update suffix
        self.input_suffix = self.output_suffix

    def wvcalib(self):
        '''
        Wavelength calibration.

        .. csv-table:: Accepted options in config file
           :header: Option, Type, Description
           :widths: 25, 10, 60

           **skip**,          *bool*,    Skip this step if *yes* and **mode** = *'debug'*.
           **suffix**,        *string*,  Surfix of the extracted files.
           **linelist**,      *string*,  Name of the wavelength standard list.
           **window_size**,   *integer*, Size of the window in pixel to search for the lines.
           **xorder**,        *integer*, Order of polynomial along main dispersion direction.
           **yorder**,        *integer*, Order of polynomial along cross-dispersion direction.
           **maxiter**,       *integer*, Mximum number of polnomial fitting.
           **clipping**,      *float*,   Sigma-clipping threshold.
           **snr_threshold**, *float*,   Signal-to-noise ratio threshold of the emission line fitting.

        '''

        section = self.config['wvcalib']

        # find output suffix for fits
        self.output_suffix = section.get('suffix')

        if section.getboolean('skip'):
            logger.info('Skip [wavcalib] according to the config file')
            self.input_suffix = self.output_suffix
            return True

        # path alias
        midproc = self.paths['midproc']
        report  = self.paths['report']

        # get parameters from config file
        linelist      = section.get('linelist')
        window_size   = section.getint('window_size')
        xorder        = section.getint('xorder')
        yorder        = section.getint('yorder')
        maxiter       = section.getint('maxiter')
        clipping      = section.getfloat('clipping')
        snr_threshold = section.getfloat('snr_threshold')

        if self.nchannels == 0:
            # single fiber calibration

            # check comparison lamp
            if self.config.has_option('reduction', 'fileid.thar'):
                print('kkkkkkk') 
                # find comparison file list
                string = self.config.get('reduction', 'fileid.thar')
                id_lst = parse_num_seq(string)
                print(id_lst)
            
                # find linelist
                linelistname = section.get('linelist')
            
                extract_suffix = self.config.get('extract', 'suffix')
            
                for item in self.log:
                    if item.frameid in id_lst:
                        infilename  = '%s%s.fits'%(item.fileid, extract_suffix)
                        mskfilename = '%s%s.fits'%(item.fileid, self.mask_suffix)
                        infilepath  = os.path.join(midproc, infilename)
                        mskfilepath = os.path.join(midproc, mskfilename)
                        coeff = wvcalib(infilepath, linelistname)
                        break
            
                # find science file list
                string = self.config.get('reduction','fileid.science')
                sci_id_lst = parse_num_seq(string)
            
                for item in self.log:
                    if item.frameid in sci_id_lst:
                        infilename  = '%s%s.fits'%(item.fileid, extract_suffix)
                        outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)
                        infilepath  = os.path.join(midproc, infilename)
                        outfilepath = os.path.join(midproc, outfilename)
                        reference_wv(infilepath, outfilepath, coeff)
        else:
            # multifiber calibration

            # loop all channels
            result_lst = {}

            # find thar list for each channel
            thar_lst = {}
            for ich in range(self.nchannels):
                channel = chr(ich+65)
                thar_lst[channel] = [item for item in self.log
                                     if len(item.objectname) == self.nchannels and 
                                     item.objectname[ich] == 'ThAr']

            # calib_lst is a hierarchical dict of calibration results
            calib_lst = {}
            # calib_lst = {
            #       'frameid1': {'A': calib_dict1, 'B': calib_dict2, ...},
            #       'frameid2': {'A': calib_dict1, 'B': calib_dict2, ...},
            #       ... ...
            #       }
            for ich, (channel, item_lst) in enumerate(sorted(thar_lst.items())):
                # loop for every fiber channel

                for i, item in enumerate(item_lst):
                    # loop for item
                    infilename  = '%s%s.fits'%(item.fileid, self.input_suffix)
                    idtfilename = '%s_idt.dat'%item.fileid
                    figfilename = 'wvcalib-%s-%s.png'%(item.fileid, channel)

                    infilepath  = os.path.join(midproc, infilename)
                    idtfilepath = os.path.join(midproc, idtfilename)
                    figfilepath = os.path.join(report, figfilename)

                    spec1d = fits.getdata(infilepath)
                    mask = spec['channel']==channel
                    spec = spec[mask]

                    if ich == 0 and i == 0:
                        calib = wvcalib(spec1d,
                                        filename      = infilepath,
                                        identfilename = idtfilepath,
                                        figfilename   = figfilepath,
                                        channel       = channel,
                                        linelist      = linelist,
                                        window_size   = window_size,
                                        xorder        = xorder,
                                        yorder        = yorder,
                                        maxiter       = maxiter,
                                        clipping      = clipping,
                                        snr_threshold = snr_threshold,
                                        )
                        ref_calib = calib
                        spec = fits.getdata(infilepath)
                        ref_spec = spec[spec['channel']==channel]
                    else:
                        calib = recalib(infilepath,
                                        identfilename = idtfilepath,
                                        figfilename   = figfilepath,
                                        ref_spec      = ref_spec,
                                        channel       = channel,
                                        linelist      = linelist,
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
                    if item.frameid not in calib_lst:
                        calib_lst[item.frameid] = {}
                    calib_lst[item.frameid][channel] = calib

            comp_item_lst = self.find_comp()
            sci_item_lst  = self.find_science()

            # reference wavelength for comparison lamps
            for item in comp_item_lst:

                infilename  = '%s%s.fits'%(item.fileid, self.input_suffix)
                outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)
                regfilename = '%s%s.reg'%(item.fileid, self.output_suffix)

                infilepath  = os.path.join(midproc, infilename)
                outfilepath = os.path.join(midproc, outfilename)
                regfilepath = os.path.join(midproc, regfilename)

                reference_wv(infilepath, outfilepath, regfilepath, item.frameid, calib_lst)

            # reference wavelength for science targets
            for item in sci_item_lst:

                infilename  = '%s%s.fits'%(item.fileid, self.input_suffix)
                outfilename = '%s%s.fits'%(item.fileid, self.output_suffix)
                regfilename = '%s%s.reg'%(item.fileid, self.output_suffix)

                infilepath  = os.path.join(midproc, infilename)
                outfilepath = os.path.join(midproc, outfilename)
                regfilepath = os.path.join(midproc, regfilename)

                reference_wv(infilepath, outfilepath, regfilepath, item.frameid, calib_lst)

