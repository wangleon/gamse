import os
import datetime
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils    import obslog
from ..ccdproc  import save_fits, array_to_table
from .reduction import Reduction

class FOCES(Reduction):
    '''Reduction pipleline for FOCES.
    '''

    def __init__(self):
        super(FOCES, self).__init__()

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
           :header: "Option", "Type", "Description"
           :widths: 20, 10, 50

           "**overscan.skip**",    "*bool*",   "Skip this step if *yes* and **mode** = *'debug'*."
           "**overscan.surfix**",  "*string*", "Surfix of the corrected files."
           "**overscan.plot**",    "*bool*",   "Plot the overscan levels if *yes*."
           "**overscan.var_fig**", "*string*", "Filename of the overscan variation figure."


        '''

        # find output surfix for fits
        self.output_surfix = self.config.get('reduction', 'overscan.surfix')

        if self.config.getboolean('reduction', 'overscan.skip'):
            logger.info('Skip [overscan] according to the config file')
            self.input_surfix = self.output_surfix
            return True

        self.report_file.write('<h2>Overscan</h2>'+os.linesep)
        t_lst, frameid_lst, fileid_lst = [], [], []
        ovr1_lst, ovr1_std_lst = [], []
        ovr2_lst, ovr2_std_lst = [], []
        
        # saturated CCD count
        saturation_adu = 65535
    
        # loop over all files to correct for the overscan

        count = 0
        for item in self.log:
            logger.info('Correct overscan for item %3d: "%s"'%(
                         item.frameid, item.fileid))

            # read in of the data
            filename = '%s%s.fits'%(item.fileid, self.input_surfix)
            fname = os.path.join(self.paths['data'], filename)
            data, head = fits.getdata(fname, header=True)
    
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
            if count%5 == 0:
                fig = plt.figure(figsize=(8,6), dpi=150)

            ax1 = fig.add_axes([0.10, 0.83-(count%5)*0.185, 0.42, 0.15])
            ax2 = fig.add_axes([0.55, 0.83-(count%5)*0.185, 0.42, 0.15])
            ax1.plot(ovr_lst1, 'r-', alpha=0.3)
            ax2.plot(ovr_lst2, 'b-', alpha=0.3)
            y = np.arange(vy1, vy2)
            ax1.plot(y, ovr_lst1[vy1:vy2], 'r-', alpha=0.7)
            ax2.plot(y, ovr_lst2[vy1:vy2], 'b-', alpha=0.7)
            x1,x2 = 0, ovr_lst1.size-1
            ax1.plot([x1,x2],[ovrmean1,         ovrmean1],         'm-')
            ax1.plot([x1,x2],[ovrmean1-ovrstd1, ovrmean1-ovrstd1], 'm:')
            ax1.plot([x1,x2],[ovrmean1+ovrstd1, ovrmean1+ovrstd1], 'm:')
            ax2.plot([x1,x2],[ovrmean2,         ovrmean2],         'c-')
            ax2.plot([x1,x2],[ovrmean2-ovrstd2, ovrmean2-ovrstd2], 'c:')
            ax2.plot([x1,x2],[ovrmean2+ovrstd2, ovrmean2+ovrstd2], 'c:')
            ax1.set_ylabel('ADU')
            ax2.set_ylabel('')
            y11,y12 = ax1.get_ylim()
            y21,y22 = ax2.get_ylim()
            y1 = min(y11,y21)
            y2 = max(y12,y22)
            ax1.text(0.95*x1+0.05*x2, 0.2*y1+0.8*y2,
                     '%s (%s)'%(item.fileid, item.objectname), fontsize=9)
            for ax in [ax1, ax2]:
                ax.set_xlim(x1,x2)
                ax.set_ylim(y1,y2)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
            ax2.set_yticklabels([])
            if count%5==4:
                ax1.set_xlabel('Y (pixel)')
                ax2.set_xlabel('Y (pixel)')
                figname = 'overscan_%02d.png'%(int(count/5)+1)
                figpath = os.path.join(self.paths['report_img'], figname)
                fig.savefig(figpath)
                logger.info('Save image: %s'%figpath)
                plt.close(fig)
                self.report_file.write('<img src="images/%s">'%figname+os.linesep)
    
            # find saturated pixels and saved them in FITS files
            mask_sat   = (data[:,20:2068]>=saturation_adu)
            mask       = np.int16(mask_sat)*4
            mask_table = array_to_table(mask)
            mask_fname = os.path.join(self.paths['midproc'],
                         '%s%s.fits'%(item.fileid, self.mask_surfix))
            # save the mask.
            save_fits(mask_fname, mask_table)
    
            # subtract overscan
            new_data = data[:,20:2068] - ovrmean1
    
            # update fits header
            # head['BLANK'] is only valid for integer arrays.
            del head['BLANK']
            head['HIERARCH EDRS OVERSCAN']        = True
            head['HIERARCH EDRS OVERSCAN METHOD'] = 'mean'
            head['HIERARCH EDRS OVERSCAN AXIS-1'] = '1:20'
            head['HIERARCH EDRS OVERSCAN AXIS-2'] = '%d:%d'%(vy1,vy2)
            head['HIERARCH EDRS OVERSCAN MEAN']   = ovrmean1
            head['HIERARCH EDRS OVERSCAN STDEV']  = ovrstd1
    
            # save data
            newfilename = '%s%s.fits'%(item.fileid, self.output_surfix)
            newpath = os.path.join(self.paths['midproc'], newfilename)
            save_fits(newpath, new_data, head)
            print('Correct Overscan {} -> {}'.format(filename, newfilename))
    
            # quality check of the mean value of the overscan with time
            # therefore the time and the mean overscan values are stored in 
            # a list to be analyzed later
            t_lst.append(head['UTC-STA'])
            frameid_lst.append(item.frameid)
            fileid_lst.append(item.fileid)
            ovr1_lst.append(ovrmean1)
            ovr2_lst.append(ovrmean2)
            ovr1_std_lst.append(ovrstd1)
            ovr2_std_lst.append(ovrstd2)

            count += 1
    
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
    
        self.plot_overscan_variation(t_lst, ovr1_lst)

        logger.info('Overscan corrected. Change surfix: %s -> %s'%
                    (self.input_surfix, self.output_surfix))
        self.input_surfix = self.output_surfix

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
        data, head = fits.getdata(os.path.join(path, fname), header=True)
        scidata = data[:,20:-20]
        obsdate = head['UTC-STA']
        exptime = head['EXPTIME']
        try:
            objectname = head['OBJECT']
        except:
            objectname = 'Unknown'

        # determine the fraction of saturated pixels permillage
        mask_sat = (scidata>=63000)
        prop = float(mask_sat.sum())/scidata.size*1e3

        # find the brightness index in the central region
        h, w = scidata.shape
        data1 = scidata[int(h*0.3):int(h*0.7),int(w/2)-2:int(w/2)+3]
        brightness = np.median(data1,axis=1).mean()

        item = obslog.LogItem(
                   fileid     = fileid,
                   obsdate    = obsdate,
                   exptime    = exptime,
                   objectname = objectname,
                   saturation = prop,
                   brightness = brightness,
                   )
        log.add_item(item)

    log.sort('obsdate')

    # make info list
    all_info_lst = []
    columns = ['frameid (i)', 'fileid (s)', 'objectname (s) ', 'exptime (f)',
               'obsdate (s)', 'saturation (f)', 'brightness (f)']
    prev_frameid = -1
    for item in log:
        frameid = int(item.fileid.split('_')[1])
        if frameid <= prev_frameid:
            print('Warning: frameid {} > prev_frameid {}'.format(frameid, prev_frameid))
        info_lst = [
                    str(frameid),
                    str(item.fileid),
                    str(item.objectname),
                    '%.3f'%item.exptime,
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
    string = '% columns = '+', '.join(columns)
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
        list: A list containing the records

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

