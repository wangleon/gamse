import os
import re
import sys
import shutil
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils.obslog import read_obslog
from ..utils.misc   import write_system_info

from . import common
from . import (feros, foces, hds, hires, levy, lhrs, sarg, xinglong216hrs)

instrument_lst = [
    ('foces',           'Fraunhofer',       'FOCES'),
    ('xinglong216hrs',  'Xinglong216',      'HRS'),
    ('hires',           'Keck-I',           'HIRES'),
    ('levy',            'APF',              'Levy'),
    ('hds',             'Subaru',           'HDS'),
    ('lhrs',            'LAMOST',           'HRS'),
    ('feros',           'MPG/ESO-2.2m',     'FEROS'),
    ]

def reduce_echelle():
    """Automatically select the instrument and reduce echelle spectra
    accordingly.

    Available instruments include:
        
        * *FOCES*: FOCES on 2m Fraunhofer Telescope in Wendelstein Observatory,
            Germany.
        * *Xinglong216HRS*: HRS on 2.16m telescope in Xinglong Station, China.

    """

    log_filename = 'gamse.log'
    # initialize running log
    log_fmt = ' '.join(['*',
                        '%(asctime)s.%(msecs)03d',
                        '[%(levelname)s]',
                        '%(name)s - %(lineno)d - %(funcName)s():'+os.linesep,
                        ' %(message)s'+os.linesep+'-'*80,
                        ])
    # check if there's already an existing log file
    if os.path.exists(log_filename):
        # if logfile already exists, rename it with its creation time
        time_str = None
        file1 = open(log_filename)
        for row in file1:
            # find the first time string in the contents
            mobj = re.search('(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', row)
            if mobj:
                time_str = mobj.group()
                break
        file1.close()

        if time_str is None:
            # time string not found
            # rename it to gamse.DDDD.log, where DDD is an increasing number
            i = 1
            while(True):
                newfilename = 'gamse.{}.log'.format(i)
                if os.path.exists(newfilename):
                    i += 1
                    continue
                else:
                    break
        else:
            # time string is found, rename it to gamse.YYYY-MM-DDTHH-MM-SS.log
            time_str = time_str.replace(':', '-')
            newfilename = 'gamse.{}.log'.format(time_str)

        # rename the existing gamse.log file
        shutil.move(log_filename, newfilename)

    # load config file in current directory
    config_file_lst = [fname for fname in os.listdir(os.curdir)
                        if fname.endswith('.cfg')]
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # the level of running log depends on the mode in the config
    mode = config['reduce']['mode']
    if mode == 'normal':
        level = logging.INFO
    elif mode == 'debug':
        level = logging.DEBUG
    else:
        level = logging.INFO

    # initialize running log
    logging.basicConfig(
            filename = log_filename,
            level    = level,
            format   = log_fmt,
            datefmt  = '%Y-%m-%dT%H:%M:%S',
            )
    logger = logging.getLogger(__name__)

    # write some system info into the running log
    write_system_info()

    # find telescope and instrument from config file
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']

    logger.info('Start reducing {}, {} data'.format(telescope, instrument))

    for row in instrument_lst:
        if telescope == row[1] and instrument == row[2]:
            eval(row[0]).reduce_rawdata()
            exit()

    print('Unknown Instrument: {} - {}'.format(telescope, instrument))

def make_obslog():
    """Scan the path to the raw FITS files and generate an observing log.

    Before generating the observing log file, this function will scan the local
    directory and look for *all* files with their names ending with ".cfg", and
    read them as config files.
    The config files are used to find the name of the instrument that the data
    was obtained with.
    """
    config_file_lst = []

    # find local config file
    for fname in os.listdir(os.curdir):
        if fname.endswith('.cfg'):
            config_file_lst.append(fname)

    # load ALL local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # find the telescope and instrument name
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']

    for row in instrument_lst:
        if telescope == row[1] and instrument == row[2]:
            eval(row[0]).make_obslog()
            exit()

    print('Unknown Instrument: {} - {}'.format(telescope, instrument))

def make_config():
    """Print a list of supported instrument and generate a config file according
    to user's selection.
    
    """

    # display a list of supported instruments
    print('List of supported instruments:')
    for i, row in enumerate(instrument_lst):
        telescope  = row[1]
        instrument = row[2]
        print('[{}] {}/{}'.format(i+1, telescope, instrument))

    # select instrument
    while(True):
        string = input('Select the instrument: ')
        if string.isdigit():
            select = int(string)
            break
        else:
            print('Error: invalid input')
            continue

    # use individual functions in each pipeline
    modulename = instrument_lst[select-1][0]
    eval(modulename).make_config()

def show_onedspec():
    """Show 1-D spectra in a pop-up window.

    Args:
        filename_lst (list): List of filenames of 1-D spectra.
    """

    # load obslog
    logname_lst = [fname for fname in os.listdir(os.curdir)
                        if fname.endswith('.obslog')]
    if len(logname_lst)==0:
        logtable = None
    else:
        logtable = read_obslog(logname_lst[0])

    # load config files in the current directory
    config_file_lst = [fname for fname in os.listdir(os.curdir)
                        if fname.endswith('.cfg')]
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    filename_lst = []
    for arg in sys.argv[2:]:

        # first, check if argument is a filename.
        if os.path.exists(arg):
            filename_lst.append(arg)
        # if not a filename, try to find the corresponding items in obslog
        else:
            if config is None:
                config = load_config('\S*\.cfg$')
            if logtable is None:
                logtable = load_obslog('\S*\.obslog$')

            # if arg is a number, find the corresponding filename in obslog
            if arg.isdigit():
                arg = int(arg)
                section = config['reduce']
                for logitem in logtable:
                    if arg == logitem['frameid']:
                        # get the path to the 1d spectra
                        odspath = section.get('odspath', None)
                        if odspath is None:
                            odspath = section.get('oned_spec')

                        # get the filename suffix for 1d spectra
                        oned_suffix = config['reduce'].get('oned_suffix')

                        fname = '{}_{}.fits'.format(
                                logitem['fileid'], oned_suffix)
                        filename = os.path.join(odspath, fname)
                        if os.path.exists(filename):
                            filename_lst.append(filename)
                        break

    if len(filename_lst)==0:
        exit()

    spec_lst = []
    for filename in filename_lst:
        data = fits.getdata(filename)

        # determine the column name of flux that will be shown
        if 'flux' in data.dtype.names:
            flux_key = 'flux'
        elif 'flux_sum' in data.dtype.names:
            flux_key = 'flux_sum'
        else:
            flux_key = ''
            pass

        if 'fiber' in data.dtype.names:
            # multi fiber
            for fiber in np.unique(data['fiber']):
                spec = {}
                mask = data['fiber']==fiber
                for row in data[mask]:
                    order = row['order']
                    wave  = row['wavelength']
                    flux  = row[flux_key]
                    spec[order] = (wave, flux)
                label = os.path.basename(filename) + ' Fiber {}'.format(fiber)
                spec_lst.append((spec, label))
        else:
            spec = {}
            for row in data:
                order = row['order']
                wave  = row['wavelength']
                flux  = row[flux_key]
                spec[order] = (wave, flux)
            label = os.path.basename(filename)
            spec_lst.append((spec, label))
    ################################################

    fig = plt.figure(figsize=(15, 8), dpi=150)
    ax = fig.add_axes([0.07, 0.1, 0.88, 0.8])

    def plot_order(order):
        ax.cla()
        ax.currentorder = order
        wave_min, wave_max = 1e9, 0
        flux_min = 1e9
        for i, (spec, label) in enumerate(spec_lst):
            if order in spec:
                wave = spec[order][0]
                flux = spec[order][1]
                ax.plot(wave, flux, '-', alpha=0.8, lw=0.8, label=label)
                wave_min = min(wave_min, wave.min())
                wave_max = max(wave_max, wave.max())
                flux_min = min(flux_min, flux.min())
        leg = ax.legend(loc='upper right')
        leg.get_frame().set_alpha(0.1)
        ax.set_xlabel(u'Wavelength (\xc5)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title('Order %d'%(order), fontsize=14)
        ax.set_xlim(wave_min, wave_max)
        ax.axhline(y=0, color='k', ls='--', lw=0.5)
        if flux_min > 0:
            ax.set_ylim(0,)
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'up':
            can_plot = False
            for spec, label in spec_lst:
                if ax.currentorder + 1 in spec:
                    can_plot=True
                    break
            if can_plot:
                plot_order(ax.currentorder + 1)
        elif event.key == 'down':
            can_plot = False
            for spec, label in spec_lst:
                if ax.currentorder - 1 in spec:
                    can_plot=True
                    break
            if can_plot:
                plot_order(ax.currentorder - 1)
        else:
            pass

    order0 = list(spec_lst[0][0].keys())[0]
    plot_order(order0)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def plot_spectra1d():

    # load config files in the current directory
    config_file_lst = [fname for fname in os.listdir(os.curdir)
                        if fname.endswith('.cfg')]
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # find telescope and instrument from config file
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']

    for row in instrument_lst:
        if telescope == row[1] and instrument == row[2]:
            eval(row[0]).plot_spectra1d()
            exit()

def convert_onedspec():
    """Convert one-dimensional spectra.
    """
    config = common.load_config('\S*\.cfg$', verbose=False)
    logtable = common.load_obslog('\S*\.obslog$', fmt='astropy', verbose=False)

    section = config['reduce']
    odspath     = section.get('odspath', None)
    oned_suffix = section.get('oned_suffix')

    filename_lst = []
    if len(sys.argv)==2:
        # no addtional args. convert all of the onedspec
        for fname in sorted(os.listdir(odspath)):
            if fname.endswith('.fits') or fname.endswith('.fit'):
                filename = os.path.join(odspath, fname)
                filename_lst.append(filename)
    else:
        for arg in sys.argv[2:]:
            if os.path.exists(arg):
                filename_lst.append(arg)
            elif os.path.exists(os.path.join(odspath, arg)):
                filename_lst.append(os.path.join(odspath, arg))
            else:
                if arg.isdigit():
                    arg = int(arg)
                for logitem in logtable:
                    if arg == logitem['frameid'] or arg == logitem['fileid']:
                        pattern = str(logitem['fileid'])+'\S*'
                        for fname in sorted(os.listdir(odspath)):
                            filename = os.path.join(odspath, fname)
                            if os.path.isfile(filename) \
                                and re.match(pattern, fname):
                                filename_lst.append(filename)

    for filename in filename_lst:
        data = fits.getdata(filename)

        if 'flux' in data.dtype.names:
            flux_key = 'flux'
        elif 'flux_sum' in data.dtype.names:
            flux_key = 'flux_sum'
        else:
            pass

        spec = {}
        for row in data:
            order = row['order']
            wave = row['wavelength']
            flux = row[flux_key]

            if wave[0]> wave[-1]:
                wave = wave[::-1]
                flux = flux[::-1]

            spec[order] = (wave, flux)
            ascii_prefix = os.path.splitext(os.path.basename(filename))[0]
            target_path = os.path.join(odspath, ascii_prefix)
            target_fname = '{}_order_{:03d}.txt'.format(ascii_prefix, order)
            target_filename = os.path.join(target_path, target_fname)

            if not os.path.exists(target_path):
                os.mkdir(target_path)

            if os.path.exists(target_filename):
                print('Warning: {} is overwritten'.format(target_filename))

            outfile = open(target_filename, 'w')
            for w, f in zip(wave, flux):
                outfile.write('{:11.5f} {:+16.8e}'.format(w, f)+os.linesep)
            outfile.close()

        print('Convert {} to {} files with ASCII formats in {}'.format(
                filename, len(data), target_path))
