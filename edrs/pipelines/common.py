import os
import logging
logger = logging.getLogger(__name__)
import configparser

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils.obslog import read_log, find_log
from ..utils.misc   import write_system_info

from . import xinglong216hrs
from . import foces
from . import levy
from . import sarg

def reduce_echelle():
    '''Automatically select the instrument and reduce echelle spectra
    accordingly.

    Available instruments include:
        
        * *FOCES*: FOCES on 2m Fraunhofer Telescope in Wendelstein Observatory,
            Germany.
        * *Xinglong216HRS*: HRS on 2.16m telescope in Xinglong Station, China.

    '''

    # initialize running log
    log_fmt = ' '.join(['*',
                        '%(asctime)s.%(msecs)03d',
                        '[%(levelname)s]',
                        '%(name)s - %(lineno)d - %(funcName)s():'+os.linesep,
                        ' %(message)s'+os.linesep+'-'*80,
                        ])
    logging.basicConfig(filename='edrs.log',level=logging.DEBUG,
            format=log_fmt, datefmt='%Y-%m-%dT%H:%M:%S')
    logger = logging.getLogger(__name__)

    # write system info
    write_system_info()

    # load config file in current directory
    config_file_lst = [fname for fname in os.listdir(os.curdir)
                        if fname[-4:]=='.cfg']
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )
    config.read(config_file_lst)

    # find telescope and instrument from config file
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']

    logger.info('Start reducing %s, %s data'%(telescope, instrument))

    if telescope == 'Fraunhofer' and instrument == 'FOCES':
        reduction = foces.FOCES()
        reduction.reduce()
    elif telescope == 'Xinglong216' and instrument == 'HRS':
        xinglong216hrs.reduce()
    elif telescope == 'APF' and instrument == 'Levy':
        levy.reduce()
    else:
        print('Unknown Instrument: %s - %s'%(telescope, instrument))
        exit()

def make_log():
    '''Scan the path to the raw FITS files and generate an observing log.
    '''
    config_file = find_config('./')
    config = read_config(config_file)
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']
    rawdata    = section['rawdata']
    key = get_instrument()

    if key == ('Fraunhofer', 'FOCES'):
        foces.make_log(rawdata)
    elif key == ('Xinglong216', 'HRS'):
        xinglong216hrs.make_log(rawdata)
    elif key == ('APF', 'Levy'):
        levy.make_log(rawdata)
    else:
        print('Unknown Instrument: %s - %s'%(telescope, instrument))
        exit()

def get_instrument():
    '''Find the telescope and instrument by checking the raw FITS files.

    Returns:
        str: Name of the instrument.
    '''
    config_file = find_config('./')
    config = read_config(config_file)
    section = config['data']
    telescope  = section['telescope']
    instrument = section['instrument']
    return telescope, instrument


def find_rawdata():
    '''Find the path to the raw images.

    Returns:
        *str* or *None*: Path to the raw images. Return *None* if path not found.
    '''

    if os.path.exists('rawdata'):
        return 'rawdata'
    else:
        config_file = find_config('./')
        config = read_config(config_file)
        if config.has_section('path') and \
           config.has_option('path', 'rawdata'):
            return config.get_option('path', 'rawdata')
        else:
            return None

def plot_spectra1d():
    '''Plot 1d spectra.
    '''
    config = read_config('')

    obslog_file = find_log(os.curdir)
    log = read_log(obslog_file)

    section = config['data']

    midproc = section['midproc']
    report  = section['report']

    steps_string = config['reduction']['steps']
    step_lst = steps_string.split(',')
    suffix = config[step_lst[-1].strip()]['suffix']
    image_path = 'images'
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    color_lst = 'rgbcmyk'

    for item in log:
        if item.imagetype == 'sci':
            filename = os.path.join(midproc, '%s%s.fits'%(item.fileid, suffix))
            if not os.path.exists(filename):
                continue
            data = fits.getdata(filename)

            omin = data['order'].min()
            omax = data['order'].max()
            order_lst = np.arange(omin, omax+1)

            for io, order in enumerate(order_lst):
                if io%10 == 0:
                    fig = plt.figure(figsize=(14.14,10), dpi=150)
                ax = fig.add_axes([0.055+(io%2)*0.50,
                                   0.06 + (4-int((io%10)/2.))*0.188, 0.43, 0.16])

                wavemin, wavemax = 1e9, 0
                channels = sorted(np.unique(data['channel']))
                for ich, channel in enumerate(channels):
                    mask1 = (data['channel']==channel)
                    mask2 = (data['order']==order)
                    mask = mask1*mask2
                    if mask.sum()==0:
                        continue
                    row = data[mask][0]
                    wave = row['wavelength']
                    flux = row['flux']
                    color = color_lst[ich%7]
                    ax.plot(wave, flux, color+'-', lw=0.7, alpha=0.7)
                    wavemin = min(wavemin, wave.min())
                    wavemax = max(wavemax, wave.max())
                ax.set_xlabel(u'Wavelength (\xc5)')
                x1, x2 = wavemin, wavemax
                y1, y2 = ax.get_ylim()
                ax.text(0.97*x1+0.03*x2, 0.8*y2, 'Order %d'%order)
                ax.set_xlim(x1, x2)
                ax.set_ylim(0, y2)
                if io%10 == 9:
                    fig.savefig(os.path.join(image_path, 'spec_%s_%02d.png'%(item.fileid, int(io/10.))))
                    plt.close(fig)
            fig.savefig(os.path.join(image_path, 'spec_%s_%02d.png'%(item.fileid, int(io/10.))))
            plt.close(fig)

def show_spectra1d(filename_lst):
    '''Show 1-D spectra in a pop-up window.

    Args:
        filename_lst (list): List of filenames of 1-D spectra.
    '''
    spec_lst = []
    for filename in filename_lst:
        data = fits.getdata(filename)
        spec = {}
        for row in data:
            order = row['order']
            wave  = row['wavelength']
            flux  = row['flux']
            spec[order] = (wave, flux)
        spec_lst.append(spec)

    fig = plt.figure(figsize=(15, 8), dpi=150)
    ax = fig.add_axes([0.07, 0.1, 0.88, 0.8])

    def plot_order(order):
        ax.cla()
        ax.currentorder = order
        wave_min, wave_max = 1e9, 0
        flux_min = 1e9
        for i, spec in enumerate(spec_lst):
            if order in spec:
                wave = spec[order][0]
                flux = spec[order][1]
                ax.plot(wave, flux, '-', alpha=0.8, lw=0.8,
                        label=os.path.basename(filename_lst[i]))
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
            for spec in spec_lst:
                if ax.currentorder + 1 in spec:
                    can_plot=True
                    break
            if can_plot:
                plot_order(ax.currentorder + 1)
        elif event.key == 'down':
            can_plot = False
            for spec in spec_lst:
                if ax.currentorder - 1 in spec:
                    can_plot=True
                    break
            if can_plot:
                plot_order(ax.currentorder - 1)
        else:
            pass

    order0 = list(spec_lst[0].keys())[0]
    plot_order(order0)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
