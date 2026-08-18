import os
import re
import time
import configparser
from typing import List, Tuple, Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import astropy.io.fits as fits

from ..utils.obslog import read_obslog

from .engine import FrameStep, resolve_reference, FrameResult
from .base import ImageFrame

def load_config(pattern, verbose=True):
    """Load the config file.

    Args:
        pattern (str):
        verbose (bool):

    Returns:
        config
    """
    # load config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation = configparser.ExtendedInterpolation(),
                )
    # find local config file
    for fname in os.listdir(os.curdir):
        if re.match(pattern, fname):
            config.read(fname)
            if verbose:
                message = 'Load congfig file: "{}"'.format(fname)
                print(message)
            break
    return config

def load_obslog(pattern, fmt='obslog', verbose=True):
    """Find and read the observing log file.

    Args:
        pattern (str): Pattern of the filename of observing log.
        fmt (str):
        verbose (bool):

    Returns:
        :class:`astropy.io.Table`: Observing log table.
    """

    # find observing log in the current workin gdirectory
    logname_lst = [fname for fname in os.listdir(os.curdir)
                            if re.match(pattern, fname)]

    if len(logname_lst)==0:
        print('No observation log found')
        return None
    elif len(logname_lst)==1:
        select_logname = logname_lst[0]
    elif len(logname_lst)>1:
        nlog = len(logname_lst)
        # maximum length of log filename
        maxlen = max([len(logname) for logname in logname_lst])
        # maximum length of log number
        maxdgt = len(str(nlog))
        fmt_string = (' - [{{:{:d}d}}] {{:{:d}s}}     '
                      'Last modified in {{:s}}').format(maxdgt, maxlen)

        # build a list of (filename, modified time)
        nametime_lst = [(logname, os.path.getmtime(logname))
                                for logname in logname_lst]

        # sort with last modified time
        nametime_lst = sorted(nametime_lst, key=lambda v:v[1])

        # print lognames one by one
        for i, (logname, mtime) in enumerate(nametime_lst):
            t = time.localtime(mtime)
            time_str = '{0:02d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
                        *t)
            print(fmt_string.format(i, logname, time_str))

        # repeat the loop until user give a valid logname ID
        while(True):
            string = input('Select an observing log: ')
            if string.isdigit() and int(string) < nlog:
                select_logname = nametime_lst[int(string)][0]
                break
            elif len(string.strip())==0:
                print('Warning: no logfile selected')
            else:
                print('Warning: {} is not a valid log ID'.format(string))
    else:
        pass

    if verbose:
        message = 'Load obslog file: "{}"'.format(select_logname)
        print(message)

    logtable = read_obslog(select_logname, fmt=fmt)
    return logtable


def plot_spectra1d():
    """Plot 1d spectra.
    """
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


FITS_PREFIX = 'HIERARCH GAMSE '

def _get_cards_from_header(head):
    card_lst = []
    for _key, _value in head.items():
        if _key.startswith(FITS_PREFIX):
            _key = _key[len(FITS_PREFIX):].strip()
            card_lst.append((_key, _value))
    return card_lst

def _save_fits_image(data, card_lst, outfile):

    head = fits.Header()
    for _key, _value in card_lst:
        _key = FITS_PREFIX + _key
        head.append((_key, _value))

    hdulst = fits.HDUList([
                fits.PrimaryHDU(data=data, header=head)
                ])
    hdulst.writeto(outfile, overwrite=True)

def _save_fits_table(data, card_lst, outfile):

    head = fits.Header()
    for _key, _value in card_lst:
        _key = FITS_PREFIX + _key
        head.append((_key, _value))

    hdulst = fits.HDUList([
                fits.PrimaryHDU(header=head),
                fits.BinTableHDU(data=data)
                ])
    hdulst.writeto(outfile, overwrite=True)


def save_fits(data, card_lst, outfile):
    if data.dtype.kind in ('i', 'f'):
        #  data is a normal array. save it as image
        _save_fits_image(data, card_lst, outfile)
    elif data.dtype.kind == 'V':
        # data is a structured array. save it as table in the first extension
        _save_fits_table(data, card_lst, outifle)
    else:
        print('Data type error')
        raise ValueError

def get_spectype(n):
    types = [
            ('aperture',    np.int16),
            ('order',       np.int16),
            ('x',           (np.float32, n)),
            ('y',           (np.float32, n)),
            ('wavelength',  (np.float64, n)),
            ('flux',        (np.float32, n)),
            ('error',       (np.float32, n)),
            ('background',  (np.float32, n)),
            ('mask',        (np.int16,   n)),
            ]
    names, formats = list(zip(*types))
    spectype = np.dtype({'names': names, 'formats': formats})
    return spectype

@runtime_checkable
class Processor(Protocol):

    name: str

    def process(self):
        ...

class BiasSubtractionStep(FrameStep):
    def run(self, result, context, **options):

        dataframe = result.frame

        # get bias from from context
        bias_product = resolve_reference(options['input'], context)
        bias_frame = bias_product.value

        data = dataframe.data - bias_frame.data

        extra_head = dataframe.extra_head.copy()
        prefix = 'HIERARCH REDUCTION BIAS '
        extra_head.append((prefix+'CORRECTED', True))
        extra_head.append((prefix+'MEAN', bias_frame.data.mean()))
        extra_head.append((prefix+'MEDIAN', np.median(bias_frame.data)))

        new_dataframe = ImageFrame(data = data,
                                   head = dataframe.head,
                                   mask = dataframe.mask,
                                   extra_head = extra_head,
                                   )
        new_result = FrameResult(frame = new_dataframe)
        return new_result
