import os
import re
import time
import configparser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils.obslog import read_obslog

def load_config(pattern):
    """Load the config file.
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
            message = 'Load congfig file: "{}"'.format(fname)
            print(message)
            break
    return config

def load_obslog(pattern):
    """Find and read the observing log file.

    Args:
        pattern (str): Pattern of the filename of observing log.

    Returns:
        :class:`astropy.io.Table`: Observing log table.
    """

    # find observing log in the current workin gdirectory
    logname_lst = [fname for fname in os.listdir(os.curdir)
                            if re.match(pattern, fname)]

    if len(logname_lst)==0:
        print('No observation log found')
        exit()
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

    message = 'Load obslog file: "{}"'.format(select_logname)
    print(message)
    logtable = read_obslog(select_logname)
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


def plot_background_aspect1(data, stray, figname):
    """Plot a figure showing the image before background correction and the
    stray light.

    Args:
        data (:class:`numpy.ndarray`): Image before background correction.
        stray (:class:`numpy.ndarray`): Stray light.
        figname (str): Name of the output figure.

    """
    h, w = data.shape

    fig = plt.figure(figsize=(16,7), dpi=150)
    _width = 0.37
    _height = _width/w*h*16/7

    ax21 = fig.add_axes([0.06, 0.1, _width, _height])
    ax22 = fig.add_axes([0.55, 0.1, _width, _height])
    ax21c = fig.add_axes([0.06+_width+0.01, 0.1, 0.015, _height])
    ax22c = fig.add_axes([0.55+_width+0.01, 0.1, 0.015, _height])

    # find the minimum and maximum value of plotting
    s = np.sort(data.flatten())
    vmin = s[int(0.05*data.size)]
    vmax = s[int(0.95*data.size)]

    cax_data  = ax21.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
    cax_stray = ax22.imshow(stray, cmap='viridis')
    cs = ax22.contour(stray, colors='r', linewidths=0.5)
    ax22.clabel(cs, inline=1, fontsize=9, use_clabeltext=True)
    fig.colorbar(cax_data, cax=ax21c)
    fig.colorbar(cax_stray, cax=ax22c)
    for ax in [ax21, ax22]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.xaxis.set_major_locator(tck.MultipleLocator(500))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.yaxis.set_major_locator(tck.MultipleLocator(500))
        ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
    fig.savefig(figname)
    plt.close(fig)

def plot_background_aspect1_alt(data, stray, figname1, figname2):
    h, w = data.shape

    fig1 = plt.figure(figsize=(8.5,7), dpi=150)
    fig2 = plt.figure(figsize=(8.5,7), dpi=150)

    ax21  = fig1.add_axes([0.08, 0.08, 0.80, 0.9])
    ax22  = fig2.add_axes([0.08, 0.08, 0.80, 0.9])
    ax21c = fig1.add_axes([0.88, 0.08, 0.03, 0.9])
    ax22c = fig2.add_axes([0.88, 0.08, 0.03, 0.9])

    # find the minimum and maximum value of plotting
    s = np.sort(data.flatten())
    vmin = s[int(0.05*data.size)]
    vmax = s[int(0.95*data.size)]

    cax_data  = ax21.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
    cax_stray = ax22.imshow(stray, cmap='viridis')
    cs = ax22.contour(stray, colors='r', linewidths=0.5)
    ax22.clabel(cs, inline=1, fontsize=12, fmt='%g', use_clabeltext=True)
    fig1.colorbar(cax_data, cax=ax21c)
    fig2.colorbar(cax_stray, cax=ax22c)
    for ax in [ax21, ax22]:
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.xaxis.set_major_locator(tck.MultipleLocator(500))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.yaxis.set_major_locator(tck.MultipleLocator(500))
        ax.yaxis.set_minor_locator(tck.MultipleLocator(100))
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
    for axc in [ax21c, ax22c]:
        for tick in axc.yaxis.get_major_ticks():
            tick.label2.set_fontsize(14)

    fig1.savefig(figname1)
    fig2.savefig(figname2)


class PrintInfo(object):
    def __init__(self, columns):
        self.columns = columns

    def add_columns(self, columns):
        new_columns = self.columns.copy()
        for element in columns:
            new_columns.append(element)
        return PrintInfo(new_columns)

    def get_title(self):
        titles, _, fmt_title, _ = zip(*self.columns)
        fmt_title = ' '.join(fmt_title)
        return fmt_title.format(*titles)

    def get_dtype(self):
        _, dtypes, fmt_title, _ = zip(*self.columns)
        fmt_title = ' '.join(fmt_title)
        return fmt_title.format(*dtypes)

    def get_separator(self):
        lst = ['-'*len(fmt.format(title)) for title, _, fmt, _ in self.columns]
        return ' '.join(lst)

    def get_format(self):
        _, _, _, fmt_item = zip(*self.columns)
        return ' '.join(fmt_item)


class FormattedInfo(object):
    def __init__(self, all_columns, selected_columns=None):
        if selected_columns is None:
            # if selected_columns is not given, pick up all columns in
            # all_columns
            self.columns = all_columns
        else:
            # selected_columns is given. only pick up columns in
            # selected_columns
            column_lst = []
            for columns in selected_columns:
                for item in all_columns:
                    if item[0] == columns:
                        column_lst.append(item)
                        break
            self.columns = column_lst

    def add_columns(self, columns):
        new_columns = self.columns.copy()
        for element in columns:
            new_columns.append(element)
        return FormattedInfo(new_columns)

    def get_title(self, delimiter=' '):
        """Get the title string.
        Args：
            delimiter (str): Delimiter of the columns.
        Returns:
            str:
        """
        titles, _, fmt_title, _ = zip(*self.columns)
        fmt_title = delimiter.join(fmt_title)
        return fmt_title.format(*titles)

    def get_dtype(self, delimiter=' '):
        """Get the datatype string.
        Args：
            delimiter (str): Delimiter of the columns.
        Returns:
            str:
        """
        _, dtypes, fmt_title, _ = zip(*self.columns)
        fmt_title = delimiter.join(fmt_title)
        return fmt_title.format(*dtypes)

    def get_separator(self, delimiter=' '):
        """Get the separator string.
        Args：
            delimiter (str): Delimiter of the columns.
        Returns:
            str:
        """
        lst = ['-'*len(fmt.format(title)) for title, _, fmt, _ in self.columns]
        return delimiter.join(lst)

    def get_format(self, has_esc=True, color_rule=None, delimiter=' '):
        """
        Args：
            has_esc (bool):
            color_rule (str):
            delimiter (str): Delimiter of the columns.
        Returns:
            str:
        """
        _, _, _, fmt_item = zip(*self.columns)
        fmt_item = list(fmt_item)
        if has_esc:
            return delimiter.join(fmt_item)
        else:
            pattern = re.compile('\x1b\[[\d;]+m([\s\S]*)\x1b\[0m')
            newfmt_item = []
            for item in fmt_item:
                mobj = pattern.match(item)
                if mobj:
                    newfmt_item.append(mobj.group(1))
                else:
                    newfmt_item.append(item)
            return delimiter.join(newfmt_item)

