import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

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
            self.columns = all_columns
        else:
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

    def get_format(self, has_esc=True, color_rule=None):
        _, _, _, fmt_item = zip(*self.columns)
        fmt_item = list(fmt_item)
        if has_esc:
            return ' '.join(fmt_item)
        else:
            pattern = re.compile('\x1b\[[\d;]+m([\s\S]*)\x1b\[0m')
            newfmt_item = []
            for item in fmt_item:
                mobj = pattern.match(item)
                if mobj is None:
                    newfmt_item.append(item)
                else:
                    newfmt_item.append(mobj.group(1))
            return ' '.join(newfmt_item)

