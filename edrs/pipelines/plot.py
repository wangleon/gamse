import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from ..utils.config  import read_config
from ..utils.obslog  import read_log, find_log

def plot_spectra1d():
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


