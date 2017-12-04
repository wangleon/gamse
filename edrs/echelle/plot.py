import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def plot(filename):
    f = fits.open(filename)
    spec = f[1].data
    head = f[1].header
    f.close()

    fig = plt.figure(figsize=(14,8))
    height = 0.8
    ax = fig.add_axes([0.1,0.1,0.65,height])
    thumb_w = 0.18
    ratio = 10.
    thumb_height = height/(5*ratio+4)*ratio
    axt_lst = []
    for i in range(5):
        axt = fig.add_axes(
                [0.78,0.1+thumb_height*(1./ratio+1)*i,thumb_w,thumb_height])
        axt_lst.append(axt)
            
    ax.current_row = 0

    def plot_order():
        specdata = spec[ax.current_row]
        if 'wavelength' in spec.dtype.names:
            xdata = specdata['wavelength']
        else:
            xdata = np.arange(specdata['points'])
        ydata = specdata['flux']
        ax.cla()
        ax.plot(xdata,ydata,'r-')
        ax.set_xlim(xdata[0],xdata[-1])
        y1, y2 = ax.get_ylim()
        y1 = min(y1,0)
        ax.set_ylim(y1,y2)
        if 'wavelength' in spec.dtype.names:
            ax.xaxis.set_major_locator(tck.MultipleLocator(5))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(1))
            ax.set_xlabel(u'Wavelength (\xc5)')
        else:
            ax.xaxis.set_major_locator(tck.MultipleLocator(500))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
            ax.set_xlabel('Pixel')
        ax.set_title('Order %d'%specdata['order'])

        for iaxt, axt in enumerate(axt_lst):
            axt.cla()
            trow = ax.current_row + iaxt - 2
            if trow <= spec['order'].size-1 and trow >= 0:
                axt.set_axis_bgcolor('w')
                axt.set_axis_on()
                specdata = spec[trow]
                xdata = np.arange(specdata['points'])
                ydata = specdata['flux']
                if iaxt == 2:
                    color = 'r'
                    for spine in axt.spines:
                        axt.spines[spine].set_color('r')
                else:
                    color = 'gray'
                axt.plot(xdata, ydata, '-', color=color)
                axt.set_xlim(xdata[0], xdata[-1])
                x1,x2 = axt.get_xlim()
                y1,y2 = axt.get_ylim()
                axt.text(0.95*x1+0.05*x2, 0.8*y2+0.2*y1,
                        'Order %d'%spec['order'][trow], fontsize=9)
                y1 = min(y1,0)
                axt.set_ylim(y1,y2)
            else:
                axt.set_axis_bgcolor(fig.get_facecolor())
                axt.set_axis_off()
            axt.set_xticks([])
            axt.set_yticks([])
            axt.set_xticklabels([])
            axt.set_yticklabels([])


        fig.canvas.draw()

    def on_key(event):
        if event.key == 'up':
            if ax.current_row < spec['order'].size-1:
                ax.current_row += 1
                plot_order()
        elif event.key == 'down':
            if ax.current_row > 0:
                ax.current_row -= 1
                plot_order()
        else:
            pass

    fig.canvas.mpl_connect('key_press_event', on_key)


    plot_order()
    plt.show() 
