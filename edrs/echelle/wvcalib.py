import os
import re
import sys
import math
import itertools

import numpy as np
import numpy.polynomial as poly
import astropy.io.fits as fits
import scipy.interpolate as intp
import scipy.optimize as opt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

from ..ccdproc import save_fits
from ..utils.regression import polyfit2d, polyval2d
from ..utils.onedarray import pairwise

identlinetype = np.dtype({
    'names':  ['channel','aperture','order','pixel','wavelength','snr','mask',
               'residual','method'],
    'formats':['S1', np.int16, np.int16, np.float32, np.float64, np.float32,
               np.int16, np.float64, 'S1'],
    })

class CustomToolbar(NavigationToolbar2TkAgg):
    '''Class for customized matplotlib toolbar.

    Args:
        canvas (:class:`matplotlib.backends.backend_agg.FigureCanvasAgg`):
            Canvas object used in :class:`CalibWindow`.
        master (Tkinter widget): Parent widget.
    '''
    def __init__(self, canvas, master):
        '''Constructor of :class:`CustomToolbar`.
        '''
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move','pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect','zoom'),
            ('Subplots', 'Configure subplots', 'subplots','configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        NavigationToolbar2TkAgg.__init__(self, canvas, master)

    def set_message(self, msg):
        '''Remove the coordinate displayed in the toolbar.
        '''
        pass

class PlotFrame(tk.Frame):
    '''The frame for plotting spectrum in the :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (integer): Width of frame.
        height (integer): Height of frame.
        dpi (integer): DPI of figure.
    '''
    def __init__(self, master, width, height, dpi, identlist, linelist):
        '''Constructor of :class:`PlotFrame`.
        '''

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fig = CalibFigure(width    = width,
                               height   = height,
                               dpi      = dpi,
                               filename = os.path.basename(master.param['filename']),
                               channel  = master.param['channel'],
                               )
        self.ax1 = self.fig._ax1
        self.ax2 = self.fig._ax2
        self.ax3 = self.fig._ax3

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.canvas.mpl_connect('button_press_event', master.on_click)
        self.canvas.mpl_connect('draw_event', master.on_draw)

        aperture = master.param['aperture']
        self.ax1._aperture_text.set_text('Aperture %d'%aperture)

        # add toolbar
        self.toolbar = CustomToolbar(self.canvas, master=self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT)

        self.pack()

class InfoFrame(tk.Frame):
    '''The frame for buttons and tables on the right side of the :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (integer): Width of the frame.
        height (integer): Height of the frame.
        linelist (list): List of wavelength standards.
        identlist (dict): Dict of identified lines.
    '''
    def __init__(self, master, width, height, linelist, identlist):
        '''Constuctor of :class:`InfoFrame`.
        '''

        self.master = master

        filename = os.path.basename(master.param['filename'])
        channel  = master.param['channel']

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fname_label = tk.Label(master = self,
                                    width  = width,
                                    font   = ('Arial', 14),
                                    text   = filename,
                                    )
        self.order_label = tk.Label(master = self,
                                    width  = width,
                                    font   = ('Arial', 10),
                                    text   = '',
                                    )
        self.fname_label.pack(side=tk.TOP,pady=(30,5))
        self.order_label.pack(side=tk.TOP,pady=(5,10))

        button_width = 13

        self.switch_frame = tk.Frame(master=self, width=width, height=30)
        self.prev_button = tk.Button(master  = self.switch_frame,
                                     text    = '◀',
                                     width   = button_width,
                                     font    = ('Arial',10),
                                     command = master.prev_aperture,
                                     )
        self.next_button = tk.Button(master  = self.switch_frame,
                                     text    = '▶',
                                     width   = button_width,
                                     font    = ('Arial',10),
                                     command = master.next_aperture,
                                     )
        self.prev_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.RIGHT)
        self.switch_frame.pack(side=tk.TOP, pady=5, padx=10, fill=tk.X)

        # line table area
        self.line_frame = LineTable(master    = self,
                                    width     = width-30,
                                    height    = 900,
                                    identlist = identlist,
                                    linelist  = linelist)
        self.line_frame.pack(side=tk.TOP, padx=10, pady=5)


        # batch operation buttons
        button_width2 = 13
        self.batch_frame = tk.Frame(master=self, width=width, height=30)
        self.recenter_button = tk.Button(master  = self.batch_frame,
                                         text    = 'recenter',
                                         width   = button_width2,
                                         font    = ('Arial',10),
                                         command = master.recenter,
                                         )
        self.clearall_button = tk.Button(master  = self.batch_frame,
                                         text    = 'clear all',
                                         width   = button_width2,
                                         font    = ('Arial',10),
                                         command = master.clearall,
                                         )
        self.recenter_button.pack(side=tk.LEFT)
        self.clearall_button.pack(side=tk.RIGHT)
        self.batch_frame.pack(side=tk.TOP, pady=5, padx=10, fill=tk.X)


        # fit buttons
        self.auto_button = tk.Button(master=self, text='Auto Identify',
                            font = ('Arial', 10), width=25,
                            command = master.auto_identify)
        self.fit_button = tk.Button(master=self, text='Fit',
                            font = ('Arial', 10), width=25,
                            command = master.fit)
        self.switch_button = tk.Button(master=self, text='Plot',
                            font = ('Arial', 10), width=25,
                            command = master.switch)
        # set status
        self.auto_button.config(state=tk.DISABLED)
        self.fit_button.config(state=tk.DISABLED)
        self.switch_button.config(state=tk.DISABLED)

        # Now pack from bottom to top
        self.switch_button.pack(side=tk.BOTTOM, pady=(5,30))
        self.fit_button.pack(side=tk.BOTTOM, pady=5)
        self.auto_button.pack(side=tk.BOTTOM, pady=5)

        self.fitpara_frame = FitparaFrame(master=self, width=width-20, height=35)
        self.fitpara_frame.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

        self.pack()

    def update_nav_buttons(self):
        '''Update the navigation buttons.
        '''
        mode = self.master.param['mode']
        if mode == 'ident':
            aperture = self.master.param['aperture']
        
            if aperture == self.master.spec['aperture'].min():
                state = tk.DISABLED
            else:
                state = tk.NORMAL
            self.prev_button.config(state=state)
            
            if aperture == self.master.spec['aperture'].max():
                state = tk.DISABLED
            else:
                state = tk.NORMAL
            self.next_button.config(state=state)
        elif mode == 'fit':
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            pass

    def update_aperture_label(self):
        '''Update the order information to be displayed on the top.
        '''
        mode     = self.master.param['mode']
        aperture = self.master.param['aperture']
        k        = self.master.param['k']
        offset   = self.master.param['offset']

        if mode == 'ident':
            if None in (k, offset):
                order = '?'
            else:
                order = str(k*aperture + offset)
            text = 'Order %s (Aperture %d)'%(order, aperture)
            self.order_label.config(text=text)
        elif mode == 'fit':
            self.order_label.config(text='')
        else:
            pass


class LineTable(tk.Frame):
    '''A table for the input spectral lines embedded in the :class:`InfoFrame`.
    
    Args:
        master (Tkinter widget): Parent widget.
        width (integer): Width of line table.
        height (integer): Height of line table.
        identlist (dict): Dict of identified lines.
        linelist (list): List of wavelength standards.
    '''
    def __init__(self, master, width, height, identlist, linelist):
        '''Constructor of :class:`LineTable`.
        '''
        self.master = master

        font = ('Arial', 10)

        tk.Frame.__init__(self, master=master, width=width, height=height)

        self.tool_frame = tk.Frame(master=self, width=width, height=40)
        self.tool_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        self.search_text = tk.StringVar()
        self.search_entry = tk.Entry(master=self.tool_frame, width=10,
                                     font=font, textvariable=self.search_text)
        self.search_entry.pack(side=tk.LEFT, fil=tk.Y, padx=0)

        # create 3 buttons
        self.clr_button = tk.Button(master=self.tool_frame, text='Clear',
                                    font=font, width=5,
                                    command=self.on_clear_search)
        self.add_button = tk.Button(master=self.tool_frame, text='Add',
                                    font=font, width=5,
                                    command=master.master.on_add_ident)
        self.del_button = tk.Button(master=self.tool_frame, text='Del',
                                    font=font, width=5,
                                    command=master.master.on_delete_ident)

        # put 3 buttons
        self.del_button.pack(side=tk.RIGHT, padx=(5,0))
        self.add_button.pack(side=tk.RIGHT, padx=(5,0))
        self.clr_button.pack(side=tk.RIGHT, padx=(5,0))

        # update status of 3 buttons
        self.clr_button.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.del_button.config(state=tk.DISABLED)

        # create line table
        self.data_frame = tk.Frame(master=self, width=width)
        self.data_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # create line tree
        self.line_tree = ttk.Treeview(master  = self.data_frame,
                                      columns    = ('wv', 'species', 'status'),
                                      show       = 'headings',
                                      style      = 'Treeview',
                                      height     = 22,
                                      selectmode ='browse')
        self.line_tree.bind('<Button-1>', self.on_click_item)

        self.scrollbar = tk.Scrollbar(master = self.data_frame,
                                      orient = tk.VERTICAL,
                                      width  = 20)

        self.line_tree.column('wv',      width=160)
        self.line_tree.column('species', width=140)
        self.line_tree.column('status',  width=width-160-140-20)
        self.line_tree.heading('wv',      text=u'\u03bb in air (\xc5)')
        self.line_tree.heading('species', text='Species')
        self.line_tree.heading('status',  text='Status')
        self.line_tree.config(yscrollcommand=self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=35)
        style.configure('Treeview.Heading', font=('Arial', 10))

        self.scrollbar.config(command=self.line_tree.yview)

        self.item_lst = []
        for line in linelist:
            wv, species = line
            iid = self.line_tree.insert('',tk.END,
                    values=(wv, species, ''), tags='normal')
            self.item_lst.append((iid,  wv))
        self.line_tree.tag_configure('normal', font=('Arial', 10))

        self.line_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.data_frame.pack(side=tk.TOP, fill=tk.Y)
        self.pack()

    def on_clear_search(self):
        '''Clear the search bar.
        '''
        # clear the search bar
        self.search_text.set('')

        # clear the identified line
        self.master.master.param['ident']=None

        # de-select the line table
        sel_items = self.line_tree.selection()
        self.line_tree.selection_remove(sel_items)
        #self.line_tree.selection_clear()
        # doesn't work?

        # update the status of 3 button
        self.clr_button.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.del_button.config(state=tk.DISABLED)

        # replot
        self.master.master.plot_aperture()

        # set focus to canvas
        self.master.master.plot_frame.canvas.get_tk_widget().focus()
        
    def on_click_item(self, event):
        '''Event response function for clicking lines.
        '''

        identlist = self.master.master.identlist
        aperture = self.master.master.param['aperture']

        # find the clicked item
        item = self.line_tree.identify_row(event.y)
        values = self.line_tree.item(item, 'values')
        # put the wavelength into the search bar
        self.search_text.set(values[0])
        # update status
        self.clr_button.config(state=tk.NORMAL)


        # find if the clicked line is in ident list.
        # if yes, replot the figure with idented line with blue color, and set
        # the delete button to normal. Otherwise, replot the figure with black,
        # and disable the delete button.

        if aperture in identlist:

            list1 = identlist[aperture]
        
            wv_diff = np.abs(list1['wavelength'] - float(values[0]))
            mindiff = wv_diff.min()
            argmin  = wv_diff.argmin()
            if mindiff < 1e-3:
                # the selected line is in identlist of this aperture
                xpos = list1[argmin]['pixel']
                for line, text in self.master.master.ident_objects:
                    if abs(line.get_xdata()[0] - xpos)<1e-3:
                        plt.setp(line, color='b')
                        plt.setp(text, color='b')
                    else:
                        plt.setp(line, color='k')
                        plt.setp(text, color='k')
                # update the status of del button
                self.del_button.config(state=tk.NORMAL)
            else:
                # the selected line is not in identlist of this aperture
                for line, text in self.master.master.ident_objects:
                    plt.setp(line, color='k')
                    plt.setp(text, color='k')
                # update the status of del button
                self.del_button.config(state=tk.DISABLED)
            
            self.master.master.plot_frame.canvas.draw()

        else:
            # if the current aperture is not in identlist, do nothing
            pass

class FitparaFrame(tk.Frame):
    '''Frame for the fitting parameters embedded in the :class:`InfoFrame`.
    
    Args:
        master (Tkinter widget): Parent widget.
        width (integer): Width of frame.
        height (integer): Height of frame.
    '''
    def __init__(self, master, width, height):
        '''Constructor of :class:`FitparaFrame`.
        '''

        self.master = master

        font = ('Arial', 10)

        tk.Frame.__init__(self, master, width=width, height=height)

        # the first row
        self.row1_frame = tk.Frame(master=self, width=width)

        self.xorder_label = tk.Label(master = self.row1_frame,
                                     text   = 'X ord =',
                                     font   = font)

        # spinbox for adjusting xorder
        self.xorder_str = tk.StringVar()
        self.xorder_str.set(master.master.param['xorder'])
        self.xorder_box = tk.Spinbox(master = self.row1_frame,
                                     from_        = 1,
                                     to_          = 10,
                                     font         = font,
                                     width        = 2,
                                     textvariable = self.xorder_str,
                                     command      = self.on_change_xorder)

        self.yorder_label = tk.Label(master = self.row1_frame,
                                     text   = 'Y ord =',
                                     font   = font)
        # spinbox for adjusting yorder
        self.yorder_str = tk.StringVar()
        self.yorder_str.set(master.master.param['yorder'])
        self.yorder_box = tk.Spinbox(master       = self.row1_frame,
                                     from_        = 1,
                                     to_          = 10,
                                     font         = font,
                                     width        = 2,
                                     textvariable = self.yorder_str,
                                     command      = self.on_change_yorder)

        self.maxiter_label  = tk.Label(master = self.row1_frame,
                                       text   = 'N =',
                                       font   = font)
        self.maxiter_str = tk.StringVar()
        self.maxiter_str.set(master.master.param['maxiter'])
        self.maxiter_box = tk.Spinbox(master       = self.row1_frame,
                                      from_        = 1,
                                      to_          = 20,
                                      font         = font,
                                      width        = 2,
                                      textvariable = self.maxiter_str,
                                      command      = self.on_change_maxiter)

        self.xorder_label.pack(side=tk.LEFT)
        self.xorder_box.pack(side=tk.LEFT)
        self.yorder_label.pack(side=tk.LEFT, padx=(10,0))
        self.yorder_box.pack(side=tk.LEFT)
        self.maxiter_label.pack(side=tk.LEFT, padx=(10,0))
        self.maxiter_box.pack(side=tk.LEFT)

        self.row1_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,2))

        # the second row
        self.row2_frame = tk.Frame(master=self, width=width)

        self.clip_label = tk.Label(master = self.row2_frame,
                                   text   = 'Clipping =',
                                   font   = font,
                                   width  = width,
                                   anchor = tk.W)
        self.clip_scale = tk.Scale(master     = self.row2_frame,
                                   from_      = 1.0,
                                   to         = 5.0,
                                   orient     = tk.HORIZONTAL,
                                   resolution = 0.1,
                                   command    = self.on_change_clipping)
        self.clip_scale.set(master.master.param['clipping'])

        self.clip_label.pack(side=tk.TOP)
        self.clip_scale.pack(side=tk.TOP, fill=tk.X)
        self.row2_frame.pack(side=tk.TOP, fill=tk.X)

        self.pack()

    def on_change_xorder(self):
        '''Response function of changing order of polynomial along x-axis.
        '''
        self.master.master.param['xorder'] = int(self.xorder_box.get())

    def on_change_yorder(self):
        '''Response function of changing order of polynomial along y-axis.
        '''
        self.master.master.param['yorder'] = int(self.yorder_box.get())

    def on_change_maxiter(self):
        '''Response function of changing maximum number of iteration.
        '''
        self.master.master.param['maxiter'] = int(self.maxiter_box.get())

    def on_change_clipping(self, value):
        '''Response function of changing clipping value.
        '''
        self.master.master.param['clipping'] = float(value)

class CalibWindow(tk.Frame):
    '''Frame of the wavelength calibration window.

    Args:
        master (:class:`tk.TK`): Tkinter root window.
        width (integer): Width of window.
        height (integer): Height of window.
        dpi (integer): DPI of figure.
        spec (:class:`numpy.dtype`): Spectra data.
        filename (string): Filename of the spectra data.
        figfilename (string): Filename of the output wavelength calibration
            figure.
        identlist (dict): Identification line list.
        linelist (list): List of wavelength standards (wavelength, species).
        channel (string): Name of channel in the input spectra.
        window_size (integer): Size of the window in pixel to search for line
            peaks.
        xorder (integer): Degree of polynomial along X direction.
        yorder (integer): Degree of polynomial along Y direction.
        maxiter (integer): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        snr_threshold (float): Minimum S/N of the spectral lines to be accepted
            in the wavelength fitting.
    '''
    def __init__(self, master, width, height, dpi, spec, filename, figfilename,
            identlist, linelist, channel, window_size, xorder, yorder, maxiter,
            clipping, snr_threshold):
        '''Constructor of :class:`CalibWindow`.
        '''
        
        self.master    = master
        self.spec      = spec
        self.identlist = identlist
        self.linelist  = linelist

        tk.Frame.__init__(self, master, width=width, height=height)


        self.param = {
            'mode':          'ident',
            'aperture':      self.spec['aperture'].min(),
            'filename':      filename,
            'figfilename':   figfilename,
            'channel':       channel,
            'aperture_min':  self.spec['aperture'].min(),
            'aperture_max':  self.spec['aperture'].max(),
            'npixel':        self.spec['points'].max(),
            # parameters of displaying
            'xlim':          {},
            'ylim':          {},
            'ident':         None,
            # parameters of converting aperture and order
            'k':             None,
            'offset':        None,
            # wavelength fitting parameters
            'window_size':   window_size,
            'xorder':        xorder,
            'yorder':        yorder,
            'maxiter':       maxiter,
            'clipping':      clipping,
            'snr_threshold': snr_threshold,
            # wavelength fitting results
            'std':           0,
            'coeff':         np.array([]),
            'nuse':          0,
            'ntot':          0,
            }

        for row in self.spec:
            aperture = row['aperture']
            self.param['xlim'][aperture] = (0, row['points']-1)
            self.param['ylim'][aperture] = (None, None)

        # determine widget size
        info_width    = 500
        info_height   = height
        canvas_width  = width - info_width
        canvas_height = height
        # generate plot frame and info frame
        self.plot_frame = PlotFrame(master    = self,
                                    width     = canvas_width,
                                    height    = canvas_height,
                                    dpi       = dpi,
                                    identlist = self.identlist,
                                    linelist  = self.linelist,
                                    )
        self.info_frame = InfoFrame(master    = self,
                                    width     = info_width,
                                    height    = info_height,
                                    identlist = self.identlist,
                                    linelist  = self.linelist,
                                    )
        # pack plot frame and info frame
        self.plot_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        self.pack()

        self.plot_aperture()

        self.update_fit_buttons()

    def fit(self):
        '''Fit the wavelength and plot the solution in the figure.'''

        coeff, std, k, offset, nuse, ntot = fit_wv(
                identlist = self.identlist, 
                npixel    = self.param['npixel'],
                xorder    = self.param['xorder'],
                yorder    = self.param['yorder'],
                maxiter   = self.param['maxiter'],
                clipping  = self.param['clipping'],
                )
        self.param['coeff']  = coeff
        self.param['std']    = std
        self.param['k']      = k
        self.param['offset'] = offset
        self.param['nuse']   = nuse
        self.param['ntot']   = ntot

        self.plot_wv()

        # udpdate the order/aperture string
        aperture = self.param['aperture']
        order = k*aperture + offset
        text = 'Order %d (Aperture %d)'%(order, aperture)
        self.info_frame.order_label.config(text=text)

        self.update_fit_buttons()

    def recenter(self):
        '''Relocate the peaks for all the identified lines.
        '''
        for aperture, list1 in self.identlist.items():
            mask = (self.spec['aperture'] == aperture)
            specdata = self.spec[mask][0]
            flux = specdata['flux']
            for row in list1:
                pix = int(round(row['pixel']))
                window_size = self.param['window_size']
                _, _, param, _ = find_local_peak(flux, pix, window_size)
                peak_x = param[1]
                row['pixel'] = peak_x

        # replot
        self.plot_aperture()

    def clearall(self):
        '''Clear all the identified lines.'''

        self.identlist = {}

        self.param['k']      = None
        self.param['offset'] = None
        self.param['std']    = 0
        self.param['coeff']  = np.array([])
        self.param['nuse']   = 0
        self.param['ntot']   = 0

        info_frame = self.info_frame
        # update the status of 3 buttons
        info_frame.line_frame.clr_button.config(state=tk.DISABLED)
        info_frame.line_frame.add_button.config(state=tk.DISABLED)
        info_frame.line_frame.del_button.config(state=tk.DISABLED)

        # update buttons
        info_frame.recenter_button.config(state=tk.DISABLED)
        info_frame.clearall_button.config(state=tk.DISABLED)
        info_frame.switch_button.config(state=tk.DISABLED)
        info_frame.auto_button.config(state=tk.DISABLED)

        # replot
        self.plot_aperture()
        
    def auto_identify(self):
        '''Identify all lines in the wavelength standard list automatically.
        '''
        k       = self.param['k']
        offset  = self.param['offset']
        coeff   = self.param['coeff']
        npixel  = self.param['npixel']
        channel = self.param['channel']
        for aperture in sorted(self.spec['aperture']):
            mask = self.spec['aperture'] == aperture
            flux = self.spec[mask][0]['flux']

            # scan every order and find the upper and lower limit of wavelength
            order = k*aperture + offset

            # generated the wavelengths for every pixel in this oirder
            x = np.arange(npixel)
            wvs = get_wv_val(coeff, npixel, x, np.repeat(order, x.size))
            wv1 = min(wvs[0], wvs[-1])
            wv2 = max(wvs[0], wvs[-1])

            has_insert = False
            # now scan the linelist
            for line in self.linelist:
                if line[0]<wv1:
                    continue
                elif line[0]>wv2:
                    break
                else:
                    # wavelength in the range of this order
                    # check if this line has already been identified
                    if is_identified(line[0], self.identlist, aperture):
                        continue

                    # now has not been identified. find peaks for this line
                    diff = np.abs(wvs - line[0])
                    i = diff.argmin()
                    i1, i2, param, std = find_local_peak(flux, i, self.param['window_size'])
                    peak_x = param[1]

                    if param[0] < 0.:
                        continue
                    if param[1] < i1 or param[1] > i2:
                        continue
                    if param[2] < 1.0:
                        continue
                    if param[2] > 50. or param[2] < 1.0:
                        continue
                    if param[3] < -0.5*param[0]:
                        continue
                    snr = param[0]/std
                    if snr < self.param['snr_threshold']:
                        continue
                    
                    if False:
                        fig = plt.figure(figsize=(6,4),tight_layout=True)
                        ax = fig.gca()
                        ax.plot(np.arange(i1,i2), flux[i1:i2], 'ro')
                        newx = np.arange(i1, i2, 0.1)
                        ax.plot(newx, gaussian_bkg(param[0],param[1],param[2],param[3], newx), 'b-')
                        ax.axvline(x=param[1], color='k',ls='--')
                        y1,y2 = ax.get_ylim()
                        ax.text(0.9*i1+0.1*i2, 0.1*y1+0.9*y2, 'A=%.1f'%param[0])
                        ax.text(0.9*i1+0.1*i2, 0.2*y1+0.8*y2, 'BKG=%.1f'%param[3])
                        ax.text(0.9*i1+0.1*i2, 0.3*y1+0.7*y2, 'FWHM=%.1f'%param[2])
                        ax.text(0.9*i1+0.1*i2, 0.4*y1+0.6*y2, 'SNR=%.1f'%snr)
                        ax.set_xlim(i1,i2)
                        fig.savefig('tmp/%d-%d-%d.png'%(aperture, i1, i2))
                        plt.close(fig)

                    
                    if aperture not in self.identlist:
                        self.identlist[aperture] = np.array([], dtype=identlinetype)

                    item = np.array((channel, aperture, order, peak_x, line[0], snr, True, 0.0, 'a'),
                                    dtype=identlinetype)
                    self.identlist[aperture] = np.append(self.identlist[aperture], item)
                    has_insert = True
                    #print('insert', aperture, line[0], peak_x, i)

            # resort this order if there's new line inserted
            if has_insert:
                self.identlist[aperture] = np.sort(self.identlist[aperture], order='pixel')

        self.fit()

    def switch(self):
        '''Response funtion of switching between "ident" and "fit" mode.
        '''

        if self.param['mode']=='ident':
            # switch to fit mode
            self.param['mode']='fit'

            self.plot_wv()

            self.info_frame.switch_button.config(text='Identify')

        elif self.param['mode']=='fit':
            # switch to ident mode
            self.param['mode']='ident'

            self.plot_aperture()

            self.info_frame.switch_button.config(text='Plot')
        else:
            pass

        # update order navigation and aperture label
        self.info_frame.update_nav_buttons()
        self.info_frame.update_aperture_label()

    def next_aperture(self):
        '''Response function of pressing the next aperture.'''
        if self.param['aperture'] < self.spec['aperture'].max():
            self.param['aperture'] += 1
            self.plot_aperture()

    def prev_aperture(self):
        '''Response function of pressing the previous aperture.'''
        if self.param['aperture'] > self.spec['aperture'].min():
            self.param['aperture'] -= 1
            self.plot_aperture()

    def plot_aperture(self):
        '''Plot a specific aperture in the figure.
        '''
        aperture = self.param['aperture']
        mask = (self.spec['aperture'] == aperture)
        specdata = self.spec[mask][0]
        xdata = np.arange(specdata['points'])
        ydata = specdata['flux']

        # redraw spectra in ax1
        ax1 = self.plot_frame.ax1
        fig = self.plot_frame.fig
        ax1.cla()
        ax1.plot(xdata, ydata, 'r-')
        x1, x2 = self.param['xlim'][aperture]
        y1, y2 = self.param['ylim'][aperture]
        if y1 is None:
            y1 = ax1.get_ylim()[0]
        if y2 is None:
            y2 = ax1.get_ylim()[1]

        #x1, x2 = xdata[0], xdata[-1]
        #y1, y2 = ax1.get_ylim()
        y1 = min(y1, 0)
        # plot identified line list
        # calculate ratio = value/pixel
        bbox = ax1.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
        axwidth_pixel = bbox.width*fig.dpi
        pix_ratio = abs(x2-x1)/axwidth_pixel

        # plot identified lines with vertial dash lines
        if aperture in self.identlist and len(self.identlist[aperture])>0:
            self.ident_objects = []
            list1 = self.identlist[aperture]
            for item in list1:
                pixel      = item['pixel']
                wavelength = item['wavelength']

                # draw vertial dash line
                line = ax1.axvline(pixel, ls='--', color='k')

                # draw text
                x = pixel+pix_ratio*10
                y = 0.4*y1+0.6*y2
                text = ax1.text(x, y, '%.4f'%wavelength, color='k',
                                rotation='vertical', fontstyle='italic',
                                fontsize=10)
                self.ident_objects.append((line, text))

        # plot the temporarily identified line
        if self.param['ident'] is not None:
            ax1.axvline(self.param['ident'], linestyle='--', color='k')


        # update the aperture number
        ax1._aperture_text.set_text('Aperture %d'%aperture)
        ax1.set_ylim(y1, y2)
        ax1.set_xlim(x1, x2)
        ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
        ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Flux')

        self.plot_frame.canvas.draw()

        # update order navigation and aperture label
        self.info_frame.update_nav_buttons()
        self.info_frame.update_aperture_label()

    def plot_wv(self):
        '''A wrap for plotting the wavelength solution.'''

        aperture_lst = np.arange(self.param['aperture_min'], 
                                 self.param['aperture_max']+1)

        kwargs = {
                'offset': self.param['offset'],
                'k':      self.param['k'],
                'coeff':  self.param['coeff'],
                'npixel': self.param['npixel'],
                'std':    self.param['std'],
                'nuse':   self.param['nuse'],
                'ntot':   self.param['ntot'],
                }

        if self.param['mode']=='fit':
            plot_ax1 = True
        else:
            plot_ax1 = False

        self.plot_frame.fig.plot_solution(self.identlist, aperture_lst, plot_ax1, **kwargs)

        self.plot_frame.canvas.draw()
        self.plot_frame.fig.savefig(self.param['figfilename'])

    def on_click(self, event):
        '''Response function of clicking the axes. Double click means find the
        local peak and prepare to add a new identified line.
        '''
        # double click on ax1: want to add a new identified line
        if event.inaxes == self.plot_frame.ax1 and event.dblclick:
            fig = self.plot_frame.fig
            ax1 = self.plot_frame.ax1
            line_frame = self.info_frame.line_frame
            aperture = self.param['aperture']
            if aperture in self.identlist:
                list1 = self.identlist[aperture]

            # get width of current ax in pixel
            x1, x2 = ax1.get_xlim()
            y1, y2 = ax1.get_ylim()
            iarray = fig.dpi_scale_trans.inverted()
            bbox = ax1.get_window_extent().transformed(iarray)
            width, height = bbox.width, bbox.height
            axwidth_pixel = width*fig.dpi
            # get physical Values Per screen Pixel (VPP) in x direction
            vpp = abs(x2-x1)/axwidth_pixel

            # check if peak has already been identified
            if aperture in self.identlist:

                dist = np.array([abs(line.get_xdata()[0] - event.xdata)/vpp
                                 for line, text in self.ident_objects])

                if dist.min() < 5:
                    # found. change the color of this line
                    imin = dist.argmin()
                    for i, (line, text) in enumerate(self.ident_objects):
                        if i == imin:
                            plt.setp(line, color='b')
                            plt.setp(text, color='b')
                        else:
                            plt.setp(line, color='k')
                            plt.setp(text, color='k')
                    # redraw the canvas
                    self.plot_frame.canvas.draw()

                    wv = list1[imin]['wavelength']

                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        item = record[0]
                        wave = record[1]
                        if abs(wv - wave)<1e-3:
                            break

                    line_frame.line_tree.selection_set(item)
                    pos = i/float(len(line_frame.item_lst))
                    line_frame.line_tree.yview_moveto(pos)

                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(wv))

                    # update the status of 3 buttons
                    line_frame.clr_button.config(state=tk.NORMAL)
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.NORMAL)

                    # end this function without more actions
                    return True
                else:
                    # not found. all of the colors are normal
                    for line, text in self.ident_objects:
                        plt.setp(line, color='k')
                        plt.setp(text, color='k')

                    # clear the search bar
                    line_frame.search_text.set('')

                    # de-select the line table
                    sel_items = line_frame.line_tree.selection()
                    line_frame.line_tree.selection_remove(sel_items)

                    # update the status of 3 button
                    line_frame.clr_button.config(state=tk.DISABLED)
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.DISABLED)

                    # continue to act

            mask = (self.spec['aperture'] == aperture)
            specdata = self.spec[mask][0]
            flux = specdata['flux']

            # find peak
            # first, find the local maximum around the clicking point
            # the searching window is set to be +-5 pixels
            i0 = int(event.xdata)
            i1 = max(int(round(i0 - 5*vpp)), 0)
            i2 = min(int(round(i0 + 5*vpp)), flux.size)
            local_max = i1 + flux[i1:i2].argmax()
            # now found the local max point
            window_size = self.param['window_size']
            _, _, param, _ = find_local_peak(flux, local_max, window_size)
            peak_x = param[1]

            self.param['ident'] = peak_x

            # temporarily plot this line
            self.plot_aperture()

            line_frame.clr_button.config(state=tk.NORMAL)

            # guess the input wavelength
            guess_wv = guess_wavelength(peak_x, aperture, self.identlist,
                                        self.linelist, self.param)

            if guess_wv is None:
                # wavelength guess failed
                #line_frame.search_entry.focus()
                # update buttons
                line_frame.add_button.config(state=tk.NORMAL)
                line_frame.del_button.config(state=tk.DISABLED)
            else:
                # wavelength guess succeed

                # check whether wavelength has already been identified
                if is_identified(guess_wv, self.identlist, aperture):
                    # has been identified, do nothing
                    # update buttons
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.NORMAL)
                else:
                    # has not been identified yet
                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(guess_wv))
                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        iid  = record[0]
                        wave = record[1]
                        if abs(guess_wv - wave)<1e-3:
                            break
                    line_frame.line_tree.selection_set(iid)
                    pos = i/float(len(line_frame.item_lst))
                    line_frame.line_tree.yview_moveto(pos)
                    # update buttons
                    line_frame.add_button.config(state=tk.NORMAL)
                    line_frame.del_button.config(state=tk.DISABLED)
                    # unset focus
                    self.focus()

    def on_add_ident(self):
        '''Response function of identifying a new line.
        '''
        aperture = self.param['aperture']
        channel  = self.param['channel']
        k        = self.param['k']
        offset   = self.param['offset']

        line_frame = self.info_frame.line_frame
        
        if aperture not in self.identlist:
            self.identlist[aperture] = np.array([], dtype=identlinetype)

        list1 = self.identlist[aperture]

        pixel = self.param['ident']
        selected_iid_lst = line_frame.line_tree.selection()
        iid = selected_iid_lst[0]
        wavelength = float(line_frame.line_tree.item(iid, 'values')[0])
        line_frame.line_tree.selection_remove(selected_iid_lst)
        
        # find the insert position
        insert_pos = np.searchsorted(list1['pixel'], pixel)

        if None in [k, offset]:
            order = 0
        else:
            order = k*aperture + offset

        item = np.array((channel, aperture, order, pixel, wavelength, -1., True, 0.0, 'm'),
                        dtype=identlinetype)
        # insert into identified line list
        self.identlist[aperture] = np.insert(self.identlist[aperture],
                                             insert_pos, item)

        # reset ident
        self.param['ident'] = None

        # reset the line table
        line_frame.search_text.set('')
        
        # update the status of 3 buttons
        line_frame.clr_button.config(state=tk.NORMAL)
        line_frame.add_button.config(state=tk.DISABLED)
        line_frame.del_button.config(state=tk.NORMAL)

        self.update_fit_buttons()

        # replot
        self.plot_aperture()

    def on_delete_ident(self):
        '''Response function of deleting an identified line.
        '''
        line_frame = self.info_frame.line_frame
        target_wv = float(line_frame.search_text.get())
        aperture = self.param['aperture']
        list1 = self.identlist[aperture]

        wv_diff = np.abs(list1['wavelength'] - target_wv)
        mindiff = wv_diff.min()
        argmin  = wv_diff.argmin()
        if mindiff < 1e-3:
            # delete this line from ident list
            list1 = np.delete(list1, argmin)
            self.identlist[aperture] = list1

            # clear the search bar
            line_frame.search_text.set('')

            # de-select the line table
            sel_items = line_frame.line_tree.selection()
            line_frame.line_tree.selection_remove(sel_items)

            # update the status of 3 buttons
            line_frame.clr_button.config(state=tk.DISABLED)
            line_frame.add_button.config(state=tk.DISABLED)
            line_frame.del_button.config(state=tk.DISABLED)

            # update fit buttons
            self.update_fit_buttons()

            # replot
            self.plot_aperture()

    def on_draw(self, event):
        '''
        Response function of drawing.
        '''
        if self.param['mode'] == 'ident':
            ax1 = self.plot_frame.ax1
            aperture = self.param['aperture']
            self.param['xlim'][aperture] = ax1.get_xlim()
            self.param['ylim'][aperture] = ax1.get_ylim()

    def update_fit_buttons(self):
        '''
        Update the status of fitting buttons.
        '''
        nident = 0
        for aperture, list1 in self.identlist.items():
            nident += list1.size

        xorder = self.param['xorder']
        yorder = self.param['yorder']

        info_frame = self.info_frame

        if nident > (xorder+1)*(yorder+1) and len(self.identlist) > yorder+1:
            info_frame.fit_button.config(state=tk.NORMAL)
        else:
            info_frame.fit_button.config(state=tk.DISABLED)

        if len(self.param['coeff'])>0:
            info_frame.switch_button.config(state=tk.NORMAL)
            info_frame.auto_button.config(state=tk.NORMAL)
        else:
            info_frame.switch_button.config(state=tk.DISABLED)
            info_frame.auto_button.config(state=tk.DISABLED)

    def update_batch_buttons(self):
        '''
        Update the status of batch buttons (recenter and clearall).
        '''
        # count how many identified lines
        nident = 0
        for aperture, list1 in self.identlist.items():
            nident += list1.size

        info_frame = self.info_frame

        if nident > 0:
            info_frame.recenter_button.config(state=tk.NORMAL)
            info_frame.clearall_button.config(state=tk.NORMAL)
        else:
            info_frame.recenter_button.config(state=tk.DISABLED)
            info_frame.clearall_button.config(state=tk.DISABLED)



def wvcalib(filename, identfilename, figfilename, linelist, channel, window_size=13,
        xorder=3, yorder=3, maxiter=10, clipping=3, snr_threshold=10):
    '''Wavelength calibration.

    Args:
        filename (string): Filename of the 1-D spectra.
        identfilename (string): Filename of wavelength identification.
        figfilename (string): Filename of the output wavelength figure.
        linelist (string): Name of wavelength standard file.
        channel (string): Name of the input channel.
        window_size (integer): Size of the window in pixel to search for the
            lines.
        xorder (integer): Degree of polynomial along X direction.
        yorder (integer): Degree of polynomial along Y direction.
        maxiter (integer): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        snr_threshold (float): Minimum S/N of the spectral lines to be accepted
            in the wavelength fitting.
    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.array`): Coefficient array.
            * **npixel** (*integer*): Number of pixels along the main dispersion
              direction.
            * **k** (*integer*): Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*integer*): Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*): Standard deviation of wavelength fitting in Å.
            * **nuse** (*integer*): Number of lines used in the wavelength
              fitting.
            * **ntot** (*integer*): Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*): Dict of identified lines.
            * **window_size** (*integer*): Length of window in searching the
              line centers.
            * **xorder** (*integer*): Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*integer*): Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*integer*): Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*): Clipping value of the wavelength fitting.
            * **snr_threshold** (*float*): Minimum S/N of the spectral lines to
              be accepted in the wavelength fitting.
        
    See also:
        :func:`recalib`

    '''

    spec = fits.getdata(filename)
    if channel != '':
        mask = spec['channel']==channel
        spec = spec[mask]

    # initialize fitting list
    if os.path.exists(identfilename):
        identlist, _ = load_ident(identfilename, channel=channel)
    else:
        identlist = {}

    # load the wavelengths
    linefilename = search_linelist(linelist)
    if linefilename is None:
        print('Error: Cannot find linelist file: %s'%linelist)
        exit()
    line_list = load_linelist(linefilename)

    # display an interactive figure
    # reset keyboard shortcuts
    mpl.rcParams['keymap.pan']        = ''   # reset 'p'
    mpl.rcParams['keymap.fullscreen'] = ''   # reset 'f'
    mpl.rcParams['keymap.back']       = ''   # reset 'c'

    # initialize tkinter window
    master = tk.Tk()
    master.resizable(width=False, height=False)

    screen_width  = master.winfo_screenwidth()
    screen_height = master.winfo_screenheight()

    fig_width  = 2500
    fig_height = 1500
    fig_dpi    = 150
    if None in [fig_width, fig_height]:
        # detremine window size and position
        window_width = int(screen_width-200)
        window_height = int(screen_height-200)
    else:
        window_width = fig_width + 500
        window_height = fig_height + 34

    x = int((screen_width-window_width)/2.)
    y = int((screen_height-window_height)/2.)
    master.geometry('%dx%d+%d+%d'%(window_width, window_height, x, y))

    # display window
    calibwindow = CalibWindow(master,
                              width         = window_width,
                              height        = window_height-34,
                              dpi           = fig_dpi,
                              spec          = spec,
                              filename      = filename,
                              figfilename   = figfilename,
                              channel       = channel,
                              identlist     = identlist,
                              linelist      = line_list,
                              window_size   = window_size,
                              xorder        = xorder,
                              yorder        = yorder,
                              maxiter       = maxiter,
                              clipping      = clipping,
                              snr_threshold = snr_threshold,
                              )

    master.mainloop()

    # organize results
    result = {
              'coeff':         calibwindow.param['coeff'],
              'npixel':        calibwindow.param['npixel'],
              'k':             calibwindow.param['k'],
              'offset':        calibwindow.param['offset'],
              'std':           calibwindow.param['std'],
              'nuse':          calibwindow.param['nuse'],
              'ntot':          calibwindow.param['ntot'],
              'identlist':     calibwindow.identlist,
              'window_size':   calibwindow.param['window_size'],
              'xorder':        calibwindow.param['xorder'],
              'yorder':        calibwindow.param['yorder'],
              'maxiter':       calibwindow.param['maxiter'],
              'clipping':      calibwindow.param['clipping'],
              'snr_threshold': calibwindow.param['snr_threshold'],
            }

    # save ident list
    if len(calibwindow.identlist)>0:
        save_ident(calibwindow.identlist, calibwindow.param['coeff'], identfilename, channel)

    return result

def fit_wv(identlist, npixel, xorder, yorder, maxiter, clipping):
    '''Fit the wavelength using 2-D polynomial.
    
    Args:
        identlist (dict): Dict of identification lines for different apertures.
        npixel (integer): Number of pixels for each order.
        xorder (integer): Order of polynomial along X direction.
        yorder (integer): Order of polynomial along Y direction.
        maxiter (integer): Maximim number of iterations in the polynomial
            fitting.
        clipping (float): Threshold of sigma-clipping.

    Returns:
        tuple: A tuple containing:
        
            * **coeff** (:class:`numpy.array`): Coefficients array.
            * **std** (*float*): Standard deviation.
            * **k** (*int*): *k* in the relationship between aperture numbers
              and diffraction orders: `order = k*aperture + offset`.
            * **offset** (*int*): *offset* in the relationship between aperture
              numbers and diffraction orders: `order = k*aperture + offset`.
            * **nuse** (*int*): Number of lines used in the fitting.
            * **ntot** (*int*): Number of lines found.

    See also:
        :func:`get_wv_val`

    '''
    # find physical order
    k, offset = find_order(identlist, npixel)

    # convert indent_line_lst into fitting inputs
    fit_p_lst, fit_o_lst, fit_w_lst = [], [], []
    # this list is used to find the position (aperture, no) of each line
    lineid_lst = []
    for aperture, list1 in sorted(identlist.items()):
        order = k*aperture + offset
        #norm_order = 50./order
        #norm_order = order/50.
        list1['order'][:] = order
        for iline, item in enumerate(list1):
            norm_pixel = item['pixel']*2/(npixel-1) - 1
            fit_p_lst.append(norm_pixel)
            fit_o_lst.append(order)
            #fit_o_lst.append(norm_order)
            #fit_w_lst.append(item['wavelength'])
            fit_w_lst.append(item['wavelength']*order)
            lineid_lst.append((aperture, iline))
    fit_p_lst = np.array(fit_p_lst)
    fit_o_lst = np.array(fit_o_lst)
    fit_w_lst = np.array(fit_w_lst)

    # sigma clipping fitting
    mask = np.ones_like(fit_p_lst, dtype=np.bool)
    for nite in range(maxiter):
        coeff = polyfit2d(fit_p_lst[mask], fit_o_lst[mask], fit_w_lst[mask],
                          xorder=xorder, yorder=yorder)
        res_lst = fit_w_lst - polyval2d(fit_p_lst, fit_o_lst, coeff)
        res_lst = res_lst/fit_o_lst

        mean = res_lst[mask].mean(dtype=np.float64)
        std  = res_lst[mask].std(dtype=np.float64)
        m1 = res_lst > mean - clipping*std
        m2 = res_lst < mean + clipping*std
        new_mask = m1*m2
        if new_mask.sum() == mask.sum():
            break
        else:
            mask = new_mask

    # convert mask back to ident_line_lst
    for lineid, ma, res in zip(lineid_lst, mask, res_lst):
        aperture, iline = lineid
        identlist[aperture][iline]['mask']     = ma
        identlist[aperture][iline]['residual'] = res

    # number of lines and used lines
    nuse = mask.sum()
    ntot = fit_w_lst.size
    return coeff, std, k, offset, nuse, ntot

def get_wv_val(coeff, npixel, pixel, order):
    '''Get the wavelength solution.
    
    Args:
        coeff (:class:`numpy.array`): 2-D Coefficient array.
        npixel (integer): Number of pixels along the main dispersion direction.
        pixel (integer or :class:`numpy.array`): Pixel coordinates.
        order (integer or :class:`numpy.array`): Diffraction order number. Must
            have the same length as **pixel**.
    Returns:
        float or :class:`numpy.array`: Wavelength solution of the given pixels.
    See also:
        :func:`fit_wv`

    '''
    # convert aperture to order
    norm_pixel = pixel*2./(npixel-1) - 1
    #norm_order  = 50./order
    #norm_order  = order/50.
    return polyval2d(norm_pixel, order, coeff)/order

def guess_wavelength(x, aperture, identlist, linelist, param):
    '''Guess wavelength according to the identified lines.
    First, try to guess the wavelength from the identified lines in the same
    order (aperture) by fitting polynomials.
    If failed, find the rough wavelength the global wavelength solution.
    Finally, pick up the closet wavelength from the wavelength standards.

    Args:
        x (float): Pixel coordinate.
        aperture (integer): Aperture number.
        identlist (dict): Dict of identified lines for different apertures.
        linelist (list): List of wavelength standards.
        param (dict): Parameters of the :class:`CalibWindow`.
    Returns:
        float: Guessed wavelength. If failed, return *None*.
        
    '''
    rough_wv = None

    # guess wv from the identified lines in this order
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size >= 2:
            fit_order = min(list1.size-1, 2)
            local_coeff = np.polyfit(list1['pixel'], list1['wavelength'], deg=fit_order)
            rough_wv = np.polyval(local_coeff, x)

    # guess wavelength from global wavelength solution
    if rough_wv is None and param['coeff'].size > 0:
        npixel = param['npixel']
        order = aperture*param['k'] + param['offset']
        rough_wv = get_wv_val(param['coeff'], param['npixel'], x, order)

    if rough_wv is None:
        return None
    else:
        # now find the nearest wavelength in linelist
        wave_list = np.array([line[0] for line in linelist])
        iguess = np.abs(wave_list-rough_wv).argmin()
        guess_wv = wave_list[iguess]
        return guess_wv

def is_identified(wavelength, identlist, aperture):
    '''Check if the input wavelength has already been identified.

    Args:
        wavelength (float): Wavelength of the input line.
        identlist (dict): Dict of identified lines.
        aperture (integer): Aperture number.
    Returns:
        bool: *True* if **wavelength** and **aperture** in **identlist**.
    
    '''
    if aperture in identlist:
        list1 = identlist[aperture]
        diff = np.abs(list1['wavelength'] - wavelength)
        if diff.min()<1e-3:
            return True
        else:
            return False
    else:
        return False

def find_order(identlist, npixel):
    '''Find the linear relation between the aperture numbers and diffraction
    orders.
    The relationship is `order = k*aperture + offset`.
    Longer wavelength has lower order number.

    Args:
        identlist (dict): Dict of identified lines.
        npixel (integer): Number of pixels along the main dispersion direction.
    Returns:
        tuple: A tuple containg (`k`, `offset`), where

            * **k** (*integer*): Coefficient in the relationship
              `order = k*aperture + offset`.
            * **offset** (*integer*): Coefficient in the relationship
              `order = k*aperture + offset`.

    '''
    aper_lst, wvc_lst = [], []
    for aperture, list1 in sorted(identlist.items()):
        if list1.size<3:
            continue
        less_half, more_half = False, False
        for pix, wav in zip(list1['pixel'], list1['wavelength']):
            if pix < npixel/2.:
                less_half = True
            elif pix >= npixel/2.:
                more_half = True
        if less_half and more_half:
            if list1['pixel'].size>2:
                deg = 2
            else:
                deg = 1
            c = np.polyfit(list1['pixel'], list1['wavelength'], deg=deg)
            wvc = np.polyval(c, npixel/2.)
            aper_lst.append(aperture)
            wvc_lst.append(wvc)
    aper_lst = np.array(aper_lst)
    wvc_lst  = np.array(wvc_lst)
    if wvc_lst[0] > wvc_lst[-1]:
        k = 1
    else:
        k = -1

    offset_lst = np.arange(-500, 500)
    eva_lst = []
    for offset in offset_lst:
        const = (k*aper_lst + offset)*wvc_lst
        diffconst = np.diff(const)
        eva = (diffconst**2).sum()
        eva_lst.append(eva)
    eva_lst = np.array(eva_lst)
    offset = offset_lst[eva_lst.argmin()]

    return k, offset

def save_ident(identlist, coeff, filename, channel):
    '''Write the ident line list and coefficients into an ASCII file.
    The existing informations in the ASCII file will not be affected.
    Only the input channel will be overwritten.

    Args:
        identlist (dict): Dict of identified lines.
        coeff (:class:`numpy.array`): Coefficient array.
        result (dict): A dict containing identification results.
        filename (string): Name of the ASCII file.
        channel (string): Name of channel.
    Returns:
        No returns.

    See also:
        :func:`load_ident`

    '''
    exist_row_lst = []
    if os.path.exists(filename):
        # if filename already exist, only overwrite the current channel
        infile = open(filename)
        for row in infile:
            row = row.strip()
            if len(row)==0 or row[0] in '#$%^@!':
                continue
            g = row.split()
            if g[0] != channel:
                exist_row_lst.append(row)
        infile.close()

    outfile = open(filename, 'w')

    # write other channels
    if len(exist_row_lst)>0:
        outfile.write(os.linesep.join(exist_row_lst))

    # write current channel

    # write identified lines
    for aperture, list1 in sorted(identlist.items()):
        for pix, wav, mask, res, method in zip(list1['pixel'],
                list1['wavelength'], list1['mask'], list1['residual'],
                list1['method']):
            outfile.write('%1s LINE %03d %10.4f %10.4f %1d %+10.6f %1s'%(
                channel, aperture, pix, wav, int(mask), res, method.decode('ascii'))+os.linesep)

    # write coefficients
    for irow in range(coeff.shape[0]):
        string = ' '.join(['%18.10e'%v for v in coeff[irow]])
        outfile.write('%1s COEFF %s'%(channel, string)+os.linesep)

    outfile.close()


def load_ident(filename, channel):
    '''Load identified line list from an ASCII file.

    Args:
        filename (string): Name of the identification file.
        channel (string): Name of channel.
    Returns:
        tuple: A tuple (`identlist`, `coeff`), where

            * **identlist** (*dict*):  Identified lines for all orders
            * **coeff** (:class:`numpy.array`): Coefficients of wavelengths

    See also:
        :func:`save_ident`
    
    '''
    identlist = {}
    coeff = []

    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#$%^@!':
            continue
        g = row.split()
        if g[0] != channel:
            continue
        key = g[1]
        if key == 'LINE':
            aperture    = int(g[2])
            pixel       = float(g[3])
            wavelength  = float(g[4])
            mask        = bool(g[5])
            residual    = float(g[6])
            method      = g[7].strip()
            item = np.array((channel,aperture,0,pixel,wavelength,0.,mask,
                             residual,method),dtype=identlinetype)

            if aperture not in identlist:
                identlist[aperture] = []
            identlist[aperture].append(item)
        elif key == 'COEFF':
            coeff.append([float(v) for v in g[2:]])

    infile.close()

    # convert list of every order to numpy structured array
    for aperture, list1 in identlist.items():
        identlist[aperture] = np.array(list1, dtype=identlinetype)

    # convert coeff to numpy array
    coeff = np.array(coeff)

    return identlist, coeff

def gaussian(A,center,fwhm,x):
    sigma = fwhm/2.35482
    return A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc(p,x,y):
    return y - gaussian(p[0],p[1],p[2],x)

def gaussian_bkg(A,center,fwhm,bkg,x):
    sigma = fwhm/2.35482
    return bkg + A*np.exp(-(x-center)**2/2./sigma**2)
def errfunc2(p,x,y):
    return y - gaussian_bkg(p[0],p[1],p[2],p[3],x)

def find_local_peak(flux, x, width):
    '''Find the central pixel of an emission line.

    Args:
        flux (:class:`numpy.array`): Flux array.
        x (integer): The approximate coordinate of the peak pixel.
        width (integer): Window of profile fitting.
    Returns:
        tuple: A tuple containing (`i1`, `i2`, `p1`, `std`), where

            * **i1** (*integer*): Index of the left side.
            * **i2** (*integer*): Index of the right side.
            * **p1** (*list*): List of fitting parameters.
            * **std** (*float*): Standard devation of the fitting.
        
    '''
    width = int(round(width))
    if width%2 != 1:
        width += 1
    half = int((width-1)/2)

    i = int(round(x))

    # find the peak in a narrow range

    i1, i2 = max(0, i-half), min(flux.size, i+half+1)
    # find the peak position
    imax = flux[i1:i2].argmax() + i1
    xdata = np.arange(i1,i2)
    ydata = flux[i1:i2]
    # determine the initial parameters for gaussian fitting + background
    p0 = [ydata.max()-ydata.min(), imax, 3., ydata.min()]
    # least square fitting
    p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))

    res_lst = errfunc2(p1, xdata, ydata)
    std = math.sqrt((res_lst**2).sum()/(res_lst.size-len(p0)-1))

    return i1, i2, p1, std

def recenter(flux, center):
    '''Relocate the profile center of the lines.
    
    Args:
        flux (:class:`numpy.array`): Flux array.
        center (float): Center of the line.
    Returns:
        float: The new center of the line profile.
    '''
    y1, y2 = int(center)-3, int(center)+4
    ydata = flux[y1:y2]
    xdata = np.arange(y1,y2)
    p0 = [ydata.min(), ydata.max()-ydata.min(), ydata.argmax()+y1, 2.5]
    p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))
    return p1[2]
    
def search_linelist(linelistname):
    '''Search the line list file and load the list.

    Args:
        linelistname (string): Name of the line list file
    Returns:
        string: Path to the line list file
    '''

    # first, seach $LINELIST in current working directory
    if os.path.exists(linelistname):
        return linelistname

    # seach $LINELIST.dat in current working directory
    newname = linelistname+'.dat'
    if os.path.exists(newname):
        return newname

    # seach $LINELIST in data path of edrs
    data_path = os.path.join(os.path.dirname(__file__),
                '../data/linelist/')
    newname = os.path.join(data_path, linelistname)
    if os.path.exists(newname):
        return newname

    # seach $LINELIST.dat in data path of edrs
    newname = os.path.join(data_path, linelistname+'.dat')
    if os.path.exists(newname):
        return newname

    # seach EDRS_DATA path
    edrs_data = os.getenv('EDRS_DATA')
    if len(edrs_data)>0:
        data_path = os.path.join(edrs_data, 'linelist')
        newname = os.path.join(data_path, linelistname+'.dat')
        if os.path.exists(newname):
            return newname

    return None

def load_linelist(filename):
    '''Load standard wavelength line list from a given file.

    Args:
        filename (string): Name of the wavelength standard list file.
    Returns:
        list: A list containing (wavelength, species).
    '''
    linelist = []
    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#%!@':
            continue
        g = row.split()
        wv = float(g[0])
        if len(g)>1:
            species = g[1]
        else:
            species = ''
        linelist.append((wv, species))
    infile.close()
    return linelist

def find_shift_ccf(f1, f2, shift0=0.0):
    '''Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.array`): Flux array.
        f2 (:class:`numpy.array`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.
    Returns:
        float: Relative shift between the two flux arrays.
    '''
    x = np.arange(f1.size)
    interf = intp.InterpolatedUnivariateSpline(x, f1, k=3)
    func = lambda shift: -(interf(x - shift)*f2).sum(dtype=np.float64)
    res = opt.minimize(func, shift0, method='Powell')
    return res['x']

def find_shift_ccf2(f1, f2, shift0=0.0):
    '''Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.array`): Flux array.
        f2 (:class:`numpy.array`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.
    Returns:
        float: Relative shift between the two flux arrays.
    '''
    n = f1.size
    def aaa(shift):
        shift = int(np.round(shift))
        s1 = f1[max(0,shift):min(n,n+shift)]
        s2 = f2[max(0,-shift):min(n,n-shift)]
        c1 = math.sqrt((s1**2).sum())
        c2 = math.sqrt((s2**2).sum())
        return -np.correlate(s1, s2)/c1/c2
    res = opt.minimize(aaa, shift0, method='Powell')
    return res['x']


def find_drift(spec1, spec2, offset=0):
    '''Find the drift between two spectra. The apertures of the two spectra must
    be aligned.

    Args:
        spec1 (:class:`numpy.dtype`): Spectra array.
        spec2 (:class:`numpy.dtype`): Spectra array.
        offset (float): Approximate relative shift between the two spectra
            arrays.
    Returns:
        float: Calculated relative shift between the two spectra arrays.
    '''

    shift_lst = []
    for item1 in spec1:
        aperture1 = item1['aperture']
        if aperture1 + offset in spec2['aperture']:
            i = list(spec2['aperture']).index(aperture1 + offset)
            item2 = spec2[i]
            flux1 = item1['flux']
            flux2 = item2['flux']

            #shift = find_shift_ccf(flux1, flux2)
            #shift = find_shift_ccf_pixel(flux1, flux2, 100)
            shift = find_shift_ccf(flux1, flux2)

            shift_lst.append(shift)

    drift = np.median(np.array(shift_lst))
    return drift

class CalibFigure(Figure):
    '''Figure class for wavelength calibration.

    Args:
        width (integer): Width of figure.
        height (integer): Height of figure.
        dpi (integer): DPI of figure.
        filename (string): Filename of input spectra.
        channel (string): Channel name of input spectra.
    '''

    def __init__(self, width, height, dpi, filename, channel):
        '''Constuctor of :class:`CalibFigure`.
        '''
        super(CalibFigure, self).__init__(figsize=(width/dpi, height/dpi), dpi=dpi)
        self.patch.set_facecolor('#d9d9d9')

        # add axes
        self._ax1 = self.add_axes([0.06,0.07,0.52,0.87])
        self._ax2 = self.add_axes([0.65,0.07,0.32,0.4])
        self._ax3 = self.add_axes([0.65,0.54,0.32,0.4])

        # add title
        title = '%s - channel %s'%(filename, channel)
        self.suptitle(title, fontsize=15)

        #draw the aperture number to the corner of ax1
        bbox = self._ax1.get_position()
        self._ax1._aperture_text = self.text(bbox.x0 + 0.05, bbox.y1-0.1,
                                  '', fontsize=15)

        # draw residual and number of identified lines in ax2
        bbox = self._ax3.get_position()
        self._ax3._residual_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=12)

    def plot_solution(self, identlist, aperture_lst, plot_ax1=False,  **kwargs):
        '''Plot the wavelength solution.

        Args:
            identlist (dict): Dict of identified lines.
            aperture_lst (list): List of apertures to be plotted.
            plot_ax1 (bool): Whether to plot the first axes.
            coeff (:class:`numpy.array`): Coefficient array.
            k (integer): `k` value in the relationship `order = k*aperture +
                offset`.
            offset (integer): `offset` value in the relationship `order =
                k*aperture + offset`.
            npixel (integer): Number of pixels along the main dispersion
                direction.
            std (float): Standard deviation of wavelength fitting.
            nuse (integer): Number of lines actually used in the wavelength
                fitting.
            ntot (integer): Number of lines identified.
        '''
        coeff  = kwargs.pop('coeff')
        k      = kwargs.pop('k')
        offset = kwargs.pop('offset')
        npixel = kwargs.pop('npixel')
        std    = kwargs.pop('std')
        nuse   = kwargs.pop('nuse')
        ntot   = kwargs.pop('ntot')
        
        colors = 'rgbcmyk'

        self._ax2.cla()
        self._ax3.cla()

        if plot_ax1:
            self._ax1.cla()
            x = np.linspace(0, npixel-1, 100, dtype=np.float64)
            wv_min, wv_max = 1e9,0

        for aperture in aperture_lst:
            color = colors[aperture%7]
            order = k*aperture + offset

            if plot_ax1:
                w = get_wv_val(coeff, npixel, x, np.repeat(order, x.size))
                wv_max = max(wv_max, w.max())
                wv_min = min(wv_min, w.min())
                self._ax1.plot(x, w, color=color, ls='-', alpha=0.7)

            if aperture in identlist:
                list1 = identlist[aperture]
                pix_lst = list1['pixel']
                wav_lst = list1['wavelength']
                mask    = list1['mask'].astype(bool)
                res_lst = list1['residual']

                if plot_ax1:
                    self._ax1.scatter(pix_lst[mask],  wav_lst[mask],
                                      c=color, s=20, lw=0, alpha=0.7)
                    self._ax1.scatter(pix_lst[~mask], wav_lst[~mask],
                                      c='w', s=16, lw=0.7, alpha=0.7, edgecolor=color)

                repeat_aper_lst = np.repeat(aperture, pix_lst.size)
                self._ax2.scatter(repeat_aper_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.7)
                self._ax2.scatter(repeat_aper_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.7, edgecolor=color)
                self._ax3.scatter(pix_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.7)
                self._ax3.scatter(pix_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.7, edgecolor=color)

        self._ax3._residual_text.set_text('R.M.S. = %.5f, N = %d/%d'%(std, nuse, ntot))

        # adjust layout for ax1
        if plot_ax1:
            self._ax1.set_xlim(0, npixel-1)
            self._ax1.set_ylim(wv_min, wv_max)
            self._ax1.set_xlabel('Pixel')
            self._ax1.set_ylabel(u'$\lambda$ (\xc5)')
            self._ax1._aperture_text.set_text('')

        # adjust axis layout for ax2 (residual on aperture space)
        self._ax2.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax2.axhline(y=i*std, color='k', ls=':', lw=0.5)
        x1, x2 = self._ax2.get_xlim()
        x1 = max(x1,aperture_lst.min())
        x2 = min(x2,aperture_lst.max())
        self._ax2.set_xlim(x1, x2)
        self._ax2.set_ylim(-6*std, 6*std)
        self._ax2.set_xlabel('Aperture')
        self._ax2.set_ylabel(u'Residual on $\lambda$ (\xc5)')

        ## adjust axis layout for ax3 (residual on pixel space)
        self._ax3.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax3.axhline(y=i*std, color='k', ls=':', lw=0.5)
        self._ax3.set_xlim(0, npixel-1)
        self._ax3.set_ylim(-6*std, 6*std)
        self._ax3.set_xlabel('Pixel')
        self._ax3.set_ylabel(u'Residual on $\lambda$ (\xc5)')

def recalib(filename, identfilename, figfilename, ref_spec, linelist, channel,
    coeff, npixel, k, offset, window_size=13, xorder=3, yorder=3, maxiter=10,
    clipping=3, snr_threshold=10):
    '''
    Re-calibrate the wavelength of an input spectra file using another spectra
    as reference.

    Args:
        filename (string): Filename of the 1-D spectra.
        identfilename (string): Filename of wavelength identification.
        figfilename (string): Filename of the output wavelength figure.
        ref_spec (:class:`numpy.dtype`): Reference spectra.
        linelist (string): Name of wavelength standard file.
        channel (string): Name of the input channel.
        coeff (:class:`numpy.array`): Coefficients of the reference wavelength.
        npixel (integer): Number of pixels along the main-dispersion direction.
        k (integer): -1 or 1, depending on the relationship `order = k*aperture
            + offset`.
        offset (integer): coefficient in the relationship `order = k*aperture +
            offset`.
        window_size (integer): Size of the window in pixel to search for the
            lines.
        xorder (integer): Order of polynomial along X axis.
        yorder (integer): Order of polynomial along Y axis.
        maxiter (integer): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        snr_threshold (float): Minimum S/N of the spectral lines to be accepted
            in the wavelength fitting.
    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.array`): Coefficient array.
            * **npixel** (*integer*): Number of pixels along the main dispersion
              direction.
            * **k** (*integer*): Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*integer*): Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*): Standard deviation of wavelength fitting in Å.
            * **nuse** (*integer*): Number of lines used in the wavelength
              fitting.
            * **ntot** (*integer*): Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*): Dict of identified lines.
            * **window_size** (*integer*): Length of window in searching the
              line centers.
            * **xorder** (*integer*): Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*integer*): Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*integer*): Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*): Clipping value of the wavelength fitting.
            * **snr_threshold** (*float*): Minimum S/N of the spectral lines to
              be accepted in the wavelength fitting.
    See also:
        :func:`wvcalib`
        
    '''

    spec = fits.getdata(filename)
    if channel != '':
        mask = spec['channel']==channel
        spec = spec[mask]

    # find initial shift with cross-corelation functions
    shift = find_drift(ref_spec, spec)

    string = '%s channel %s shift = %+8.6f pixel'
    print(string%(os.path.basename(filename), channel, shift))

    # initialize the identlist
    identlist = {}

    # load the wavelengths
    linefilename = search_linelist(linelist)
    if linefilename is None:
        print('Error: Cannot find linelist file: %s'%linelist)
        exit()
    line_list = load_linelist(linefilename)

    x = np.arange(npixel)

    for row in spec:
        aperture = row['aperture']
        flux     = row['flux']
        order = k*aperture + offset
        wvs = get_wv_val(coeff, npixel, x-shift, np.repeat(order, npixel))
        wv1 = min(wvs[0], wvs[-1])
        wv2 = max(wvs[0], wvs[-1])
        has_insert = False
        for line in line_list:
            if line[0]<wv1:
                continue
            elif line[0]>wv2:
                break
            else:
                # wavelength in the range of this order
                diff = np.abs(wvs - line[0])
                i = diff.argmin()
                i1, i2, param, std = find_local_peak(flux, i, window_size)
                peak_x = param[1]

                if param[0] < 0.:
                    continue
                if param[1] < i1 or param[1] > i2:
                    continue
                if param[2] < 1.0:
                    continue
                if param[2] > 50. or param[2] < 1.0:
                    continue
                if param[3] < -0.5*param[0]:
                    continue
                snr = param[0]/std
                if snr < snr_threshold:
                    continue

                if aperture not in identlist:
                    identlist[aperture] = np.array([], dtype=identlinetype)

                item = np.array((channel, aperture, order, peak_x, line[0], snr, True, 0.0, 'a'),
                                dtype=identlinetype)
                identlist[aperture] = np.append(identlist[aperture], item)
                has_insert = True
        if has_insert:
            identlist[aperture] = np.sort(identlist[aperture], order='pixel')
        
    #for aperture, list1 in identlist.items():
    #    for row in list1:
    #        print(aperture, row['pixel'], row['wavelength'], row['snr'])

    coeff, std, k, offset, nuse, ntot = fit_wv(
        identlist = identlist, 
        npixel    = npixel,
        xorder    = xorder,
        yorder    = yorder,
        maxiter   = maxiter,
        clipping  = clipping,
        )

    fig_width  = 2500
    fig_height = 1500
    fig_dpi    = 150

    fig = CalibFigure(width    = fig_width,
                      height   = fig_height,
                      dpi      = fig_dpi,
                      filename = os.path.basename(filename),
                      channel  = channel,
                      )
    canvas = FigureCanvasAgg(fig)

    fig.plot_solution(identlist,
                      aperture_lst = spec['aperture'],
                      plot_ax1     = True,
                      coeff        = coeff,
                      k            = k,
                      offset       = offset,
                      npixel       = npixel,
                      std          = std,
                      nuse         = nuse,
                      ntot         = ntot,
                      )
    fig.savefig(figfilename)
    plt.close(fig)

    # organize results
    result = {
              'coeff':         coeff,
              'npixel':        npixel,
              'k':             k,
              'offset':        offset,
              'std':           std,
              'nuse':          nuse,
              'ntot':          ntot,
              'identlist':     identlist,
              'window_size':   window_size,
              'xorder':        xorder,
              'yorder':        yorder,
              'maxiter':       maxiter,
              'clipping':      clipping,
              'snr_threshold': snr_threshold,
            }

    return result

def reference_wv(infilename, outfilename, regfilename, frameid, calib_lst):
    '''
    Reference the wavelength and write the wavelength solution to the FITS file.

    Args:
        infilename (string): Filename of input spectra.
        outfilename (string): Filename of output spectra.
        regfilename (string): Filename of output region file for SAO-DS9.
        frameid (integer): FrameID of the input spectra. The frameid is used to
            find the proper calibration solution in calib_lst.
        calib_lst (dict): A dict with key of frameids, and values of calibration
            solutions for different channels.
    Returns:
        No returns.
    See also:
        :func:`wvcalib`

    '''
    data, head = fits.getdata(infilename, header=True)

    npoints = data['points'].max()

    newdescr = [descr for descr in data.dtype.descr]
    # add new columns
    newdescr.append(('order',np.int16))
    newdescr.append(('wavelength','>f8',(npoints,)))

    newspec = []

    # prepare for self reference. means one channel is ThAr
    file_identlist = []

    # find unique channels in the input spectra
    channel_lst = np.unique(data['channel'])

    # open region file and write headers
    regfile = open(regfilename, 'w')
    regfile.write('# Region file format: DS9 version 4.1'+os.linesep)
    regfile.write('global dashlist=8 3 width=1 font="helvetica 10 normal roman" ')
    regfile.write('select=1 highlite=1 dash=0 fixed=1 edit=0 move=0 delete=0 include=1 source=1'+os.linesep)

    # find aperture locations
    aperture_coeffs = get_aperture_coeffs_in_header(head)

    # loop all channels
    for channel in sorted(channel_lst):

        # filter the spectra in current channel
        mask = (data['channel'] == channel)
        if mask.sum() == 0:
            continue
        spec = data[mask]

        # check if the current frameid & channel are in calib_lst
        if frameid in calib_lst and channel in calib_lst[frameid]:
            self_reference = True
            calib = calib_lst[frameid][channel]
        else:
            self_reference = False
            # find the closet ThAr
            refcalib_lst = []
            for direction in [-1, +1]:
                _frameid = frameid
                while(True):
                    _frameid += direction
                    if _frameid in calib_lst and channel in calib_lst[_frameid]:
                        calib = calib_lst[_frameid][channel]
                        refcalib_lst.append(calib)
                        #print(item.frameid, 'append',channel, frameid)
                        break
                    elif _frameid <= min(calib_lst) or _frameid >= max(calib_lst):
                        break
                    else:
                        continue

        # get variable shortcuts.
        # in principle, these parameters in refcalib_lst should have the same
        # values. so just use the last calib solution
        k      = calib['k']
        offset = calib['offset']
        xorder = calib['xorder']
        yorder = calib['yorder']

        if self_reference:
            coeff = calib['coeff']
        else:
            # calculate the average coefficients
            coeff_lst = np.array([_calib['coeff'] for _calib in refcalib_lst])
            coeff = coeff_lst.mean(axis=0, dtype=np.float64)

        # write important parameters into the FITS header
        leading_str = 'HIERARCH EDRS WVCALIB CHANNEL %s'%channel
        head[leading_str+' K']      = k
        head[leading_str+' OFFSET'] = offset
        head[leading_str+' XORDER'] = xorder
        head[leading_str+' YORDER'] = yorder

        # write the coefficients
        for j, i in itertools.product(range(yorder+1), range(xorder+1)):
            head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

        # if the input spectra is a wavelength standard frame (e.g. ThAr), write
        # calibration solutions into FITS header
        if self_reference:
            head[leading_str+' MAXITER']    = calib['maxiter']
            head[leading_str+' STDDEV']     = calib['std']
            head[leading_str+' WINDOWSIZE'] = calib['window_size']
            head[leading_str+' NTOT']       = calib['ntot']
            head[leading_str+' NUSE']       = calib['nuse']
            head[leading_str+' NPIXEL']     = calib['npixel']

            # pack the identfied line list
            for aperture, list1 in calib['identlist'].items():
                for row in list1:
                    file_identlist.append(row)

        for row in spec:
            aperture = row['aperture']
            npixel   = row['points']
            order = aperture*k + offset
            wv = get_wv_val(coeff, npixel, np.arange(npixel), np.repeat(order, npixel))
            
            # add wavelength into FITS table
            item = list(row)
            item.append(order)
            item.append(wv)
            newspec.append(tuple(item))

            # write wavlength information into regfile
            if (channel, aperture) in aperture_coeffs:
                coeffs = aperture_coeffs[(channel, aperture)]
                position = poly.Chebyshev(coef=coeffs, domain=[0, npixel-1])
                color = {'A': 'red', 'B': 'green'}[channel]

                # write text in the left edge
                x = -6
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={A%d, O%d} color=%s'
                text = string%(x+1, y+1, aperture, order, color)
                regfile.write(text+os.linesep)

                # write text in the right edge
                x = npixel-1+6
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={A%d, O%d} color=%s'
                text = string%(x+1, y+1, aperture, order, color)
                regfile.write(text+os.linesep)

                # write text in the center
                x = npixel/2.
                y = position(x)
                string = '# text(%7.2f, %7.2f) text={Channel %s, Aperture %3d, Order %3d} color=%s'
                text = string%(x+1, y+1+5, channel, aperture, order, color)
                regfile.write(text+os.linesep)

                # draw lines
                x = np.linspace(0, npixel-1, 50)
                y = position(x)
                for (x1,x2), (y1, y2) in zip(pairwise(x), pairwise(y)):
                    string = 'line(%7.2f,%7.2f,%7.2f,%7.2f) # color=%s'
                    text = string%(x1+1, y1+1, x2+1, y2+1, color)
                    regfile.write(text+os.linesep)

                # draw ticks at integer wavelegths
                pix = np.arange(npixel)
                if wv[0] > wv[-1]:
                    wv = wv[::-1]
                    pix= pix[::-1]
                f = intp.InterpolatedUnivariateSpline(wv, pix, k=3)
                w1 = wv.min()
                w2 = wv.max()
                for w in np.arange(int(math.ceil(w1)), int(math.floor(w2))+1):
                    x = f(w)
                    y = position(x)
                    if w%10==0:
                        ticklen = 3
                        string = '# text(%7.2f, %7.2f) text={%4d} color=%s'
                        text = string%(x+1+20, y+1+5, w, color)
                        regfile.write(text+os.linesep)
                    else:
                        ticklen = 1
                    string = 'line(%7.2f, %7.2f, %7.2f, %7.2f) # color=%s wl=%d'
                    text = string%(x+1+20, y+1, x+1+20, y+1+ticklen, color, w)
                    regfile.write(text+os.linesep)

                # draw identified lines in region file
                if self_reference and aperture in calib['identlist']:
                    list1 = calib['identlist'][aperture]
                    for row in list1:
                        x = row['pixel']
                        y = position(x)
                        ps = ('x', 'circle')[row['mask']]
                        string = 'point(%7.2f, %7.2f) # point=%s color=%s wl=%9.4f'
                        text = string%(x+1, y+1, ps, color, row['wavelength'])
                        regfile.write(text+os.linesep)

    newspec = np.array(newspec, dtype=newdescr)

    regfile.close()

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(newspec)
    lst = [pri_hdu, tbl_hdu1]

    if len(file_identlist)>0:
        file_identlist = np.array(file_identlist, dtype=identlinetype)
        tbl_hdu2 = fits.BinTableHDU(file_identlist)
        lst.append(tbl_hdu2)
    hdu_lst  = fits.HDUList(lst)

    if os.path.exists(outfilename):
        os.remove(outfilename)
    hdu_lst.writeto(outfilename)

def get_aperture_coeffs_in_header(head):
    '''Get coefficients of each aperture from the FITS header.

    Args:
        head (:class:`astropy.io.fits.Header`): Header of FITS file.
    Returns:
        dict: A dict containing coefficients for each aperture and each channel.
    '''
    
    coeffs = {}
    for key, value in head.items():
        exp = '^EDRS TRACE CHANNEL [A-Z] APERTURE \d+ COEFF \d+$'
        if re.match(exp, key) is not None:
            g = key.split()
            channel  = g[3]
            aperture = int(g[5])
            icoeff   = int(g[7])
            if (channel, aperture) not in coeffs:
                coeffs[(channel, aperture)] = []
            if len(coeffs[(channel, aperture)]) == icoeff:
                coeffs[(channel, aperture)].append(value)
    return coeffs

