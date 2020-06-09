import os
import re
import sys
import math
import datetime
import dateutil.parser
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
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backends.backend_agg import FigureCanvasAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

from ..utils.regression2d import polyfit2d, polyval2d
from ..utils.onedarray    import pairwise
from .trace import load_aperture_set_from_header

# Data format for identified line table
identlinetype = np.dtype({
    'names':  ['aperture','order','pixel','wavelength','q','mask',
               'residual','method'],
    'formats':[np.int16, np.int16, np.float32, np.float64, np.float32,
               np.int16, np.float64, 'S1'],
    })

class CustomToolbar(NavigationToolbar2Tk):
    """Class for customized matplotlib toolbar.

    Args:
        canvas (:class:`matplotlib.backends.backend_agg.FigureCanvasAgg`):
            Canvas object used in :class:`CalibWindow`.
        master (Tkinter widget): Parent widget.
    """

    def __init__(self, canvas, master):
        """Constructor of :class:`CustomToolbar`.
        """
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move','pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect','zoom'),
            ('Subplots', 'Configure subplots', 'subplots','configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        NavigationToolbar2Tk.__init__(self, canvas, master)

    def set_message(self, msg):
        """Remove the coordinate displayed in the toolbar.
        """
        pass

class PlotFrame(tk.Frame):
    """The frame for plotting spectrum in the :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of frame.
        height (int): Height of frame.
        dpi (int): DPI of figure.
        identlist (dict): Dict of identified lines.
        linelist (list): List of wavelength standards.
    """
    def __init__(self, master, width, height, dpi, identlist, linelist):
        """Constructor of :class:`PlotFrame`.
        """

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fig = CalibFigure(width  = width,
                               height = height,
                               dpi    = dpi,
                               title  = master.param['title'],
                               )
        self.ax1 = self.fig._ax1
        self.ax2 = self.fig._ax2
        self.ax3 = self.fig._ax3

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
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
    """The frame for buttons and tables on the right side of the
    :class:`CalibWindow`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of the frame.
        height (int): Height of the frame.
        linelist (list): List of wavelength standards.
        identlist (dict): Dict of identified lines.
    """

    def __init__(self, master, width, height, linelist, identlist):
        """Constuctor of :class:`InfoFrame`.
        """

        self.master = master

        title = master.param['title']

        tk.Frame.__init__(self, master, width=width, height=height)

        self.fname_label = tk.Label(master = self,
                                    width  = width,
                                    font   = ('Arial', 14),
                                    text   = title,
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
        """Update the navigation buttons.
        """
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
        """Update the order information to be displayed on the top.
        """
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
    """A table for the input spectral lines embedded in the :class:`InfoFrame`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of line table.
        height (int): Height of line table.
        identlist (dict): Dict of identified lines.
        linelist (list): List of wavelength standards.
    """
    def __init__(self, master, width, height, identlist, linelist):
        """Constructor of :class:`LineTable`.
        """
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
                                      columns    = ('wl', 'species', 'status'),
                                      show       = 'headings',
                                      style      = 'Treeview',
                                      height     = 22,
                                      selectmode ='browse')
        self.line_tree.bind('<Button-1>', self.on_click_item)

        self.scrollbar = tk.Scrollbar(master = self.data_frame,
                                      orient = tk.VERTICAL,
                                      width  = 20)

        self.line_tree.column('wl',      width=160)
        self.line_tree.column('species', width=140)
        self.line_tree.column('status',  width=width-160-140-20)
        self.line_tree.heading('wl',      text=u'\u03bb in air (\xc5)')
        self.line_tree.heading('species', text='Species')
        self.line_tree.heading('status',  text='Status')
        self.line_tree.config(yscrollcommand=self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=35)
        style.configure('Treeview.Heading', font=('Arial', 10))

        self.scrollbar.config(command=self.line_tree.yview)

        self.item_lst = []
        for line in linelist:
            wl, species = line
            iid = self.line_tree.insert('',tk.END,
                    values=(wl, species, ''), tags='normal')
            self.item_lst.append((iid,  wl))
        self.line_tree.tag_configure('normal', font=('Arial', 10))

        self.line_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.data_frame.pack(side=tk.TOP, fill=tk.Y)
        self.pack()

    def on_clear_search(self):
        """Clear the search bar.
        """
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
        """Event response function for clicking lines.
        """

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

            wl_diff = np.abs(list1['wavelength'] - float(values[0]))
            mindiff = wl_diff.min()
            argmin  = wl_diff.argmin()
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
    """Frame for the fitting parameters embedded in the :class:`InfoFrame`.

    Args:
        master (Tkinter widget): Parent widget.
        width (int): Width of frame.
        height (int): Height of frame.
    """
    def __init__(self, master, width, height):
        """Constructor of :class:`FitparaFrame`.
        """

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
        """Response function of changing order of polynomial along x-axis.
        """
        self.master.master.param['xorder'] = int(self.xorder_box.get())

    def on_change_yorder(self):
        """Response function of changing order of polynomial along y-axis.
        """
        self.master.master.param['yorder'] = int(self.yorder_box.get())

    def on_change_maxiter(self):
        """Response function of changing maximum number of iteration.
        """
        self.master.master.param['maxiter'] = int(self.maxiter_box.get())

    def on_change_clipping(self, value):
        """Response function of changing clipping value.
        """
        self.master.master.param['clipping'] = float(value)

class CalibWindow(tk.Frame):
    """Frame of the wavelength calibration window.

    Args:
        master (:class:`tk.TK`): Tkinter root window.
        width (int): Width of window.
        height (int): Height of window.
        dpi (int): DPI of figure.
        spec (:class:`numpy.dtype`): Spectra data.
        figfilename (str): Filename of the output wavelength calibration
            figure.
        title (str): A string to display as the title of calib figure.
        identlist (dict): Identification line list.
        linelist (list): List of wavelength standards (wavelength, species).
        window_size (int): Size of the window in pixel to search for line
            peaks.
        xorder (int): Degree of polynomial along X direction.
        yorder (int): Degree of polynomial along Y direction.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        snr_threshold (float): Minimum S/N of the spectral lines to be accepted
            in the wavelength fitting.
    """
    def __init__(self, master, width, height, dpi, spec, figfilename, title,
            identlist, linelist, window_size, xorder, yorder, maxiter, clipping,
            q_threshold, fit_filter):
        """Constructor of :class:`CalibWindow`.
        """

        self.master    = master
        self.spec      = spec
        self.identlist = identlist
        self.linelist  = linelist

        tk.Frame.__init__(self, master, width=width, height=height)

        self.param = {
            'mode':         'ident',
            'aperture':     self.spec['aperture'].min(),
            'figfilename':  figfilename,
            'title':        title,
            'aperture_min': self.spec['aperture'].min(),
            'aperture_max': self.spec['aperture'].max(),
            'npixel':       self.spec['points'].max(),
            # parameters of displaying
            'xlim':         {},
            'ylim':         {},
            'ident':        None,
            # parameters of converting aperture and order
            'k':            None,
            'offset':       None,
            # wavelength fitting parameters
            'window_size':  window_size,
            'xorder':       xorder,
            'yorder':       yorder,
            'maxiter':      maxiter,
            'clipping':     clipping,
            'q_threshold':  q_threshold,
            # wavelength fitting results
            'std':          0,
            'coeff':        np.array([]),
            'nuse':         0,
            'ntot':         0,
            'direction':    '',
            'fit_filter':   fit_filter,
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
        """Fit the wavelength and plot the solution in the figure.
        """

        coeff, std, k, offset, nuse, ntot = fit_wavelength(
                identlist   = self.identlist,
                npixel      = self.param['npixel'],
                xorder      = self.param['xorder'],
                yorder      = self.param['yorder'],
                maxiter     = self.param['maxiter'],
                clipping    = self.param['clipping'],
                fit_filter  = self.param['fit_filter'],
                )

        self.param['coeff']  = coeff
        self.param['std']    = std
        self.param['k']      = k
        self.param['offset'] = offset
        self.param['nuse']   = nuse
        self.param['ntot']   = ntot

        self.plot_wavelength()

        # udpdate the order/aperture string
        aperture = self.param['aperture']
        order = k*aperture + offset
        text = 'Order %d (Aperture %d)'%(order, aperture)
        self.info_frame.order_label.config(text=text)

        self.update_fit_buttons()

    def recenter(self):
        """Relocate the peaks for all the identified lines.
        """
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
        """Clear all the identified lines."""

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
        """Identify all lines in the wavelength standard list automatically.
        """
        k       = self.param['k']
        offset  = self.param['offset']
        coeff   = self.param['coeff']
        npixel  = self.param['npixel']
        for aperture in sorted(self.spec['aperture']):
            mask = self.spec['aperture'] == aperture
            flux = self.spec[mask][0]['flux']

            # scan every order and find the upper and lower limit of wavelength
            order = k*aperture + offset

            # generated the wavelengths for every pixel in this oirder
            x = np.arange(npixel)
            wl = get_wavelength(coeff, npixel, x, np.repeat(order, x.size))
            w1 = min(wl[0], wl[-1])
            w2 = max(wl[0], wl[-1])

            has_insert = False
            # now scan the linelist
            for line in self.linelist:
                if line[0] < w1:
                    continue
                if line[0] > w2:
                    break

                # wavelength in the range of this order
                # check if this line has already been identified
                if is_identified(line[0], self.identlist, aperture):
                    continue

                # now has not been identified. find peaks for this line
                diff = np.abs(wl - line[0])
                i = diff.argmin()
                i1, i2, param, std = find_local_peak(flux, i,
                                        self.param['window_size'])
                keep = auto_line_fitting_filter(param, i1, i2)
                if not keep:
                    continue

                q = param[0]/std
                if q < self.param['q_threshold']:
                    continue
                peak_x = param[1]

                '''
                fig = plt.figure(figsize=(6,4),tight_layout=True)
                ax = fig.gca()
                ax.plot(np.arange(i1,i2), flux[i1:i2], 'ro')
                newx = np.arange(i1, i2, 0.1)
                ax.plot(newx, gaussian_bkg(param[0], param[1], param[2],
                            param[3], newx), 'b-')
                ax.axvline(x=param[1], color='k',ls='--')
                y1,y2 = ax.get_ylim()
                ax.text(0.9*i1+0.1*i2, 0.1*y1+0.9*y2, 'A=%.1f'%param[0])
                ax.text(0.9*i1+0.1*i2, 0.2*y1+0.8*y2, 'BKG=%.1f'%param[3])
                ax.text(0.9*i1+0.1*i2, 0.3*y1+0.7*y2, 'FWHM=%.1f'%param[2])
                ax.text(0.9*i1+0.1*i2, 0.4*y1+0.6*y2, 'q=%.1f'%q)
                ax.set_xlim(i1,i2)
                fig.savefig('tmp/%d-%d-%d.png'%(aperture, i1, i2))
                plt.close(fig)
                '''

                # initialize line table
                if aperture not in self.identlist:
                    self.identlist[aperture] = np.array([], dtype=identlinetype)

                item = np.array((aperture, order, peak_x, line[0], q, True, 0.0,
                                'a'), dtype=identlinetype)

                self.identlist[aperture] = np.append(self.identlist[aperture], item)
                has_insert = True
                #print('insert', aperture, line[0], peak_x, i)

            # resort this order if there's new line inserted
            if has_insert:
                self.identlist[aperture] = np.sort(self.identlist[aperture],
                                                    order='pixel')

        self.fit()

    def switch(self):
        """Response funtion of switching between "ident" and "fit" mode.
        """

        if self.param['mode']=='ident':
            # switch to fit mode
            self.param['mode']='fit'

            self.plot_wavelength()

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
        """Response function of pressing the next aperture."""
        if self.param['aperture'] < self.spec['aperture'].max():
            self.param['aperture'] += 1
            self.plot_aperture()

    def prev_aperture(self):
        """Response function of pressing the previous aperture."""
        if self.param['aperture'] > self.spec['aperture'].min():
            self.param['aperture'] -= 1
            self.plot_aperture()

    def plot_aperture(self):
        """Plot a specific aperture in the figure.
        """
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

    def plot_wavelength(self):
        """A wrap for plotting the wavelength solution."""

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

        self.plot_frame.fig.plot_solution(self.identlist, aperture_lst,
                                            plot_ax1, **kwargs)

        self.plot_frame.canvas.draw()
        self.plot_frame.fig.savefig(self.param['figfilename'])

    def on_click(self, event):
        """Response function of clicking the axes.

        Double click means find the local peak and prepare to add a new
        identified line.
        """
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

                    wl = list1[imin]['wavelength']

                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        item = record[0]
                        wave = record[1]
                        if abs(wl - wave)<1e-3:
                            break

                    line_frame.line_tree.selection_set(item)
                    pos = i/float(len(line_frame.item_lst))
                    line_frame.line_tree.yview_moveto(pos)

                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(wl))

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
            guess_wl = guess_wavelength(peak_x, aperture, self.identlist,
                                        self.linelist, self.param)

            if guess_wl is None:
                # wavelength guess failed
                #line_frame.search_entry.focus()
                # update buttons
                line_frame.add_button.config(state=tk.NORMAL)
                line_frame.del_button.config(state=tk.DISABLED)
            else:
                # wavelength guess succeed

                # check whether wavelength has already been identified
                if is_identified(guess_wl, self.identlist, aperture):
                    # has been identified, do nothing
                    # update buttons
                    line_frame.add_button.config(state=tk.DISABLED)
                    line_frame.del_button.config(state=tk.NORMAL)
                else:
                    # has not been identified yet
                    # put the wavelength in the search bar
                    line_frame.search_text.set(str(guess_wl))
                    # select this line in the linetable
                    for i, record in enumerate(line_frame.item_lst):
                        iid  = record[0]
                        wave = record[1]
                        if abs(guess_wl - wave)<1e-3:
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
        """Response function of identifying a new line.
        """
        aperture = self.param['aperture']
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

        item = np.array((aperture, order, pixel, wavelength, -1., True, 0.0,
                        'm'), dtype=identlinetype)

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
        """Response function of deleting an identified line.
        """
        line_frame = self.info_frame.line_frame
        target_wl = float(line_frame.search_text.get())
        aperture = self.param['aperture']
        list1 = self.identlist[aperture]

        wl_diff = np.abs(list1['wavelength'] - target_wl)
        mindiff = wl_diff.min()
        argmin  = wl_diff.argmin()
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
        """Response function of drawing.
        """
        if self.param['mode'] == 'ident':
            ax1 = self.plot_frame.ax1
            aperture = self.param['aperture']
            self.param['xlim'][aperture] = ax1.get_xlim()
            self.param['ylim'][aperture] = ax1.get_ylim()

    def update_fit_buttons(self):
        """Update the status of fitting buttons.
        """
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
        """Update the status of batch buttons (recenter and clearall).
        """
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


def wlcalib(spec, figfilename, title, linelist, identfilename=None,
    window_size=13, xorder=3, yorder=3, maxiter=10, clipping=3,
    q_threshold=10, fit_filter=None
    ):
    """Identify the wavelengths of emission lines in the spectrum of a
    hollow-cathode lamp.

    Args:
        spec (:class:`numpy.dtype`): 1-D spectra.
        figfilename (str): Name of the output wavelength figure to be saved.
        title (str): A string to display as the title of calib figure.
        linelist (str): Name of wavelength standard file.
        identfilename (str): Name of an ASCII formatted wavelength identification
            file.
        window_size (int): Size of the window in pixel to search for the
            lines.
        xorder (int): Degree of polynomial along X direction.
        yorder (int): Degree of polynomial along Y direction.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        q_threshold (float): Minimum *Q*-factor of the spectral lines to be
            accepted in the wavelength fitting.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.ndarray`) – Coefficient array.
            * **npixel** (*int*) – Number of pixels along the main dispersion
              direction.
            * **k** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*) – Standard deviation of wavelength fitting in Å.
            * **nuse** (*int*) – Number of lines used in the wavelength
              fitting.
            * **ntot** (*int*) – Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*) – Dict of identified lines.
            * **window_size** (*int*) – Length of window in searching the
              line centers.
            * **xorder** (*int*) – Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*int*) – Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*int*) – Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*) – Clipping value of the wavelength fitting.
            * **q_threshold** (*float*) – Minimum *Q*-factor of the spectral
              lines to be accepted in the wavelength fitting.

    Notes:
        If **identfilename** is given and exist, load the identified wavelengths
        from this ASCII file, and display them in the calibration window. If not
        exist, save the identified list into **identfilename** with ASCII
        format.

    See also:
        :func:`recalib`
    """

    # initialize fitting list
    if identfilename is not None and os.path.exists(identfilename):
        identlist, _ = load_ident(identfilename)
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
                              width       = window_width,
                              height      = window_height-34,
                              dpi         = fig_dpi,
                              spec        = spec,
                              figfilename = figfilename,
                              title       = title,
                              identlist   = identlist,
                              linelist    = line_list,
                              window_size = window_size,
                              xorder      = xorder,
                              yorder      = yorder,
                              maxiter     = maxiter,
                              clipping    = clipping,
                              q_threshold = q_threshold,
                              fit_filter  = fit_filter,
                              )

    master.mainloop()

    coeff  = calibwindow.param['coeff']
    npixel = calibwindow.param['npixel']
    k      = calibwindow.param['k']
    offset = calibwindow.param['offset']

    # find the direction code
    aper = spec['aperture'][0]
    order = k*aper + offset
    wl = get_wavelength(coeff, npixel, np.arange(npixel), order)
    # refresh the direction code
    new_direction = 'x' + {1:'r', -1:'b'}[k] + '-+'[wl[0] < wl[-1]]

    # organize results
    result = {
              'coeff':       coeff,
              'npixel':      npixel,
              'k':           k,
              'offset':      offset,
              'std':         calibwindow.param['std'],
              'nuse':        calibwindow.param['nuse'],
              'ntot':        calibwindow.param['ntot'],
              'identlist':   calibwindow.identlist,
              'window_size': calibwindow.param['window_size'],
              'xorder':      calibwindow.param['xorder'],
              'yorder':      calibwindow.param['yorder'],
              'maxiter':     calibwindow.param['maxiter'],
              'clipping':    calibwindow.param['clipping'],
              'q_threshold': calibwindow.param['q_threshold'],
              'direction':   new_direction,
            }

    # save ident list
    if len(calibwindow.identlist)>0 and \
        identfilename is not None and not os.path.exists(identfilename):
        save_ident(calibwindow.identlist, calibwindow.param['coeff'],
                    identfilename)

    return result

def fit_wavelength(identlist, npixel, xorder, yorder, maxiter, clipping,
        fit_filter=None):
    """Fit the wavelength using 2-D polynomial.

    Args:
        identlist (dict): Dict of identification lines for different apertures.
        npixel (int): Number of pixels for each order.
        xorder (int): Order of polynomial along X direction.
        yorder (int): Order of polynomial along Y direction.
        maxiter (int): Maximim number of iterations in the polynomial
            fitting.
        clipping (float): Threshold of sigma-clipping.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        tuple: A tuple containing:

            * **coeff** (:class:`numpy.ndarray`) – Coefficients array.
            * **std** (*float*) – Standard deviation.
            * **k** (*int*) – *k* in the relationship between aperture
              numbers and diffraction orders: `order = k*aperture + offset`.
            * **offset** (*int*) – *offset* in the relationship between
              aperture numbers and diffraction orders: `order = k*aperture +
              offset`.
            * **nuse** (*int*) – Number of lines used in the fitting.
            * **ntot** (*int*) – Number of lines found.

    See also:
        :func:`get_wavelength`
    """
    # find physical order
    k, offset = find_order(identlist, npixel)

    # parse the fit_filter=None
    if fit_filter is None:
        fit_filter = lambda item: True

    # convert indent_line_lst into fitting inputs
    fit_p_lst = []  # normalized pixel
    fit_o_lst = []  # diffraction order
    fit_w_lst = []  # order*wavelength
    fit_m_lst = []  # initial mask
    # the following list is used to find the position (aperture, no)
    # of each line
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
            fit_m_lst.append(fit_filter(item))
            lineid_lst.append((aperture, iline))
    fit_p_lst = np.array(fit_p_lst)
    fit_o_lst = np.array(fit_o_lst)
    fit_w_lst = np.array(fit_w_lst)
    fit_m_lst = np.array(fit_m_lst)

    mask = fit_m_lst

    for nite in range(maxiter):
        coeff = polyfit2d(fit_p_lst[mask], fit_o_lst[mask], fit_w_lst[mask],
                          xorder=xorder, yorder=yorder)
        res_lst = fit_w_lst - polyval2d(fit_p_lst, fit_o_lst, coeff)
        res_lst = res_lst/fit_o_lst

        mean = res_lst[mask].mean(dtype=np.float64)
        std  = res_lst[mask].std(dtype=np.float64)
        m1 = res_lst > mean - clipping*std
        m2 = res_lst < mean + clipping*std
        new_mask = m1*m2*mask
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

def get_wavelength(coeff, npixel, pixel, order):
    """Get wavelength.

    Args:
        coeff (:class:`numpy.ndarray`): 2-D Coefficient array.
        npixel (int): Number of pixels along the main dispersion direction.
        pixel (*int* or :class:`numpy.ndarray`): Pixel coordinates.
        order (*int* or :class:`numpy.ndarray`): Diffraction order number.
            Must have the same length as **pixel**.

    Returns:
        float or :class:`numpy.ndarray`: Wavelength solution of the given pixels.

    See also:
        :func:`fit_wavelength`
    """
    # convert aperture to order
    norm_pixel = pixel*2./(npixel-1) - 1
    #norm_order  = 50./order
    #norm_order  = order/50.
    return polyval2d(norm_pixel, order, coeff)/order

def guess_wavelength(x, aperture, identlist, linelist, param):
    """Guess wavelength according to the identified lines.
    First, try to guess the wavelength from the identified lines in the same
    order (aperture) by fitting polynomials.
    If failed, find the rough wavelength the global wavelength solution.
    Finally, pick up the closet wavelength from the wavelength standards.

    Args:
        x (float): Pixel coordinate.
        aperture (int): Aperture number.
        identlist (dict): Dict of identified lines for different apertures.
        linelist (list): List of wavelength standards.
        param (dict): Parameters of the :class:`CalibWindow`.

    Returns:
        float: Guessed wavelength. If failed, return *None*.
    """
    rough_wl = None

    # guess wavelength from the identified lines in this order
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size >= 2:
            fit_order = min(list1.size-1, 2)
            local_coeff = np.polyfit(list1['pixel'], list1['wavelength'], deg=fit_order)
            rough_wl = np.polyval(local_coeff, x)

    # guess wavelength from global wavelength solution
    if rough_wl is None and param['coeff'].size > 0:
        npixel = param['npixel']
        order = aperture*param['k'] + param['offset']
        rough_wl = get_wavelength(param['coeff'], param['npixel'], x, order)

    if rough_wl is None:
        return None
    else:
        # now find the nearest wavelength in linelist
        wave_list = np.array([line[0] for line in linelist])
        iguess = np.abs(wave_list-rough_wl).argmin()
        guess_wl = wave_list[iguess]
        return guess_wl

def is_identified(wavelength, identlist, aperture):
    """Check if the input wavelength has already been identified.

    Args:
        wavelength (float): Wavelength of the input line.
        identlist (dict): Dict of identified lines.
        aperture (int): Aperture number.

    Returns:
        bool: *True* if **wavelength** and **aperture** in **identlist**.
    """
    if aperture in identlist:
        list1 = identlist[aperture]
        if list1.size==0:
            # has no line in this aperture
            return False
        diff = np.abs(list1['wavelength'] - wavelength)
        if diff.min()<1e-3:
            return True
        else:
            return False
    else:
        return False

def find_order(identlist, npixel):
    """Find the linear relation between the aperture numbers and diffraction
    orders.
    The relationship is `order = k*aperture + offset`.
    Longer wavelength has lower order number.

    Args:
        identlist (dict): Dict of identified lines.
        npixel (int): Number of pixels along the main dispersion direction.

    Returns:
        tuple: A tuple containing:

            * **k** (*int*) – Coefficient in the relationship
              `order = k*aperture + offset`.
            * **offset** (*int*) – Coefficient in the relationship
              `order = k*aperture + offset`.
    """
    aper_lst, wlc_lst = [], []
    for aperture, list1 in sorted(identlist.items()):
        if list1.size<3:
            continue
        less_half = (list1['pixel'] < npixel/2).sum()>0
        more_half = (list1['pixel'] > npixel/2).sum()>0
        #less_half, more_half = False, False
        #for pix, wav in zip(list1['pixel'], list1['wavelength']):
        #    if pix < npixel/2.:
        #        less_half = True
        #    elif pix >= npixel/2.:
        #        more_half = True
        if less_half and more_half:
            if list1['pixel'].size>2:
                deg = 2
            else:
                deg = 1
            c = np.polyfit(list1['pixel'], list1['wavelength'], deg=deg)
            wlc = np.polyval(c, npixel/2.)
            aper_lst.append(aperture)
            wlc_lst.append(wlc)
    aper_lst = np.array(aper_lst)
    wlc_lst  = np.array(wlc_lst)
    if wlc_lst[0] > wlc_lst[-1]:
        k = 1
    else:
        k = -1

    offset_lst = np.arange(-500, 500)
    eva_lst = []
    for offset in offset_lst:
        const = (k*aper_lst + offset)*wlc_lst
        diffconst = np.diff(const)
        eva = (diffconst**2).sum()
        eva_lst.append(eva)
    eva_lst = np.array(eva_lst)
    offset = offset_lst[eva_lst.argmin()]

    return k, offset

def save_ident(identlist, coeff, filename, channel):
    """Write the ident line list and coefficients into an ASCII file.
    The existing informations in the ASCII file will not be affected.
    Only the input channel will be overwritten.

    Args:
        identlist (dict): Dict of identified lines.
        coeff (:class:`numpy.ndarray`): Coefficient array.
        result (dict): A dict containing identification results.
        filename (str): Name of the ASCII file.
        channel (str): Name of channel.

    See also:
        :func:`load_ident`
    """
    if channel is None:
        outfile = open(filename, 'w')
    else:
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
            if channel is None:
                outfile.write('LINE %03d %10.4f %10.4f %1d %+10.6f %1s'%(
                    aperture, pix, wav, int(mask), res, method.decode('ascii')))
            else:
                outfile.write('%1s LINE %03d %10.4f %10.4f %1d %+10.6f %1s'%(
                    channel, aperture, pix, wav, int(mask), res, method.decode('ascii')))
            outfile.write(os.linesep)

    # write coefficients
    for irow in range(coeff.shape[0]):
        string = ' '.join(['%18.10e'%v for v in coeff[irow]])
        if channel is None:
            outfile.write('COEFF %s'%string)
        else:
            outfile.write('%1s COEFF %s'%(channel, string))
        outfile.write(os.linesep)

    outfile.close()


def load_ident(filename):
    """Load identified line list from an ASCII file.

    Args:
        filename (str): Name of the identification file.

    Returns:
        tuple: A tuple containing:

            * **identlist** (*dict*) – Identified lines for all orders.
            * **coeff** (:class:`numpy.ndarray`) – Coefficients of wavelengths.

    See also:
        :func:`save_ident`
    """
    identlist = {}
    coeff = []

    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#$%^@!':
            continue
        g = row.split()

        key = g[0]
        if key == 'LINE':
            aperture    = int(g[1])
            pixel       = float(g[2])
            wavelength  = float(g[3])
            mask        = bool(g[4])
            residual    = float(g[5])
            method      = g[6].strip()

            item = np.array((aperture,0,pixel,wavelength,0.,mask,residual,
                                method),dtype=identlinetype)
            if aperture not in identlist:
                identlist[aperture] = []
            identlist[aperture].append(item)

        elif key == 'COEFF':
            coeff.append([float(v) for v in g[2:]])

        else:
            pass

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
    """Find the central pixel of an emission line.

    Args:
        flux (:class:`numpy.ndarray`): Flux array.
        x (int): The approximate coordinate of the peak pixel.
        width (int): Window of profile fitting.

    Returns:
        tuple: A tuple containing:

            * **i1** (*int*) – Index of the left side.
            * **i2** (*int*) – Index of the right side.
            * **p1** (*list*) – List of fitting parameters.
            * **std** (*float*) – Standard devation of the fitting.
    """
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
    """Relocate the profile center of the lines.

    Args:
        flux (:class:`numpy.ndarray`): Flux array.
        center (float): Center of the line.

    Returns:
        float: The new center of the line profile.
    """
    y1, y2 = int(center)-3, int(center)+4
    ydata = flux[y1:y2]
    xdata = np.arange(y1,y2)
    p0 = [ydata.min(), ydata.max()-ydata.min(), ydata.argmax()+y1, 2.5]
    p1,succ = opt.leastsq(errfunc2, p0[:], args=(xdata,ydata))
    return p1[2]

def search_linelist(linelistname):
    """Search the line list file and load the list.

    Args:
        linelistname (str): Name of the line list file.

    Returns:
        *string*: Path to the line list file
    """

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

    # seach GAMSE_DATA path
    gamse_data = os.getenv('GAMSE_DATA')
    if len(gamse_data)>0:
        data_path = os.path.join(gamse_data, 'linelist')
        newname = os.path.join(data_path, linelistname+'.dat')
        if os.path.exists(newname):
            return newname

    return None

def load_linelist(filename):
    """Load standard wavelength line list from a given file.

    Args:
        filename (str): Name of the wavelength standard list file.

    Returns:
        *list*: A list containing (wavelength, species).
    """
    linelist = []
    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#%!@':
            continue
        g = row.split()
        wl = float(g[0])
        if len(g)>1:
            species = g[1]
        else:
            species = ''
        linelist.append((wl, species))
    infile.close()
    return linelist

def find_shift_ccf(f1, f2, shift0=0.0):
    """Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.ndarray`): Flux array.
        f2 (:class:`numpy.ndarray`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.

    Returns:
        float: Relative shift between the two flux arrays.
    """
    x = np.arange(f1.size)
    interf = intp.InterpolatedUnivariateSpline(x, f1, k=3)
    func = lambda shift: -(interf(x - shift)*f2).sum(dtype=np.float64)
    res = opt.minimize(func, shift0, method='Powell')
    return res['x']

def find_shift_ccf2(f1, f2, shift0=0.0):
    """Find the relative shift of two arrays using cross-correlation function.

    Args:
        f1 (:class:`numpy.ndarray`): Flux array.
        f2 (:class:`numpy.ndarray`): Flux array.
        shift (float): Approximate relative shift between the two flux arrays.

    Returns:
        float: Relative shift between the two flux arrays.
    """
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


def get_simple_ccf(flux1, flux2, shift_lst):
    """Get cross-correlation function of two fluxes with the given relative
    shift.

    Args:
        flux1 (:class:`numpy.ndarray`): Input flux array.
        flux2 (:class:`numpy.ndarray`): Input flux array.
        shift_lst (:class:`numpy.ndarray`): List of pixel shifts.

    Returns:
        :class:`numpy.ndarray`: Cross-correlation function
    """

    n = flux1.size
    ccf_lst = []
    for shift in shift_lst:
        segment1 = flux1[max(0,shift):min(n,n+shift)]
        segment2 = flux2[max(0,-shift):min(n,n-shift)]
        c1 = math.sqrt((segment1**2).sum())
        c2 = math.sqrt((segment2**2).sum())
        corr = np.correlate(segment1, segment2)/c1/c2
        ccf_lst.append(corr)
    return np.array(ccf_lst)


def find_pixel_drift(spec1, spec2,
        aperture_koffset=(1, 0), pixel_koffset=(1, 0.0)
    ):
    """Find the drift between two spectra. The apertures of the two spectra must
    be aligned.

    The **aperture_offset** is defined as:

        aperture1 + aperture_offset = aperture2

    Args:
        spec1 (:class:`numpy.dtype`): Spectra array.
        spec2 (:class:`numpy.dtype`): Spectra array.
        offset (float): Approximate relative shift between the two spectra
            arrays.

    Returns:
        float: Calculated relative shift between the two spectra arrays.
    """

    aperture_k, aperture_offset = aperture_koffset
    pixel_k, pixel_offset = pixel_koffset

    shift_lst = []
    for item1 in spec1:
        aperture1 = item1['aperture']
        aperture2 = aperture_k*aperture1 + aperture_offset
        m = spec2['aperture'] == aperture2
        if m.sum()==1:
            item2 = spec2[m][0]
            flux1 = item1['flux']
            flux2 = item2['flux']

            #shift = find_shift_ccf(flux1, flux2)
            #shift = find_shift_ccf_pixel(flux1, flux2, 100)
            shift = find_shift_ccf(flux1[::pixel_k], flux2, shift0=pixel_offset)

            shift_lst.append(shift)

    drift = np.median(np.array(shift_lst))
    return drift

class CalibFigure(Figure):
    """Figure class for wavelength calibration.

    Args:
        width (int): Width of figure.
        height (int): Height of figure.
        dpi (int): DPI of figure.
        filename (str): Filename of input spectra.
        channel (str): Channel name of input spectra.
    """

    def __init__(self, width, height, dpi, title):
        """Constuctor of :class:`CalibFigure`.
        """
        super(CalibFigure, self).__init__(figsize=(width/dpi, height/dpi), dpi=dpi)
        self.patch.set_facecolor('#d9d9d9')

        # add axes
        self._ax1 = self.add_axes([0.07, 0.07,0.52,0.87])
        self._ax2 = self.add_axes([0.655,0.07,0.32,0.40])
        self._ax3 = self.add_axes([0.655,0.54,0.32,0.40])

        # add title
        self.suptitle(title, fontsize=15)

        #draw the aperture number to the corner of ax1
        bbox = self._ax1.get_position()
        self._ax1._aperture_text = self.text(bbox.x0 + 0.05, bbox.y1-0.1,
                                  '', fontsize=15)

        # draw residual and number of identified lines in ax2
        bbox = self._ax3.get_position()
        self._ax3._residual_text = self.text(bbox.x0 + 0.02, bbox.y1-0.03,
                                  '', fontsize=13)

    def plot_solution(self, identlist, aperture_lst, plot_ax1=False,  **kwargs):
        """Plot the wavelength solution.

        Args:
            identlist (dict): Dict of identified lines.
            aperture_lst (list): List of apertures to be plotted.
            plot_ax1 (bool): Whether to plot the first axes.
            coeff (:class:`numpy.ndarray`): Coefficient array.
            k (int): `k` value in the relationship `order = k*aperture +
                offset`.
            offset (int): `offset` value in the relationship `order =
                k*aperture + offset`.
            npixel (int): Number of pixels along the main dispersion
                direction.
            std (float): Standard deviation of wavelength fitting.
            nuse (int): Number of lines actually used in the wavelength
                fitting.
            ntot (int): Number of lines identified.
        """
        coeff  = kwargs.pop('coeff')
        k      = kwargs.pop('k')
        offset = kwargs.pop('offset')
        npixel = kwargs.pop('npixel')
        std    = kwargs.pop('std')
        nuse   = kwargs.pop('nuse')
        ntot   = kwargs.pop('ntot')

        label_size = 13  # fontsize for x, y labels
        tick_size  = 12  # fontsize for x, y ticks

        #wave_scale = 'linear'
        wave_scale = 'reciprocal'

        #colors = 'rgbcmyk'

        self._ax2.cla()
        self._ax3.cla()

        if plot_ax1:
            self._ax1.cla()
            x = np.linspace(0, npixel-1, 100, dtype=np.float64)

            # find the maximum and minimum wavelength
            wl_min, wl_max = 1e9,0
            allwave_lst = {}
            for aperture in aperture_lst:
                order = k*aperture + offset
                wave = get_wavelength(coeff, npixel, x, np.repeat(order, x.size))
                allwave_lst[aperture] = wave
                wl_max = max(wl_max, wave.max())
                wl_min = min(wl_min, wave.min())
            # plot maximum and minimum wavelength, to determine the display
            # range of this axes, and the tick positions
            self._ax1.plot([0, 0],[wl_min, wl_max], color='none')
            yticks = self._ax1.get_yticks()
            self._ax1.cla()


        for aperture in aperture_lst:
            order = k*aperture + offset
            color = 'C%d'%(order%10)

            # plot pixel vs. wavelength
            if plot_ax1:
                wave = allwave_lst[aperture]
                if wave_scale=='reciprocal':
                    self._ax1.plot(x, 1/wave, color=color, ls='-', alpha=0.8, lw=0.8)
                else:
                    self._ax1.plot(x, wave, color=color, ls='-', alpha=0.8, lw=0.8)

            # plot identified lines
            if aperture in identlist:
                list1 = identlist[aperture]
                pix_lst = list1['pixel']
                wav_lst = list1['wavelength']
                mask    = list1['mask'].astype(bool)
                res_lst = list1['residual']

                if plot_ax1:
                    if wave_scale=='reciprocal':
                        self._ax1.scatter(pix_lst[mask],  1/wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], 1/wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)
                    else:
                        self._ax1.scatter(pix_lst[mask],  wav_lst[mask],
                                          c=color, s=20, lw=0, alpha=0.8)
                        self._ax1.scatter(pix_lst[~mask], wav_lst[~mask],
                                          c='w', s=16, lw=0.7, alpha=0.8,
                                          edgecolor=color)

                repeat_aper_lst = np.repeat(aperture, pix_lst.size)
                self._ax2.scatter(repeat_aper_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax2.scatter(repeat_aper_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, edgecolor=color)
                self._ax3.scatter(pix_lst[mask], res_lst[mask],
                                  c=color, s=20, lw=0, alpha=0.8)
                self._ax3.scatter(pix_lst[~mask], res_lst[~mask],
                                  c='w', s=16, lw=0.7, alpha=0.8, edgecolor=color)

        self._ax3._residual_text.set_text('R.M.S. = %.5f, N = %d/%d'%(std, nuse, ntot))

        # adjust layout for ax1
        if plot_ax1:
            self._ax1.set_xlim(0, npixel-1)
            if wave_scale == 'reciprocal':
                _y11, _y22 = self._ax1.get_ylim()
                newtick_lst, newticklabel_lst = [], []
                for tick in yticks:
                    if _y11 < 1/tick < _y22:
                        newtick_lst.append(1/tick)
                        newticklabel_lst.append(tick)
                self._ax1.set_yticks(newtick_lst)
                self._ax1.set_yticklabels(newticklabel_lst)
                self._ax1.set_ylim(_y22, _y11)
            self._ax1.set_xlabel('Pixel', fontsize=label_size)
            self._ax1.set_ylabel(u'\u03bb (\xc5)', fontsize=label_size)
            self._ax1.grid(True, ls=':', color='gray', alpha=1, lw=0.5)
            self._ax1.set_axisbelow(True)
            self._ax1._aperture_text.set_text('')
            for tick in self._ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)
            for tick in self._ax1.yaxis.get_major_ticks():
                tick.label1.set_fontsize(tick_size)

        # adjust axis layout for ax2 (residual on aperture space)
        self._ax2.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax2.axhline(y=i*std, color='k', ls=':', lw=0.5)
        x1, x2 = self._ax2.get_xlim()
        x1 = max(x1,aperture_lst.min())
        x2 = min(x2,aperture_lst.max())
        self._ax2.set_xlim(x1, x2)
        self._ax2.set_ylim(-6*std, 6*std)
        self._ax2.set_xlabel('Aperture', fontsize=label_size)
        self._ax2.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)

        ## adjust axis layout for ax3 (residual on pixel space)
        self._ax3.axhline(y=0, color='k', ls='--', lw=0.5)
        for i in np.arange(-3,3+0.1):
            self._ax3.axhline(y=i*std, color='k', ls=':', lw=0.5)
        self._ax3.set_xlim(0, npixel-1)
        self._ax3.set_ylim(-6*std, 6*std)
        self._ax3.set_xlabel('Pixel', fontsize=label_size)
        self._ax3.set_ylabel(u'Residual on \u03bb (\xc5)', fontsize=label_size)
        for tick in self._ax3.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)
        for tick in self._ax3.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_size)


def select_calib_from_database(path, time_key, current_time):
    """Select a previous calibration result in database.

    Args:
        path (str): Path to search for the calibration files.
        time_key (str): Name of the key in the FITS header.
        current_time (str): Time string of the file to be calibrated.

    Returns:
        tuple: A tuple containing:

            * **spec** (:class:`numpy.dtype`): An array of previous calibrated
                spectra.
            * **calib** (dict): Previous calibration results.
    """
    if not os.path.exists(path):
        return None, None
    filename_lst = []
    datetime_lst = []
    for fname in os.listdir(path):
        if re.match('wlcalib.\S*.fits$', fname) is not None:
            filename = os.path.join(path, fname)
            head = fits.getheader(filename)
            dt = dateutil.parser.parse(head[time_key])
            filename_lst.append(filename)
            datetime_lst.append(dt)
    if len(filename_lst)==0:
        return None, None

    # select the FITS file with the shortest interval in time.
    input_datetime = dateutil.parser.parse(current_time)
    deltat_lst = np.array([abs((input_datetime - dt).total_seconds())
                            for dt in datetime_lst])
    imin = deltat_lst.argmin()
    sel_filename = filename_lst[imin]

    # load spec, calib, and aperset from selected FITS file
    f = fits.open(sel_filename)
    head = f[0].header
    spec = f[1].data

    calib = get_calib_from_header(head)

    # aperset seems unnecessary
    #aperset = load_aperture_set_from_header(head, channel=channel)
    #return spec, calib, aperset
    return spec, calib


def recalib(spec, figfilename, title, ref_spec, linelist, ref_calib,
        aperture_koffset=(1, 0), pixel_koffset=(1, None),
        xorder=None, yorder=None, maxiter=None, clipping=None, window_size=None,
        q_threshold=None, direction=None, fit_filter=None
        ):
    """Re-calibrate the wavelength of an input spectra file using another
    spectra as the reference.

    Args:
        spec (:class:`numpy.dtype`): The spectral data array to be wavelength
            calibrated.
        figfilename (str): Filename of the output wavelength figure.
        title (str): A title to display in the calib figure.
        ref_spec (:class:`numpy.dtype`): Reference spectra.
        linelist (str): Name of wavelength standard file.
        coeff (:class:`numpy.ndarray`): Coefficients of the reference wavelength.
        npixel (int): Number of pixels along the main-dispersion direction.
        k (int): -1 or 1, depending on the relationship `order = k*aperture
            + offset`.
        offset (int): coefficient in the relationship `order = k*aperture +
            offset`.
        window_size (int): Size of the window in pixel to search for the
            lines.
        xorder (int): Order of polynomial along X axis.
        yorder (int): Order of polynomial along Y axis.
        maxiter (int): Maximim number of interation in polynomial fitting.
        clipping (float): Threshold of sigma-clipping.
        q_threshold (float): Minimum *Q*-factor of the spectral lines to be
            accepted in the wavelength fitting.
        fit_filter (function): Function checking if a pixel/oder combination is
            within the accepted range.

    Returns:
        dict: A dict containing:

            * **coeff** (:class:`numpy.ndarray`) – Coefficient array.
            * **npixel** (*int*) – Number of pixels along the main
              dispersion direction.
            * **k** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **offset** (*int*) – Coefficient in the relationship `order =
              k*aperture + offset`.
            * **std** (*float*) – Standard deviation of wavelength fitting in Å.
            * **nuse** (*int*) – Number of lines used in the wavelength
              fitting.
            * **ntot** (*int*) – Number of lines found in the wavelength
              fitting.
            * **identlist** (*dict*) – Dict of identified lines.
            * **window_size** (*int*) – Length of window in searching the
              line centers.
            * **xorder** (*int*) – Order of polynomial along X axis in the
              wavelength fitting.
            * **yorder** (*int*) – Order of polynomial along Y axis in the
              wavelength fitting.
            * **maxiter** (*int*) – Maximum number of iteration in the
              wavelength fitting.
            * **clipping** (*float*) – Clipping value of the wavelength fitting.
            * **q_threshold** (*float*) – Minimum *Q*-factor of the spectral
              lines to be accepted in the wavelength fitting.

    See also:
        :func:`wlcalib`
    """

    aperture_k, aperture_offset = aperture_koffset
    pixel_k, pixel_offset       = pixel_koffset

    # unpack ref_calib
    k           = ref_calib['k']
    offset      = ref_calib['offset']
    coeff       = ref_calib['coeff']
    npixel      = ref_calib['npixel']
    xorder      = (xorder, ref_calib['xorder'])[xorder is None]
    yorder      = (yorder, ref_calib['yorder'])[yorder is None]
    maxiter     = (maxiter,  ref_calib['maxiter'])[maxiter is None]
    clipping    = (clipping, ref_calib['clipping'])[clipping is None]
    window_size = (window_size, ref_calib['window_size'])[window_size is None]
    q_threshold = (q_threshold, ref_calib['q_threshold'])[q_threshold is None]

    #if pixel_offset is None:
    if False:
        # find initial shift with cross-corelation functions
        pixel_offset = find_pixel_drift(ref_spec, spec,
                        aperture_koffset = aperture_koffset,
                        pixel_koffset   = pixel_koffset)
        print('calculated shift = ', pixel_offset)

    #message = '{} channel {} shift = {:+8.6f} pixel'.format(
    #            os.path.basename(filename), channel, shift
    #            )
    #print(message)

    # initialize the identlist
    identlist = {}

    # load the wavelengths
    linefilename = search_linelist(linelist)
    if linefilename is None:
        print('Error: Cannot find linelist file: %s'%linelist)
        exit()
    line_list = load_linelist(linefilename)

    x = np.arange(npixel)[::pixel_k] + pixel_k*pixel_offset

    for row in spec:
        # variable alias
        aperture = row['aperture']
        flux     = row['flux']
        # obtain a rough wavelength array according to the input
        # aperture_koffset and pixel_koffset
        old_aperture = (aperture - aperture_offset)/aperture_k

        # now convert the old aperture number to echelle order number (m)
        order = k*old_aperture + offset
        wl = get_wavelength(coeff, npixel, x, np.repeat(order, npixel))
        w1 = min(wl[0], wl[-1])
        w2 = max(wl[0], wl[-1])

        has_insert = False
        for line in line_list:
            if line[0] < w1:
                continue
            if line[0] > w2:
                break

            # wavelength in the range of this order
            # find the nearest pixel to the calibration line
            diff = np.abs(wl - line[0])
            i = diff.argmin()
            i1, i2, param, std = find_local_peak(flux, i, window_size)

            keep = auto_line_fitting_filter(param, i1, i2)
            if not keep:
                continue

            q = param[0]/std
            if q < q_threshold:
                continue
            peak_x = param[1]

            if aperture not in identlist:
                identlist[aperture] = np.array([], dtype=identlinetype)

            # pack the line data
            item = np.array((aperture, order, peak_x, line[0], q, True, 0.0,
                            'a'), dtype=identlinetype)

            identlist[aperture] = np.append(identlist[aperture], item)
            has_insert = True

        if has_insert:
            identlist[aperture] = np.sort(identlist[aperture], order='pixel')

    new_coeff, new_std, new_k, new_offset, new_nuse, new_ntot = fit_wavelength(
        identlist = identlist,
        npixel    = npixel,
        xorder    = xorder,
        yorder    = yorder,
        maxiter   = maxiter,
        clipping  = clipping,
        fit_filter= fit_filter,
        )

    fig_width  = 2500
    fig_height = 1500
    fig_dpi    = 150

    fig = CalibFigure(width  = fig_width,
                      height = fig_height,
                      dpi    = fig_dpi,
                      title  = title,
                      )
    canvas = FigureCanvasAgg(fig)

    fig.plot_solution(identlist,
                      aperture_lst = spec['aperture'],
                      plot_ax1     = True,
                      coeff        = new_coeff,
                      k            = new_k,
                      offset       = new_offset,
                      npixel       = npixel,
                      std          = new_std,
                      nuse         = new_nuse,
                      ntot         = new_ntot,
                      )
    fig.savefig(figfilename)
    plt.close(fig)

    # refresh the direction code
    new_direction = direction[0] + {1:'r', -1:'b'}[new_k] + '-+'[wl[0] < wl[-1]]
    # compare the new direction to the input direction. if not consistent,
    # print a warning
    if direction[1]!='?' and direction[1]!=new_direction[1]:
        print('Warning: Direction code 1 refreshed:',
                direction[1], new_direction[1])
    if direction[2]!='?' and direction[2]!=new_direction[2]:
        print('Warning: Direction code 2 refreshed:',
                direction[2], new_direction[2])


    # pack calibration results
    return {
            'coeff':       new_coeff,
            'npixel':      npixel,
            'k':           new_k,
            'offset':      new_offset,
            'std':         new_std,
            'nuse':        new_nuse,
            'ntot':        new_ntot,
            'identlist':   identlist,
            'window_size': window_size,
            'xorder':      xorder,
            'yorder':      yorder,
            'maxiter':     maxiter,
            'clipping':    clipping,
            'q_threshold': q_threshold,
            'direction':   new_direction,
            }

def find_caliblamp_offset(spec1, spec2, colname1='flux', colname2='flux',
        aperture_k=None, pixel_k=None,
        fig_ccf=None, fig_scatter=None):
    """Find the offset between two spectra.

    The aperture offset is defined as:

    of the same echelle order, `aperture1` in spec1 is marked as
    `k*aperture1 + offset` in spec2.

    Args:
        spec1 (:class:`numpy.dtype`): Input spectra as a numpy structrued array.
        spec2 (:class:`numpy.dtype`): Input spectra as a numpy structrued array.
        colname1 (str): Name of flux column in **spec1**.
        colname2 (str): Name of flux column in **spec2**.
        aperture_k (int): Aperture direction code (1 or -1) between **spec1**
            and **spec2**.
        pixel_k (int): Pixel direction code (1 or -1) between **spec1** and
            **spec2**.
        fig_ccf (string): Name of figure for cross-correlation functions (CCFs).
        fig_scatter (string): Name of figure for peak scatters.

    Returns:
        tuple: A tuple containing:

            * **offset** (*int*): Aperture offset between the two spectra.
            * **shift** (*float*): Pixel shift between the two spectra.
    """

    pixel_shift_lst = np.arange(-100, 100)
    mean_lst    = {(1, 1):[], (1, -1):[], (-1, 1):[], (-1, -1):[]}
    scatter_lst = {(1, 1):[], (1, -1):[], (-1, 1):[], (-1, -1):[]}
    all_scatter_lst = []
    all_mean_lst    = []
    scatter_id_lst = []

    aper1_lst = spec1['aperture']
    aper2_lst = spec2['aperture']
    min_aper1 = aper1_lst.min()
    max_aper1 = aper1_lst.max()
    min_aper2 = aper2_lst.min()
    max_aper2 = aper2_lst.max()

    # determine the maxium absolute offsets between the orders of the two
    # spectra
    maxoff = min(max(aper1_lst.size, aper2_lst.size)//2, 10)
    aperture_offset_lst = np.arange(-maxoff, maxoff)

    def get_aper2(aper1, k, offset):
        if k == 1:
            # (aper2 - min_aper2) = (aper1 - min_aper1) + offset
            # in this case, real_offset = offset - min_aper1 + min_aper2
            aper2 = (aper1 - min_aper1) + offset + min_aper2
        elif k == -1:
            # (aper2 - min_aper2) = -(aper1 - max_aper1) + offset
            # in this cose, real_offset = offset + max_aper1 + min_aper2
            aper2 = -aper1 + max_aper1 + offset + min_aper2
        else:
            raise ValueError
        return aper2

    # aperture_k =  1: same cross-order direction;
    #              -1: reverse cross-order direction.
    if aperture_k is None:
        search_aperture_k_lst = [1, -1]
    elif aperture_k in [1, -1]:
        search_aperture_k_lst = [aperture_k]
    else:
        print('Warning: Unknown aperture_k:', aperture_k)
        raise ValueError

    # pixel_k =  1: same main-dispersion direction;
    #           -1: reverse main-dispersion direction.
    if pixel_k is None:
        search_pixel_k_lst = [1, -1]
    elif pixel_k in [1, -1]:
        search_pixel_k_lst = [pixel_k]
    else:
        print('Warning: Unknown pixel_k:', pixel_k)
        raise ValueError


    for aperture_k in search_aperture_k_lst:
        for aperture_offset in aperture_offset_lst:
            calc_pixel_shift_lst = {1: [], -1: []}
            if fig_ccf is not None:
                fig2 = plt.figure(figsize=(10,8), dpi=150)
                axes2 = { 1: fig2.add_subplot(211),
                         -1: fig2.add_subplot(212),
                         }
            for row1 in spec1:
                aperture1 = row1['aperture']
                aperture2 = get_aper2(aperture1, aperture_k, aperture_offset)
                m = spec2['aperture'] == aperture2
                if m.sum()==0:
                    continue
                row2 = spec2[m][0]
                flux1 = row1[colname1]
                flux2 = row2[colname2]
                for pixel_k in search_pixel_k_lst:
                    '''
                    if aperture_k == -1 and pixel_k == -1:
                        fig1 = plt.figure(dpi=150)
                        ax1 = fig1.gca()
                        ax1.plot(flux1[::pixel_k], 'C0')
                        ax1.plot(flux2, 'C1')
                        ax1.set_title('Aper1 = %d, Aper2 = %d (%d, %d, %d)'%(
                            aperture1, aperture2, aperture_k, aperture_offset,
                            pixel_k))
                        fig1.savefig('check_%d_%d_%d_%02d_%02d_.png'%(
                            aperture_k, aperture_offset, pixel_k, aperture1,
                            aperture2))
                        plt.close(fig1)
                    '''

                    ccf_lst = get_simple_ccf(flux1[::pixel_k], flux2,
                                             pixel_shift_lst)
                    # find the pixel shift
                    calc_shift = pixel_shift_lst[ccf_lst.argmax()]
                    # pack the pixel shift into a list
                    calc_pixel_shift_lst[pixel_k].append(calc_shift)

                    if fig_ccf is not None:
                        axes2[pixel_k].plot(pixel_shift_lst, ccf_lst, alpha=0.4)
                    # pixel direction loop ends here
                # order-by-order loop ends here

            # adjust the ccf figure and save
            if fig_ccf is not None:
                for ax in axes2.values():
                    ax.set_xlim(pixel_shift_lst[0], pixel_shift_lst[-1])
                fig2.savefig(fig_ccf.format(aperture_k, aperture_offset))
                plt.close(fig2)

            # convert calc_pixel_shift_lst to numpy array
            pixel_shift_mean = {1: None, -1: None}
            pixel_shift_std  = {1: None, -1: None}
            for pixel_k in search_pixel_k_lst:
                tmp = np.array(calc_pixel_shift_lst[pixel_k])

                mean = tmp.mean()
                std  = tmp.std()

                mean_lst[(aperture_k, pixel_k)].append(mean)
                scatter_lst[(aperture_k, pixel_k)].append(std)

                # used to search the global minimum shift scatter along all the
                # (aperture_k, aperture_offset, pixel_k) space
                all_mean_lst.append(mean)
                all_scatter_lst.append(std)
                scatter_id_lst.append((aperture_k, aperture_offset, pixel_k))


    # direction loop ends here

    # plot the scatters of peaks and save it as a figure file
    if fig_scatter is not None:
        fig3 = plt.figure(dpi=150, figsize=(8,6))
        ax3 = fig3.gca()
        for key, scatters in scatter_lst.items():
            aperture_k, pixel_k = key
            if len(scatters)==0:
                continue
            ax3.plot(aperture_offset_lst, scatters,
                        color = {1:'C0', -1:'C1'}[aperture_k],
                        ls    = {1:'-',  -1:'--'}[pixel_k],
                        label = 'Aperture k = {}, Pixel k = {}'.format(
                            aperture_k, pixel_k))
        ax3.set_xlabel('Aperture Offset')
        ax3.set_ylabel('Scatter (pixel)')
        ax3.legend(loc='lower right')
        fig3.savefig(fig_scatter)
        plt.close(fig3)

    imin = np.argmin(all_scatter_lst)
    scatter_id = scatter_id_lst[imin]
    result_aperture_k      = scatter_id[0]
    result_aperture_offset = scatter_id[1]
    result_pixel_k         = scatter_id[2]
    result_pixel_offset    = all_mean_lst[imin]

    # convert aperture_offset to real aperture_offset
    real_aperture_offset = {
             1: result_aperture_offset - min_aper1 + min_aper2,
            -1: result_aperture_offset + max_aper1 + min_aper2,
            }[result_aperture_k]
    return (result_aperture_k, real_aperture_offset,
            result_pixel_k,    result_pixel_offset)


def save_calibrated_thar(head, spec, calib, channel):
    """Save the wavelength calibrated ThAr spectra.

    Args:
        head (:class:`astropy.io.fits.Header`):
        spec (:class:`numpy.dtype`):
        calib (tuple):
        channel (str):
    """
    k      = calib['k']
    offset = calib['offset']
    xorder = calib['xorder']
    yorder = calib['yorder']
    coeff  = calib['coeff']

    if channel is None:
        leading_str = 'HIERARCH GAMSE WLCALIB'
    else:
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
    head[leading_str+' K']      = k
    head[leading_str+' OFFSET'] = offset
    head[leading_str+' XORDER'] = xorder
    head[leading_str+' YORDER'] = yorder

    # write the coefficients
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

    head[leading_str+' MAXITER']       = calib['maxiter']
    head[leading_str+' STDDEV']        = calib['std']
    head[leading_str+' WINDOW_SIZE']   = calib['window_size']
    head[leading_str+' SNR_THRESHOLD'] = calib['snr_threshold']
    head[leading_str+' CLIPPING']      = calib['clipping']
    head[leading_str+' NTOT']          = calib['ntot']
    head[leading_str+' NUSE']          = calib['nuse']
    head[leading_str+' NPIXEL']        = calib['npixel']

    file_identlist = []

    # pack the identfied line list
    for aperture, list1 in calib['identlist'].items():
        for row in list1:
            file_identlist.append(row)

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(spec)
    lst = [pri_hdu, tbl_hdu1]
    file_identlist = np.array(file_identlist, dtype=list1.dtype)
    tbl_hdu2 = fits.BinTableHDU(file_identlist)
    lst.append(tbl_hdu2)
    hdu_lst  = fits.HDUList(lst)

    return hdu_lst

def reference_wl_new(spec, calib, head, channel, include_identlist):
    k      = calib['k']
    offset = calib['offset']
    xorder = calib['xorder']
    yorder = calib['yorder']
    coeff  = calib['coeff']

    for row in spec:
       aperture = row['aperture']
       npixel   = row['points']
       order = aperture*k + offset
       wavelength = get_wavelength(coeff, npixel,
                        np.arange(npixel),
                        np.repeat(order, npixel))
       row['order']      = order
       row['wavelength'] = wavelength

    if channel is None:
        leading_str = 'HIERARCH GAMSE WLCALIB'
    else:
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
    head[leading_str+' K']      = k
    head[leading_str+' OFFSET'] = offset
    head[leading_str+' XORDER'] = xorder
    head[leading_str+' YORDER'] = yorder

    # write the coefficients
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        head[leading_str+' COEFF %d %d'%(j, i)] = coeff[j,i]

    head[leading_str+' NPIXEL']        = calib['npixel']
    head[leading_str+' WINDOW_SIZE']   = calib['window_size']
    head[leading_str+' MAXITER']       = calib['maxiter']
    head[leading_str+' CLIPPING']      = calib['clipping']
    head[leading_str+' SNR_THRESHOLD'] = calib['snr_threshold']
    head[leading_str+' NTOT']          = calib['ntot']
    head[leading_str+' NUSE']          = calib['nuse']
    head[leading_str+' STDDEV']        = calib['std']

    pri_hdu  = fits.PrimaryHDU(header=head)
    tbl_hdu1 = fits.BinTableHDU(spec)
    hdu_lst = [pri_hdu, tbl_hdu1]

    if include_identlist:
        file_identlist = []

        # pack the identfied line list
        for aperture, list1 in calib['identlist'].items():
            for row in list1:
                file_identlist.append(row)

        file_identlist = np.array(file_identlist, dtype=list1.dtype)
        tbl_hdu2 = fits.BinTableHDU(file_identlist)
        hdu_lst.append(tbl_hdu2)

    return fits.HDUList(hdu_lst)

def get_time_weight(datetime_lst, datetime):
    """Get weight according to the time interval.

    Args:
        datetime_lst (list):
        datetime (datetime.datetime):

    Returns:
        list: A list of floats as the weights.
    """
    input_datetime = dateutil.parser.parse(datetime)
    dt_lst = [(dateutil.parser.parse(dt) - input_datetime).total_seconds()
                for dt in datetime_lst]
    dt_lst = np.array(dt_lst)
    if len(dt_lst)==1:
        # only one reference in datetime_lst
        weight_lst = [1.0]
    elif (dt_lst<0).sum()==0:
        # all elements in dt_lst > 0. means all references are after the input
        # datetime. then use the first reference
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        weight_lst[0] = 1.0
    elif (dt_lst>0).sum()==0:
        # all elements in dt_lst < 0. means all references are before the input
        # datetime. then use the last reference
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        weight_lst[-1] = 1.0
    else:
        weight_lst = np.zeros_like(dt_lst, dtype=np.float64)
        i = np.searchsorted(dt_lst, 0.0)
        w1 = -dt_lst[i-1]
        w2 = dt_lst[i]
        weight_lst[i-1] = w2/(w1+w2)
        weight_lst[i]   = w1/(w1+w2)

    return weight_lst

def combine_calib(calib_lst, weight_lst):
    """Combine a list of wavelength calibration results.

    Args:
        calib_lst (list):
        weight_lst (list):

    Return:
        dict: The combined wavelength claibration result
    """

    k      = calib_lst[0]['k']
    offset = calib_lst[0]['offset']
    xorder = calib_lst[0]['xorder']
    yorder = calib_lst[0]['yorder']
    npixel = calib_lst[0]['npixel']

    for calib in calib_lst:
        if     calib['k']      != k \
            or calib['offset'] != offset \
            or calib['xorder'] != xorder \
            or calib['yorder'] != yorder \
            or calib['npixel'] != npixel:
            print('Error: calib list is not self-consistent')
            raise ValueError


    # calculate the weighted average coefficients
    coeff = np.zeros_like(calib_lst[0]['coeff'], dtype=np.float64)
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        for calib, weight in zip(calib_lst, weight_lst):
            coeff[j, i] += calib['coeff'][j, i]*weight

    return {'k': k, 'offset': offset, 'xorder': xorder, 'yorder': yorder,
            'npixel': npixel, 'coeff': coeff}


def get_calib_from_header(header):
    """Get calib from FITS header.

    Args:
        header (:class:`astropy.io.fits.Header`): FITS header.

    Returns:
        tuple: A tuple containing calib results.
    """

    prefix = 'HIERARCH GAMSE WLCALIB'
    #prefix = 'HIERARCH EDRS WVCALIB'

    xorder = header[prefix+' XORDER']
    yorder = header[prefix+' YORDER']

    coeff = np.zeros((yorder+1, xorder+1))
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        coeff[j,i] = header[prefix+' COEFF %d %d'%(j, i)]

    calib = {
              'coeff':         coeff,
              'npixel':        header[prefix+' NPIXEL'],
              'k':             header[prefix+' K'],
              'offset':        header[prefix+' OFFSET'],
              'std':           header[prefix+' STDDEV'],
              'nuse':          header[prefix+' NUSE'],
              'ntot':          header[prefix+' NTOT'],
#             'identlist':     calibwindow.identlist,
              'window_size':   header[prefix+' WINDOW_SIZE'],
              'xorder':        xorder,
              'yorder':        yorder,
              'maxiter':       header[prefix+' MAXITER'],
              'clipping':      header[prefix+' CLIPPING'],
              'q_threshold':   header[prefix+' Q_THRESHOLD'],
              #'q_threshold':   header[prefix+' SNR_THRESHOLD'],
              'direction':     header[prefix+' DIRECTION'],
              #'direction':     'xr-',
            }
    return calib


def auto_line_fitting_filter(param, i1, i2):
    """A filter function for fitting of a single calibration line.

    Args:
        param ():
        i1 (int):
        i2 (int):

    Return:
        bool:
    """
    if param[0] <= 0.:
        # line amplitdue too small
        return False
    if param[1] < i1 or param[1] > i2:
        # line center not in the fitting range (i1, i2)
        return False
    if param[2] > 50. or param[2] < 1.0:
        # line too broad or too narrow
        return False
    if param[3] < -0.5*param[0]:
        # background too low
        return False
    return True

def reference_self_wavelength(spec, calib):
    """Calculate the wavelengths for an one dimensional spectra.

    Args:
        spec ():
        calib ():

    Returns:
        tuple: A tuple containing:
    """

    # calculate the wavelength for each aperture
    for row in spec:
        aperture = row['aperture']
        npixel   = row['points']
        order = aperture*calib['k'] + calib['offset']
        wavelength = get_wavelength(calib['coeff'], npixel,
                    np.arange(npixel), np.repeat(order, npixel))
        row['order']      = order
        row['wavelength'] = wavelength

    card_lst = []
    card_lst.append(('K',      calib['k']))
    card_lst.append(('OFFSET', calib['offset']))
    card_lst.append(('XORDER', calib['xorder']))
    card_lst.append(('YORDER', calib['yorder']))
    card_lst.append(('NPIXEL', calib['npixel']))

    # write the coefficients to fits header
    for j, i in itertools.product(range(calib['yorder']+1),
                                  range(calib['xorder']+1)):
        key   = 'COEFF {:d} {:d}'.format(j, i)
        value = calib['coeff'][j,i]
        card_lst.append((key, value))

    # write other information to fits header
    card_lst.append(('WINDOW_SIZE', calib['window_size']))
    card_lst.append(('MAXITER',     calib['maxiter']))
    card_lst.append(('CLIPPING',    calib['clipping']))
    card_lst.append(('Q_THRESHOLD', calib['q_threshold']))
    card_lst.append(('NTOT',        calib['ntot']))
    card_lst.append(('NUSE',        calib['nuse']))
    card_lst.append(('STDDEV',      calib['std']))
    card_lst.append(('DIRECTION' ,  calib['direction']))

    # pack the identfied line list
    identlist = []
    for aperture, list1 in calib['identlist'].items():
        for row in list1:
            identlist.append(row)
    identlist = np.array(identlist, dtype=list1.dtype)

    return spec, card_lst, identlist


def combine_fiber_spec(spec_lst):
    """Combine one-dimensional spectra of different fibers.

    Args:
        spec_lst (dict): A dict containing the one-dimensional spectra for all
            fibers.

    Returns:
        numpy.dtype: The combined one-dimensional spectra
    """
    spec1 = list(spec_lst.values())[0]
    newdescr = [descr for descr in spec1.dtype.descr]
    # add a new column
    newdescr.insert(0, ('fiber', 'S1'))

    newspec = []
    for fiber, spec in sorted(spec_lst.items()):
        for row in spec:
            item = list(row)
            item.insert(0, fiber)
            newspec.append(tuple(item))
    newspec = np.array(newspec, dtype=newdescr)

    return newspec

def combine_fiber_cards(card_lst):
    """Combine header cards of different fibers.

    Args:
        card_lst (dict): FITS header cards of different fibers.

    Returns:
        list: List of header cards.
    """
    newcard_lst = []
    for fiber, cards in sorted(card_lst.items()):
        for card in cards:
            key = 'FIBER {} {}'.format(fiber, card[0])
            value = card[1]
            newcard_lst.append((key, value))
    return newcard_lst

def combine_fiber_identlist(identlist_lst):
    """Combine the identified line list of different fibers.

    Args:
        identlist_lst (dict): Identified line lists of different fibers.

    Returns:
        numpy.dtype
    """
    identlist1 = list(identlist_lst.values())[0]
    newdescr = [descr for descr in identlist1.dtype.descr]
    # add a new column
    newdescr.insert(0, ('fiber', 'S1'))

    newidentlist = []
    for fiber, identlist in sorted(identlist_lst.items()):
        for row in identlist:
            item = list(row)
            item.insert(0, fiber)
            newidentlist.append(tuple(item))
    newidentlist = np.array(newidentlist, dtype=newdescr)

    return newidentlist

def reference_spec_wavelength(spec, calib_lst, weight_lst):
    """Calculate the wavelength of a spectrum with given calibration list and
    weights.

    Args:
        spec (class:`numpy.dtype`):
        calib_lst (list):
        weight_lst (list):

    Returns:
        tuple:

    See also:
        :func:`reference_pixel_wavelength`
    """
    combined_calib = combine_calib(calib_lst, weight_lst)

    k      = combined_calib['k']
    offset = combined_calib['offset']
    xorder = combined_calib['xorder']
    yorder = combined_calib['yorder']
    npixel = combined_calib['npixel']
    coeff  = combined_calib['coeff']

    # calculate the wavelength for each aperture
    for row in spec:
        aperture = row['aperture']
        npoints  = row['points']
        order = aperture*k + offset
        wavelength = get_wavelength(coeff, npixel,
                        np.arange(npoints), np.repeat(order, npoints))
        row['order']      = order
        row['wavelength'] = wavelength

    card_lst = []
    #prefix = 'HIERARCH GAMSE WLCALIB'
    #if fiber is not None:
    #    prefix = prefix + ' FIBER {}'.format(fiber)
    card_lst.append(('K', k))
    card_lst.append(('OFFSET', offset))
    card_lst.append(('XORDER', xorder))
    card_lst.append(('YORDER', yorder))
    card_lst.append(('NPIXEL', npixel))

    # write the coefficients to fits header
    for j, i in itertools.product(range(yorder+1), range(xorder+1)):
        key   = 'COEFF {:d} {:d}'.format(j, i)
        value = coeff[j,i]
        card_lst.append((key, value))

    # write information for every reference
    for icalib, (calib, weight) in enumerate(zip(calib_lst, weight_lst)):
        prefix = 'REFERENCE {:d}'.format(icalib+1)
        card_lst.append((prefix+' FILEID',   calib['fileid']))
        card_lst.append((prefix+' DATE-OBS', calib['date-obs']))
        card_lst.append((prefix+' EXPTIME',  calib['exptime']))
        card_lst.append((prefix+' WEIGHT',   weight))
        card_lst.append((prefix+' NTOT',     calib['ntot']))
        card_lst.append((prefix+' NUSE',     calib['nuse']))
        card_lst.append((prefix+' STDDEV',   calib['std']))

    return spec, card_lst

def reference_pixel_wavelength(pixels, apertures, calib_lst, weight_lst):
    """Calculate the wavelength of a list of pixels with given calibration list
    and weights.

    Args:
        pixels (*list* or class:`numpy.ndarray`):
        apertures (*list* or class:`numpy.ndarray`):
        calib_lst (*list*):
        weight_lst (*list*):

    Returns:
        tuple:

    See also:
        :func:`reference_spec_wavelength`
    """
    pixels    = np.array(pixels)
    apertures = np.array(apertures)

    combined_calib = combine_calib(calib_lst, weight_lst)

    k      = combined_calib['k']
    offset = combined_calib['offset']
    xorder = combined_calib['xorder']
    yorder = combined_calib['yorder']
    npixel = combined_calib['npixel']
    coeff  = combined_calib['coeff']

    orders = apertures*k + offset
    wavelengths = get_wavelength(coeff, npixel, pixels, orders)
    return orders, wavelengths

def reference_wl(infilename, outfilename, regfilename, frameid, calib_lst):
    """Reference the wavelength and write the wavelength solution to the FITS
    file.

    Args:
        infilename (str): Filename of input spectra.
        outfilename (str): Filename of output spectra.
        regfilename (str): Filename of output region file for SAO-DS9.
        frameid (int): FrameID of the input spectra. The frameid is used to
            find the proper calibration solution in **calib_lst**.
        calib_lst (dict): A dict with key of frameids, and values of calibration
            solutions for different channels.

    See also:
        :func:`wlcalib`
    """
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
            if frameid <= min(calib_lst):
                calib = calib_lst[min(calib_lst)][channel]
                refcalib_lst.append(calib)
            elif frameid >= max(calib_lst):
                calib = calib_lst[max(calib_lst)][channel]
                refcalib_lst.append(calib)
            else:
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
        leading_str = 'HIERARCH GAMSE WLCALIB CHANNEL %s'%channel
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
            wl = get_wavelength(coeff, npixel, np.arange(npixel), np.repeat(order, npixel))

            # add wavelength into FITS table
            item = list(row)
            item.append(order)
            item.append(wl)
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
                print('-------'+text)

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

                # draw ticks at integer wavelengths
                pix = np.arange(npixel)
                if wl[0] > wl[-1]:
                    wl  = wl[::-1]
                    pix = pix[::-1]
                f = intp.InterpolatedUnivariateSpline(wl, pix, k=3)
                w1 = wl.min()
                w2 = wl.max()
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
        #file_identlist = np.array(file_identlist, dtype=identlinetype)
        file_identlist = np.array(file_identlist, dtype=list1.dtype)
        tbl_hdu2 = fits.BinTableHDU(file_identlist)
        lst.append(tbl_hdu2)
    hdu_lst  = fits.HDUList(lst)

    if os.path.exists(outfilename):
        os.remove(outfilename)
    hdu_lst.writeto(outfilename)

def get_aperture_coeffs_in_header(head):
    """Get coefficients of each aperture from the FITS header.

    Args:
        head (:class:`astropy.io.fits.Header`): Header of FITS file.

    Returns:
        *dict*: A dict containing coefficients for each aperture and each channel.
    """

    coeffs = {}
    for key, value in head.items():
        exp = '^GAMSE TRACE CHANNEL [A-Z] APERTURE \d+ COEFF \d+$'
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
