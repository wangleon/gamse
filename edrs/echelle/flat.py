import os
import time
import math
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.signal as sg

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..ccdproc import array_to_table, table_to_array
from ..utils.onedarray import pairwise
from ..echelle.trace import ApertureSet

def mosaic_flat_interact(filename_lst, outfile, mosaic_file, reg_file,
    disp_axis=0, mask_suffix = '_msk'):
    '''
    Display an interacitve GUI to mosaic the flat images.

    Args:
        filename_lst (list): A list containing filenames of flat images.
        outfile (string): Filename of the output image.
        mosaic_file (string): Name of the ascii file recording the coefficients
            of the mosaic boundaries.
        reg_file (string): Name of the `.reg` file to be displayed in SAO-DS9.
        disp_axis (integer): Main dispersion axis of the input image. 0 means
            the echelle orders are along the *y* axis. 1 means along *x* axis.
        mask_suffix (string): Surfix of the filenames of masks.
    Returns:
        No returns.

    See Also:
        :func:`mosaic_flat_auto`

    '''

    # parse dispersion axis
    if disp_axis in ['x','X']:
        disp_axis = 1
    elif disp_axis in ['y','Y']:
        disp_axis = 0

    colors = 'rgbcmy'

    # prepare data
    data_lst     = {}  # image list
    head_lst     = {}  # header list
    xsection_lst = {}  # 1d cross-section list

    # n: number of flat images to be mosaic
    nflat = len(filename_lst)

    for ifile, filename in enumerate(filename_lst):
        data, head = fits.getdata(filename, header=True)

        if ifile == 0:
            shape = data.shape
            h, w = shape
            # rotate the dispersion axis to y axis
            if disp_axis == 0:
                # dispersion along Y axis
                n_disp, n_xdisp = h, w
            elif disp_axis == 1:
                # dispersion along X axis
                n_disp, n_xdisp = w, h
        elif data.shape != shape:
            # check if all images have the same shape
            logger.error('Image shape of "%s" (%s) does not match'%(
                         filename, str(data.shape)))
        else:
            pass
            
        if disp_axis == 0:
            # dispersion along Y axis
            xsection = data[n_disp//2,:]
        elif disp_axis == 1:
            # dispersion along X axis
            xsection = data[:,n_disp//2]

        data_lst[filename] = data
        head_lst[filename] = head
        xsection_lst[filename] = xsection

    # plot
    fig = plt.figure(figsize=(15,10), dpi=150, tight_layout=True)

    # initialize the parameters used in the mosaic boundaries
    # suppose n boundaries are identified by hand. there are n+1 areas.

    fig.bound_lst = np.zeros((1,n_disp),dtype=np.int32)
    # fig.bound_lst is a numpy array with n+1 x yrows
    # [
    #   [0, 0, 0, ... ... 0]
    # ]

    fig.boundcoeff_lst = []
    # fig.boundcoeff_lst is a list with n elements containing the coefficients
    # of the boundaries line polynomials

    fig.nodes = np.array([0])
    # fig.nodes is a numpy 1-d array with n+1 elements indicating the pixel
    # number of the boundary lines at center column of the main dispersion
    # direction. The first element is always 0, and the last element is always
    # n_disp (number of pixels along the dispersion direction).

    fig.select_lst = [None]

    ax_lst = {}
    for i in range(nflat):
        ax = fig.add_subplot(nflat+1, 1, i+1)
        # allocate filename to each ax
        filename = filename_lst[i]
        ax_lst[filename] = ax
        ax.filename = filename
    ax = fig.add_subplot(nflat+1, 1, nflat+1)
    ax.filename = None

    def replot():
        for i, filename in enumerate(filename_lst):
            ax = ax_lst[filename]
            y1,y2 = ax.get_ylim()
            ax.cla()
            # draw flat cross section with light lines
            head     = head_lst[filename]
            xsection = xsection_lst[filename]
            color    = colors[i%6]
            ax.plot(xsection, color=color, ls='-', alpha=0.3,
                    label='%s exptime=%.2f'%(filename, head['EXPTIME'])
                    )
            if len(fig.nodes)>1:
                # draw boundaries of mosaic flats
                for x in fig.nodes:
                    ax.axvline(x,color='k',ls='--')
                # draw selected flat cross section with solid lines
                for j, xfrom in enumerate(fig.nodes):
                    if j == len(fig.nodes)-1:
                        xto = n_xdisp
                    else:
                        xto = fig.nodes[j+1]
                    select = fig.select_lst[j]
                    if select is not None and select==filename:
                        ax.plot(np.arange(xfrom,xto), xsection[xfrom:xto],
                                color=color, ls='-')
            ax.set_xlim(0, n_xdisp-1)
            # if not first drawing, do not change the y range
            if len(ax.get_ylabel().strip())!=0:
                ax.set_ylim(y1,y2)
            ax.set_ylabel('Y')
            leg = ax.legend(loc='upper right')
            leg.get_frame().set_alpha(0.1)
            for text in leg.get_texts():
                text.set_fontsize(10)

        # process the mosaic flat
        ax = fig.get_axes()[-1]
        ax.cla()
        for i in range(nflat):
            filename = filename_lst[i]
            if len(fig.nodes)>1:
                # draw mosaic flat boundaries
                for x in fig.nodes[1:]:
                    ax.axvline(x,color='k',ls='--')
                # draw selected flat cross sections
                for j, xfrom in enumerate(fig.nodes):
                    if j == len(fig.nodes)-1:
                        xto = n_xdisp
                    else:
                        xto = fig.nodes[j+1]
                    select = fig.select_lst[j]
                    if select is not None and select==filename:
                        xsection = xsection_lst[filename]
                        ax.plot(np.arange(xfrom,xto),xsection[xfrom:xto],
                                color='k',ls='-')
        ax.set_xlim(0, n_xdisp-1)

        fig.canvas.draw()

    def onclick(event):
        '''Select an area when clicking on the figure.
        '''
        ax = event.inaxes
        if ax is not None and ax.filename is not None:
            i = np.searchsorted(fig.nodes, event.xdata) - 1

            # select and deselect
            if fig.select_lst[i] is None:
                fig.select_lst[i] = ax.filename
            else:
                fig.select_lst[i] = None

            replot()

    def onpress(event):
        '''Add or remove a boundary when pressing 'a' or 'd' on the keyboard.
        '''
        ax = event.inaxes
        if ax is not None and ax.filename is not None:
            if event.key == 'a':
                # when press 'a', add boundary
                data = data_lst[ax.filename]

                coeff = detect_gap(np.transpose(data),
                                   event.xdata,
                                   ccf_ulimit=30,
                                   ccf_llimit=30,
                                   )
                # now calculate the y-pixels of this boundary line
                norm_y = np.arange(n_disp, dtype=np.float32)/n_disp
                bound = np.int32(np.round(np.polyval(coeff, norm_y)*n_xdisp))
                # the node in the central column is bound[int(yrows/2.]]
                # now find the index of this node in the fig.nodes
                ii = np.searchsorted(fig.nodes, bound[n_disp//2])
                # insert this boundary line into fig.bound_lst
                fig.bound_lst = np.insert(fig.bound_lst,ii,np.array([bound]),axis=0)
                # because ii is got from fig.nodes, of which the first element
                # is always 0. so the index in fig.boundcoeff_lst should be ii-1
                fig.boundcoeff_lst.insert(ii-1,coeff)
                # fig.nodes are the pixels of the central column in fig.bound_lst
                fig.nodes = fig.bound_lst[:,n_disp//2]
                print(fig.nodes)

                # re-initialize the selected areas
                fig.select_lst = [None for v in fig.nodes]

            elif event.key == 'd':
                # when press 'd', delete a boundary
                if len(fig.nodes)>0:
                    for i,p in enumerate(fig.nodes):
                        # find the closet boundary
                        if i>0 and abs(p-event.xdata) < n_xdisp/100.:
                            fig.bound_lst   = np.delete(fig.bound_lst,i,axis=0)
                            fig.boundcoeff_lst.pop(i-1)
                            fig.nodes       = fig.bound_lst[:,n_disp//2]
                            fig.select_lst.pop(i)
                            break
            else:
                pass
            replot()

    # first drawing
    replot()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show()
    #_ = raw_input('Press [Enter] to continue ')

    # check final mosaic flat
    if None in fig.select_lst:
        logger.error('Mosaic flat is not completed')
        raise ValueError

    # calculate mosaic flat and its mask
    flat = np.zeros((h,w))
    flat_mask = np.zeros_like(flat, dtype=np.int16)
    for i in range(fig.nodes.size):
        yfrom = fig.bound_lst[i]
        if i == fig.nodes.size - 1:
            yto = np.repeat(h,w)
        else:
            yto = fig.bound_lst[i+1]
        #xfrom = xfrom.reshape((-1,1))
        #xto   = xto.reshape((-1,1))
        yfrom = yfrom.reshape((1,-1))
        yto   = yto.reshape((1,-1))

        filename = fig.select_lst[i]
        y,x = np.mgrid[:h,:w]
        m = (y >= yfrom)*(y < yto)
        colorflat,head = fits.getdata(filename,header=True)
        # now get the filename for mask
        mask_filename = '%s%s.fits'%(filename[0:-5], mask_suffix)
        # read data from mask file
        mtable = fits.getdata(mask_filename)
        colorflat_mask  = table_to_array(mtable, colorflat.shape)
        # make sure the dispersion axis is y
        if disp_axis == 0:
            colorflat = np.transpose(colorflat)
            colorflat_mask = np.transpose(colorflat_mask)
        #flat += m*colorflat/head['EXPTIME']
        flat += m*colorflat
        flat_mask += m*colorflat_mask
    header = fits.Header()
    #header['EXPTIME'] = 1.0
    if disp_axis == 0:
        flat = np.transpose(flat)
        flat_mask = np.transpose(flat_mask)
    fits.writeto(outfile, flat, header, overwrite=True)
    outfile_mask = '%s%s.fits'%(outfile[0:-5], mask_suffix)
    mtable = array_to_table(flat_mask)
    fits.writeto(outfile_mask, mtable, overwrite=True)

    # save boundary coefficients into an ascii file
    outfile1 = open(mosaic_file,'w')
    for coeff in fig.boundcoeff_lst:
        string = ' '.join(['%+12.10e'%v for v in coeff])
        outfile1.write('boundary %s%s'%(string, os.linesep))
    # save the selected areas for each filename
    for filename in fig.select_lst:
        outfile1.write('select %s'%filename+os.linesep)
    outfile1.close()

    # save boundaries data in a .reg file
    save_mosaic_reg(filename  = reg_file,
                    coeff_lst = fig.boundcoeff_lst,
                    disp_axis = disp_axis,
                    shape     = shape,
                    )

    # display the final flat and the boundaries in a ds9 window
    os.system('ds9 -language en %s -region %s'%(outfile, reg_file))

def detect_gap(data, x0, ccf_ulimit=50, ccf_llimit=50, step=50, order=4):
    '''
    Detect the curve of gap between two orders along Y-axis.

    This is realized by calculating the cross-correlation function and detect
    the position of the maximum value.

    Args:
        data (2-d :class:`numpy.array`): Data image as a numpy 2d-array.
        x0 (integer): Starting coordinate.
        ccf_ulimit (integer): Upper limit to **x0** of the data segment used to
            calculate CCF.
        ccf_llimit (integer): Lower limit to **x0** of the data segment used to
            calculate CCF.
        step (integer): Step of searching the gap.
        order (integer): Degree of polynomial used to fit the boundary.
    Returns:
        :class:`numpy.array`: Cofficients of the polynomial.


    '''

    h, w = data.shape
    row0, row1 = h//2, h//2
    xpoint = x0
    x1 = int(xpoint - ccf_ulimit)
    x2 = int(xpoint + ccf_llimit)
    data1 = data[row1,x1:x2]
    xnode_lst = np.array([xpoint])
    ynode_lst = np.array([row1])
    # search direction
    direction = +1
    while(True):
        row2 = row1 + step*direction
        if row2>h-1:
            # reverse the search direction
            direction = -1
            row1 = row0
            row2 = row1 + step*direction
            xpoint = x0
        elif row2<0:
            break
        x1 = int(xpoint - ccf_ulimit)
        x2 = int(xpoint + ccf_llimit)
        try:
            data1 = data[row1,x1:x2]
            data2 = data[row2,x1:x2]
        except:
            row1 = row2
            continue
        f = intp.InterpolatedUnivariateSpline(np.arange(data2.size),data2,k=3)
        shift_lst = np.arange(-10, 10, dtype=np.float32)
        corre_lst = []
        n1 = math.sqrt((data1**2).sum())
        # calculate cross-correlation function
        for j in shift_lst:
            data3 = f(np.arange(data2.size)+j)
            corr = (data1*data3).sum()
            n2 = math.sqrt((data3**2).sum())
            corre_lst.append(corr/n2)
        corre_lst = np.array(corre_lst)/n1
        f2 = InterpolatedUnivariateSpline(shift_lst,-corre_lst,k=3)
        # find the maximum value of cross correlation function
        result = opt.minimize(f2, 0.0, method='BFGS')
        shift = result.x
        xpoint += shift
        row1 = row2
        if -w/2 < xpoint < w*1.5:
            ii = np.searchsorted(ynode_lst,row2)
            ynode_lst = np.insert(ynode_lst,ii,row2)
            xnode_lst = np.insert(xnode_lst,ii,xpoint)

    # fit the trend with polynomial
    # normalize x and y axis
    xfit = ynode_lst/h
    yfit = xnode_lst/w
    coeff = np.polyfit(xfit,yfit,deg=order)
    return coeff

def load_mosaic(filename):
    '''
    Read mosaic boundary information from an existing ASCII file.

    Args:
        filename (string): Name of the mosaic file.
    Returns:
        tuple: A tuple containing (`coeff_lst`, `select_area`), where

            * **coeff_lst** (*list*): A list containing coefficients of
              polynomials for the boundaries.
            * **select_area** (*dict*): A dict containing the selected areas.

    '''
    coeff_lst = []
    select_area = {}

    logger.info('Read mosaic information from "%s"'%filename)

    if not os.path.exists(filename):
        logger.error('Mosaic file "%s" does not exist'%filename)
        exit()

    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row)==0 or row[0] in '#%':
            continue
        g = row.split()
        if g[0] == 'boundary':
            coeff = [float(v) for v in g[1:]]
            coeff_lst.append(coeff)
        elif g[0] == 'select' and g[1] == 'file':
            filename = g[2]
            selects = [bool(int(v)) for v in g[3:]]
            select_area[filename] = selects
        else:
            pass
    infile.close()

    # number of boundary lines
    nbounds = len(coeff_lst)

    logger.info('Load %d boundaries.'%nbounds)
    logger.info('Load %d selected images.'%len(select_area))

    # check whether the number of boundaries and the number of selected areas
    # are consistent.
    for filename, selects in select_area.items():
        if len(selects) != nbounds + 1:
            logger.error(
                'Length of selected area for "%s" (%d) != N(boundaries) + 1'%(
                filename, len(selects)))

    # check whether every element in the final mosaic is 1.
    for i in range(nbounds+1):
        sumi = 0
        for selects in select_area.values():
            sumi += selects[i]
        if sumi != 1:
            logger.error('Multiple selections for area number %d'%i)

    return coeff_lst, select_area


def mosaic_flat_auto(filename_lst, outfile, aperture_set_lst, max_count):
    '''
    Mosaic flat images automatically.

    Args:
        filename_lst (list): A list containing filenames of flat images.
        outfile (string): Filename of the output image.
        aperture_set_lst (list): Dict of :class:`ApertureSet`.
        max_count (float): Maximum count.
    Returns:
        No returns.

    See Also:
        :func:`mosaic_flat_interact`

    '''

    all_aperloc_lst = []
    # all_aperloc_lst  = [
    #  [tracename1: aper_loc, tracename2: aper_loc],
    #  [tracename1: aper_loc, tracename2: aper_loc],
    #  [tracename1: aper_loc, tracename2: aper_loc],
    # ]

    tracename_lst = []

    for itrace, (tracename, aperset) in enumerate(aperture_set_lst.items()):

        # add tracename ot tracename list
        if tracename not in tracename_lst:
            tracename_lst.append(tracename)

        for aper, aper_loc in aperset.items():
            if itrace == 0:
                # append all the apertures in the first trace file into the
                # aperloc list
                all_aperloc_lst.append({tracename: aper_loc})
            else:
                insert = False
                for ilist, list1 in enumerate(all_aperloc_lst):
                    # one aperture should not contain more than 1 apertures
                    # from the same trace file.
                    if tracename in list1:
                        continue
                    # calculate the relative distances.
                    for _tracename, _aperloc in list1.items():
                        distance = aper_loc.get_distance(_aperloc)
                        # get local seperations
                        loc_sep = aperset.get_local_seperation(aper)
                        if abs(distance)<0.3*loc_sep:
                            # append this aperture to an existing aperture
                            all_aperloc_lst[ilist][tracename] = aper_loc
                            insert = True
                            break
                    # if already added to an existing aperture, skip the rest
                    # apertures
                    if insert:
                        break

                # if this aperture does not belong to any existing aperture,
                # append it as a new aperture
                if not insert:
                    all_aperloc_lst.append({tracename: aper_loc})

    # sort the tracename list
    tracename_lst.sort()

    # prepare the information written to running log
    message = ['Aperture Information for Different Flat Files:']
    _msg1 = ['%-20s'%tracename for tracename in tracename_lst]
    _msg2 = ['center, N (sat), max' for tracename in tracename_lst]
    message.append('| '+(' | '.join(_msg1))+' |')
    message.append('| '+(' | '.join(_msg2))+' |')


    mosaic_aperset = ApertureSet()
    for list1 in all_aperloc_lst:
        # add information to running log
        _msg = []
        for tracename in tracename_lst:
            if tracename in list1:
                aper_loc = list1[tracename]
                _msg.append('%4d %4d %10.1f'%(
                    aper_loc.get_center(), aper_loc.nsat, aper_loc.max))
            else:
                _msg.append(' '*20)
        message.append('| '+(' | '.join(_msg))+' |')

        # pick up the best trace file for each aperture
        nosat_lst = {tracename: aper_loc
                    for tracename, aper_loc in list1.items()
                    if aper_loc.nsat == 0 and aper_loc.max<max_count}

        if len(nosat_lst)>0:
            # if there's aperture without saturated pixels, find the one
            # with largest median values
            nosat_sort_lst = sorted(nosat_lst.items(),
                                key=lambda item: item[1].median)
            pick_tracename, pick_aperloc = nosat_sort_lst[-1]
        else:
            # all apertures are saturated. Then find the aperture that has
            # the least number of saturated pixels.
            sat_sort_lst = sorted(list1.items(), key=lambda item: item[1].nsat)
            pick_tracename, pick_aperloc = sat_sort_lst[0]

        setattr(pick_aperloc, 'tracename', pick_tracename)
        mosaic_aperset.add_aperture(pick_aperloc)

    logger.info((os.linesep+' '*3).join(message))
    mosaic_aperset.sort()

    message = ['Flat Mosaic Information',
                'aper, yposition, flatname, N (sat), Max (count)']
    for aper, aper_loc in mosaic_aperset.items():
        message.append('%4d %5d %-15s %4d %10.1f'%(
            aper, aper_loc.get_center(), aper_loc.tracename, aper_loc.nsat,
            aper_loc.max))
    logger.info((os.linesep+' '*3).join(message))

    # read flat data and check the shape consistency
    prev_shape = None
    flatdata_lst, maskdata_lst = {}, {}
    for tracename in aperture_set_lst:
        for filename in filename_lst:
            if os.path.basename(filename)[0:-5]==tracename:
                data = fits.getdata(filename)
                flatdata_lst[tracename] = data
                maskdata_lst[tracename] = np.zeros_like(data, dtype=np.bool)
                shape = data.shape
                if prev_shape is not None and shape != prev_shape:
                    logger.error(
                        'Image shape of "%s" (%d x %d) does not match previous (%d x %d)'%(
                        tracename, shape[0], shape[1], prev_shape[0], prev_shape[1])
                    )
                prev_shape = shape
    
    for iaper, (aper, aper_loc) in enumerate(sorted(mosaic_aperset.items())):
        tracename = aper_loc.tracename
        if aper == 0:
            maskdata_lst[tracename][:,:] = True
        elif tracename != prev_tracename:
            prev_aper_loc = mosaic_aperset[iaper-1]

            h, w = aper_loc.shape
            if aper_loc.direct == 0:
                # aperture along Y axis
                center_line = aper_loc.position(np.arange(h))
                prev_center_line = prev_aper_loc.position(np.arange(h))
                cut_bound = (center_line + prev_center_line)/2.
                yy, xx = np.mgrid[:h:,:w:]
                m = xx > np.round(cut_bound)
            elif aper_loc.direct == 1:
                # aperture along X axis
                center_line = aper_loc.position(np.arange(w))
                prev_center_line = prev_aper_loc.position(np.arange(w))
                cut_bound = (center_line + prev_center_line)/2.
                yy, xx = np.mgrid[:h:,:w:]
                m = yy > np.round(cut_bound)
            maskdata_lst[prev_tracename][m] = False
            maskdata_lst[tracename][m] = True

        prev_tracename = tracename

    mos_flatdata = np.zeros(shape, dtype=np.float32)
    for tracename, maskdata in sorted(maskdata_lst.items()):
        flatdata = flatdata_lst[tracename]
        mos_flatdata += flatdata*maskdata

    # save the mosaic flat as FITS file
    fits.writeto(outfile, mos_flatdata, overwrite=True)

    return mosaic_aperset

def save_mosaic_reg(filename, coeff_lst, disp_axis, shape, npoints=20):
    '''
    Save boundaries data in a SAO-DS9 region file.

    Args:
        filename (string): Filename of the output region file.
        coeff_lst (list): List of coefficients.
        disp_axis (integer): 0 or 1, depending on the dispersion axis.
        shape (tuple): A tuple containing the shape of the image.
        npoints (integer): Number of sampling points.
    Returns:
        No returns.
    '''
    outfile = open(filename, 'w')
    outfile.write('# Region file format: DS9 version 4.1'+os.linesep)
    #outfile.write('# Filename: flat.fits'+os.linesep)
    outfile.write('global color=green dashlist=8 3 width=1 font="normal" ')
    outfile.write('select=0 highlite=1 dash=0 fixed=1 edit=0 move=0 ')
    outfile.write('delete=0 include=1 source=1'+os.linesep)
    outfile.write('physical'+os.linesep)

    h, w = shape

    if disp_axis == 0:
        # orders are along Y axis
        n_disp, n_xdisp = h, w
    elif disp_axis == 1:
        # orders are along X axis
        n_disp, n_xdisp = w, h

    pseudo_x  = np.linspace(0.5, n_disp-0.5, npoints)
    pseudo_xr = np.roll(pseudo_x, -1)
    for coeff in coeff_lst:
        pseudo_y  = np.polyval(coeff, pseudo_x/n_disp)*n_xdisp
        pseudo_yr = np.roll(pseudo_y, -1)
        for x1, y1, x2, y2 in list(zip(pseudo_x, pseudo_y, pseudo_xr, pseudo_yr))[0:-1]:
            if disp_axis == 0:
                x1, y1, x2, y2 = y1, x1, y2, x2
            outfile.write('line(%.1f,%.1f,%.1f,%.1f) # line=0 0%s'%(
                           x1+1,y1+1,x2+1,y2+1, os.linesep))
    outfile.close()

def test():
    outfile2 = open(reg_file, 'w')
    outfile2.write('# Region file format: DS9 version 4.1'+os.linesep)
    outfile2.write('# Filename: flat.fits'+os.linesep)
    outfile2.write('global color=green dashlist=8 3 width=1 font="normal" ')
    outfile2.write('select=0 highlite=1 dash=0 fixed=1 edit=0 move=0 ')
    outfile2.write('delete=0 include=1 source=1'+os.linesep)
    outfile2.write('physical'+os.linesep)
    # save data every 50 points
    step = 50
    ynode_lst = np.arange(0, yrows, step)
    if ynode_lst[-1] != yrows - 1:
        ynode_lst = np.append(ynode_lst, yrows-1)
    for bound in fig.bound_lst[1:]:
        for j,x in enumerate(ynode_lst[0:-1]):
            y1 = x
            x1 = bound[y1]
            y2 = ynode_lst[j+1]
            x2 = bound[y2]
            if disp_axis == 0:
                x1, y1 = y1, x1
                x2, y2 = y2, x2
            outfile2.write('line(%d,%d,%d,%d) # line=0 0'%(
                            x1+1,y1+1,x2+1,y2+1)+os.linesep)
    outfile2.close()

def get_flatfielding(data, mask, apertureset, nflat, slit_step=64,
        q_threshold=30, param_deg=7, fig_aperpar=None, fig_overlap=None,
        fig_slit=None, slit_file=None,
    ):
    '''Get the flat fielding image from the input file.

    Args:
        data (:class:`numpy.array`): Image data of flat fielding.
        mask (:class:`numpy.array`): Mask data of flat fielding.
        apertureset (:class:`ApertureSet`): Echelle apertures detected in the
            input file.
        nflat (integer): Number of flat fielding frames combined.
        slit_step (integer): Step of slit scanning.
        q_threshold (float): Threshold of *Q*-factor.
        param_deg (integer): Degee of parameters fitting.
        fig_aperpar (string): Path to the image of aperture profile parameters.
        fig_overlap (string): Path to the image of overlapped slit profiles.
        fig_slit (string): Path to the image of slit functions.
        slit_file (string): Path to the ASCII file of slit functions.

    Returns:
        :class:`numpy.array`: 2D response map.

    '''
    # define the fitting and error functions
    def gaussian_bkg(A, center, fwhm, bkg, x):
        s = fwhm/2./math.sqrt(2*math.log(2))
        return A*np.exp(-(x-center)**2/2./s**2) + bkg
    def fitfunc(p, x):
        return gaussian_bkg(p[0], p[1], p[2], p[3], x)
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    # define fitting and error functions
    def fitfunc2(p, xdata, interf):
        A, k, c, bkg = p
        return A*interf((xdata-c)*k) + bkg
    def errfunc2(p, xdata, ydata, interf):
        n = xdata.size
        return ydata - fitfunc2(p, xdata, interf)

    h, w = data.shape

    # find saturation mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # find the central positions and boundaries for each aperture
    newx = np.arange(w)
    positions = apertureset.get_positions(newx)
    bounds = apertureset.get_boundaries(newx)

    plot_overlap = (fig_overlap is not None)
    plot_aperpar = (fig_aperpar is not None)
    plot_slit    = (fig_slit is not None)
    plot_single  = False
    fig_fitting = 'flat_single_%04d_%02d.png'
    plot_fitting = False

    ##########first step, scan each column and get the slit function############
    # construct the x-coordinates for slit function
    left, right, step = -4, +4, 0.1
    xnodes = np.arange(left, right+1e-5, step)

    # scanning column list
    x_lst = np.arange(0, w, slit_step)
    if x_lst[-1] != w-1:
        x_lst = np.append(x_lst, w-1)

    # initialize the array for slit function
    slit_array = np.zeros((xnodes.size, x_lst.size))
    # prepare the fitting list
    fitting_lst = {'A': {}, 'fwhm': {}, 'bkg': {}, 'c': {}}
    # scan each column
    for ix, x in enumerate(x_lst):
        x = int(x)

        if plot_overlap:
            # fig2 is the overlapped profiles
            fig2 = plt.figure(figsize=(8,6), dpi=150)
            ax21 = fig2.add_subplot(211)
            ax22 = fig2.add_subplot(212)

        # initialize arrays to calcuate overlapped slit functions
        all_x, all_y, all_r = [], [], []

        # loop over all apertures
        for aper in sorted(apertureset.keys()):
            cen = positions[aper][x]
            b1 = bounds[aper][0][x]
            b2 = bounds[aper][1][x]
            b1 = np.int32(np.round(np.maximum(b1, 0)))
            b2 = np.int32(np.round(np.minimum(b2, h)))
            if b2-b1 <= 5:
                continue
            xdata = np.arange(b1, b2)
            ydata = data[b1:b2, x]
            _satmask = sat_mask[b1:b2, x]
            _badmask = bad_mask[b1:b2, x]

            # plot a single profile fitting
            if plot_single:
                n = aper%9
                if n==0:
                    figi = plt.figure(figsize=(12,8), dpi=150)
                    figi.suptitle('X = %d'%x)
                axi = figi.add_axes([0.06+(n%3)*0.32, 0.08+(2-n//3)*0.305,
                                     0.27,0.24])
                axi.plot(xdata, ydata, 'wo',
                        markeredgewidth=1, markeredgecolor='k')
                axi.axvline(cen, color='k', ls='--')

            # fit the profile if saturated pixels less than 3 and bad pixels
            # less than 3
            if _satmask.sum() < 3 and _badmask.sum() < 3:

                # iterative fitting using gaussian + bkg function
                p0 = [ydata.max()-ydata.min(), (b1+b2)/2., 3.0, ydata.min()]
                #_m = np.ones_like(xdata, dtype=np.bool)
                _m = (~_satmask)*(~_badmask)
                for i in range(10):
                    p1, succ = opt.leastsq(errfunc, p0,
                                args=(xdata[_m], ydata[_m], fitfunc))
                    res = errfunc(p1, xdata, ydata, fitfunc)
                    std = res[_m].std(ddof=1)
                    _new_m = (np.abs(res) < 3*std)*_m
                    if _m.sum() == _new_m.sum():
                        break
                    _m = _new_m

                A, c, fwhm, bkg = p1
                snr = A/std
                s = fwhm/2./math.sqrt(2*math.log(2))
                if b1 < c < b2 and A > 50 and 2 < fwhm < 10:
                    # pack the fitting parameters
                    fitting_lst['A'][(aper, x)]    = A
                    fitting_lst['c'][(aper, x)]    = c
                    fitting_lst['fwhm'][(aper, x)] = fwhm
                    fitting_lst['bkg'][(aper, x)]  = bkg

                    norm_x = (xdata[_m]-c)/s
                    norm_y = (ydata[_m]-bkg)/A
                    norm_r = res[_m]/A

                    # pack normalized x, y, r
                    for _norm_x, _norm_y, _norm_r in zip(norm_x, norm_y, norm_r):
                        all_x.append(_norm_x)
                        all_y.append(_norm_y)
                        all_r.append(_norm_r)

                    if plot_single:
                        axi.plot(xdata[_m], ydata[_m], 'ko')
                        newx = np.arange(b1, b2+1e-3, 0.1)
                        axi.plot(newx, fitfunc(p1, newx), 'r-')
                        axi.axvline(c, color='r', ls='--')
                        axi.plot(xdata[_satmask], ydata[_satmask],'yo',ms=2)
                        axi.plot(xdata[_badmask], ydata[_badmask],'go',ms=2)

            if plot_single:
                _y1, _y2 = axi.get_ylim()
                axi.text(0.95*b1+0.05*b2, 0.15*_y1+0.85*_y2,
                         'Aperture %2d'%aper)
                axi.set_xlim(b1, b2)
                if n%9==8 or aper==max(apertureset.keys()):
                    figi.savefig(fig_fitting%(x, aper))
                    plt.close(figi)
        # now aperture loop ends

        # convert all_x, all_y, all_r to numpy arrays
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_r = np.array(all_r)

        # construct slit function for this column
        step = 0.1
        _m = np.ones_like(all_x, dtype=np.bool)
        for k in range(20):
            #find y nodes
            ynodes = []
            for c in xnodes:
                _m1 = np.abs(all_x[_m]-c) < step/2
                ynodes.append(all_y[_m][_m1].mean(dtype=np.float64))
            ynodes = np.array(ynodes)
            # smoothing
            ynodes = sg.savgol_filter(ynodes, window_length=9, polyorder=5)
            f = intp.InterpolatedUnivariateSpline(xnodes, ynodes, k=3, ext=3)
            res = all_y - f(all_x)
            std = res[_m].std()
            _new_m = np.abs(res) < 3*std
            if _new_m.sum() == _m.sum():
                break
            _m = _new_m
        slit_array[:,ix] = ynodes

        # plot the overlapped slit functions
        # _s = 2.35482 = FWHM/sigma for gaussian function
        _s = 2*math.sqrt(2*math.log(2))
        if plot_overlap:
            ax21.plot(all_x, all_y, 'ro', ms=3, alpha=0.3, markeredgewidth=0)
            ax21.plot(all_x[_m], all_y[_m], 'ko', ms=1, markeredgewidth=0)
            ax21.plot(xnodes, ynodes, 'b-')
            ax22.plot(all_x, all_y-gaussian_bkg(1, 0, _s, 0, all_x),
                      'ro', ms=3, alpha=0.3, markeredgewidth=0)
            ax22.plot(all_x[_m], all_y[_m]-gaussian_bkg(1, 0, _s, 0, all_x[_m]),
                      'ko', ms=1, markeredgewidth=0)
            ax22.plot(xnodes, ynodes - gaussian_bkg(1, 0, _s, 0, xnodes), 'b-')
            newxx = np.arange(-5, 5+1e-5, 0.01)
            ax21.plot(newxx, gaussian_bkg(1, 0, _s, 0, newxx), 'k-', alpha=0.5)
            ax21.grid(True)
            ax22.grid(True)
            ax21.set_xlim(-7,7)
            ax22.set_xlim(-7,7)
            ax21.set_ylim(-0.2, 1.2)
            ax22.set_ylim(-0.25, 0.25)
            ax21.set_ylim(-0.2, 1.2)
            fig2.savefig(fig_overlap%x)
            plt.close(fig2)
    # column loop ends here

    # write the slit function into an ascii file
    if slit_file is not None:
        slitoutfile = open(slit_file, 'w')
        for row in np.arange(xnodes.size):
            slitoutfile.write('%5.2f'%xnodes[row])
            for col in np.arange(x_lst.size):
                slitoutfile.write(' %12.8f'%slit_array[row, col])
            slitoutfile.write(os.linesep)
        slitoutfile.close()

    # plot the slit function
    if plot_slit:
        fig = plt.figure(figsize=(5,9), dpi=150)
        ax  = fig.add_axes([0.13, 0.07, 0.81, 0.90])
        for ix in np.arange(slit_array.shape[1]):
            ax.plot(xnodes, slit_array[:,ix] + ix*0.15, '-', color='C0')
            ax.text(2.5, 0.03+ix*0.15, 'X=%d'%(x_lst[ix]), fontsize=10)
        ax.set_xlim(xnodes[0], xnodes[-1])
        ax.set_xlabel('$\sigma$', fontsize=16)
        ax.set_ylabel('Intensity', fontsize=16)
        fig.savefig(fig_slit)
        plt.close(fig)

    # plot the fitting list as a parameter map
    fig3 = plt.figure(figsize=(8,6), dpi=150)
    ax31 = fig3.add_subplot(221)
    ax32 = fig3.add_subplot(222)
    ax33 = fig3.add_subplot(223)
    ax34 = fig3.add_subplot(224)
    for ipara, para in enumerate(['A','fwhm','bkg']):
        _x_lst, _y_lst, _z_lst = [], [], []
        for key, v in fitting_lst[para].items():
            _x_lst.append(key[1])
            _y_lst.append(fitting_lst['c'][key])
            _z_lst.append(v)
        ax = fig3.get_axes()[ipara]
        ax.scatter(_x_lst, _y_lst, c=_z_lst, cmap='jet', lw=0, s=15)
        ax.set_xlim(0, w-1)
        ax.set_ylim(0, h-1)

    #####second step, scan each column and find the fitting parameters##########
    # construct slit functions using cubic spline interpolation for all columns
    full_slit_array = np.zeros((xnodes.size, w))
    for ix in np.arange(xnodes.size):
        f = intp.InterpolatedUnivariateSpline(x_lst, slit_array[ix, :], k=3)
        full_slit_array[ix, :] = f(np.arange(w))

    maxlst = full_slit_array.max(axis=0)
    maxilst = full_slit_array.argmax(axis=0)
    maxmask = full_slit_array>0.10*maxlst
    corr_mask_array = []
    for x in newx:
        ilst = np.nonzero(maxmask[:,x])[0]
        il = xnodes[ilst[0]]
        ir = xnodes[ilst[-1]]
        corr_mask_array.append((il, ir))

    interf_lst = []
    for x in np.arange(w):
        slitfunc = full_slit_array[:, x]
        nslit = slitfunc.size
        interf = intp.InterpolatedUnivariateSpline(
                    #np.arange(nslit)-nslit//2, slitfunc, k=3, ext=1)
                    xnodes, slitfunc, k=3, ext=1)
        interf_lst.append(interf)

    flatdata = np.ones_like(data, dtype=np.float64)

    fitpar_lst = {}

    # prepare a x list
    newx_lst = np.arange(0, w-1, 10)
    if newx_lst[-1] != w-1:
        newx_lst = np.append(newx_lst, w-1)

    for iaper, aper in enumerate(sorted(apertureset.keys())):
        fitpar_lst[aper] = []
        aperpar_lst = []

        position = positions[aper]
        lbound, ubound = bounds[aper]

        t1 = time.time()
        prev_p = None

        is_first_correct = False
        break_aperture = False

        for x in newx_lst:
            pos = position[x]
            y1 = int(max(0, lbound[x]))
            y2 = int(min(h, ubound[x]))
            xdata = np.arange(y1,y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(h, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))

            if 0 < pos < h and _satmask.sum()<3 and _badmask.sum()<3 and \
                sn > q_threshold:
                interf = interf_lst[x]
                if prev_p is None:
                    p0 = [ydata.max()-ydata.min(), 0.3, pos, max(0,ydata.min())]
                else:
                    p0 = [ydata.max()-ydata.min(), abs(prev_p[1]), pos, max(0,ydata.min())]

                #_m = np.ones_like(xdata, dtype=np.bool)
                _m = (~_satmask)*(~_badmask)
                for ite in range(10):
                    p, ier = opt.leastsq(errfunc2, p0,
                                args=(xdata[_m], ydata[_m], interf))
                    ydata_fit = fitfunc2(p, xdata, interf)
                    ydata_res = ydata - ydata_fit
                    std = ydata_res[_m].std(ddof=1)
                    _new_m = (np.abs(ydata_res) < 5*std)*_m
                    if _new_m.sum() == _m.sum():
                        break
                    _m = _new_m
                snr = p[0]/std

                # p[0]: amplitude; p[1]: k; p[2]: pos, p[3]:background
                succ = p[0]>0 and p[1]>0 and p[1]<1 and p[2]>y1 and p[2]<y2 and snr>5 and ier<=4
                prev_p = (None, p)[succ]

                if succ:
                    if not is_first_correct:
                        is_first_correct = True
                        if x > 0.25*w:
                            break_aperture = True
                            break
                    fitpar_lst[aper].append(p)
                else:
                    fitpar_lst[aper].append(np.array([np.NaN, np.NaN, np.NaN, np.NaN]))

            else:
                fitpar_lst[aper].append(np.array([np.NaN, np.NaN, np.NaN, np.NaN]))

        if break_aperture:
            break

        fitpar_lst[aper] = np.array(fitpar_lst[aper])

        if np.isnan(fitpar_lst[aper][:,0]).sum()>0.5*w:
            break

        if plot_aperpar:
            if iaper%5==0:
                fig = plt.figure(figsize=(12,8),dpi=150, tight_layout=True)
            ax1 = fig.add_subplot(5,4,1+iaper%5*4)
            ax2 = fig.add_subplot(5,4,2+iaper%5*4)
            ax3 = fig.add_subplot(5,4,3+iaper%5*4)
            ax4 = fig.add_subplot(5,4,4+iaper%5*4)
            #ax5 = fig.add_subplot(235)
            #ax6 = fig.add_subplot(236)

        mask_lst = []
        for ipara in range(4):
            # A, k, c, bkg
            yflux = fitpar_lst[aper][:,ipara]

            _m = ~np.isnan(yflux)
            if _m.sum() > 0:
                nonzeroindex = np.nonzero(_m)[0]
                i1, i2 = nonzeroindex[0], nonzeroindex[-1]+1
                for ite in range(2):
                    coeff = np.polyfit(newx_lst[_m]/w, yflux[_m], deg=param_deg)
                    yfit = np.polyval(coeff, newx_lst/w)
                    yres = yflux - yfit
                    std = yres[_m].std(ddof=1)
                    # replace np.nan by np.inf to avoid runtime warning
                    yres[~_m] = np.inf
                    _new_m = _m*(np.abs(yres)<5*std)
                    if _new_m.sum()==_m.sum():
                        break
                    _m = _new_m

            mask_lst.append(_m)

            aperpar_lst.append(coeff)

            if plot_aperpar:
                ax = fig.get_axes()[iaper%5*4+ipara]
                ax.plot(newx_lst, yflux, 'b-', lw=0.5)
                ax.plot(newx_lst[i1:i2], yfit[i1:i2], 'r-', lw=0.5)
                ax.plot(newx_lst[~_m],yflux[~_m],'ro',lw=0.5, ms=3, alpha=0.5)
                _y1, _y2 = ax.get_ylim()
                #ax.set_ylim(_y1, _y2)
                if ipara == 0:
                    ax.text(0.05*w, 0.15*_y1+0.85*_y2,'Aperture %d'%aper)
                ax.set_xlim(0, w-1)

        mask_lst = np.array(mask_lst)
        apermask = mask_lst.sum(axis=0)>0
        #ax6.plot(newx_lst[i1:i2],apermask[i1:i2], 'b-',lw=0.5)
        #ax6.set_xlim(0, w-1)
        #ax6.set_ylim(-0.5, 1.5)

        if plot_aperpar:
            for ax in fig.get_axes():
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                if w<3000:
                    ax.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
                else:
                    ax.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax.xaxis.set_minor_locator(tck.MultipleLocator(500))
                    
            #ax1.set_ylabel('A',fontsize=15)
            #ax2.set_ylabel('k',fontsize=15)
            #ax3.set_ylabel('center',fontsize=15)
            #ax4.set_ylabel('background',fontsize=15)
            #ax3.set_xlabel('x',fontsize=15)
            #ax4.set_xlabel('x',fontsize=15)
            #fig.suptitle('Aperture %d'%aper)
            if iaper%5==4 or iaper==len(apertureset)-1:
                fig.savefig(fig_aperpar%aper)
                plt.close(fig)

        for x in newx:
            interf = interf_lst[x]
            pos = position[x]
            y1 = int(max(0, lbound[x]))
            y2 = int(min(h, ubound[x]))
            if (y2-y1)<5:
                continue
            xdata = np.arange(y1,y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(h, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))

            if sn>q_threshold and _satmask.sum()<3 and _badmask.sum()<3:
                coeff_A, coeff_k, coeff_c, coeff_bkg = aperpar_lst

                A   = np.polyval(coeff_A, x/w)
                k   = np.polyval(coeff_k, x/w)
                c   = np.polyval(coeff_c, x/w)
                bkg = np.polyval(coeff_bkg, x/w)
                                                                  
                lcorr, rcorr = corr_mask_array[x]
                normx = (xdata-c)*k
                corr_mask = (normx > lcorr)*(normx < rcorr)
                flat = ydata/fitfunc2([A,k,c,bkg], xdata, interf)
                flatmask = corr_mask*~_satmask*~_badmask
                flatdata[y1:y2, x][flatmask] = flat[flatmask]

                #if aper==20:
                #    fig= plt.figure(dpi=150)
                #    ax1 = fig.add_subplot(211)
                #    ax2 = fig.add_subplot(212)
                #    ax1.plot(xdata, ydata, 'ko')
                #    ax2.plot(xdata, flat, 'r-', alpha=0.5)
                #    ax2.plot(xdata[flatmask], flat[flatmask], 'r-')
                #    ax2.axhline(y=1, color='k', ls='--')
                #    ax1.axvline(x=c, color='k', ls='--')
                #    ax1.axvline(x=c+1/k, color='k', ls=':')
                #    ax1.axvline(x=c-1/k, color='k', ls=':')
                #    ax2.axvline(x=c, color='k', ls='--')
                #    newxx=np.arange(y1, y2, 0.1)
                #    ax1.plot(newxx, fitfunc([A,k,c,bkg], newxx, interf), 'r-')
                #    ax1.set_xlim(xdata[0], xdata[-1])
                #    ax2.set_xlim(xdata[0], xdata[-1])
                #    fig.savefig('img/flat/new_%02d_%04d.png'%(aper,x))
                #    plt.close(fig)
            
        t2 = time.time()
        print('Aper %2d t = %6.1f ms'%(aper, (t2-t1)*1e3))

    return flatdata
