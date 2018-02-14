import os
import math
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

import matplotlib.pyplot as plt

from ..ccdproc import save_fits, array_to_table, table_to_array
from ..utils.onedarray import pairwise

def mosaic_flat_interact(filename_lst, outfile, mosaic_file, reg_file,
    disp_axis=0, mask_surfix = '_msk'):
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
        mask_surfix (string): Surfix of the filenames of masks.
    Returns:
        No returns.

    See Also:
        :func:`mosaic_flat_auto`

    '''
    import matplotlib.pyplot as plt

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
        mask_filename = '%s%s.fits'%(filename[0:-5],mask_surfix)
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
    save_fits(outfile,flat,header)
    outfile_mask = '%s%s.fits'%(outfile[0:-5],mask_surfix)
    mtable = array_to_table(flat_mask)
    save_fits(outfile_mask, mtable)

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
    from scipy.interpolate import InterpolatedUnivariateSpline
    from scipy.optimize    import minimize

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
        f = InterpolatedUnivariateSpline(np.arange(data2.size),data2,k=3)
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
        result = minimize(f2, 0.0, method='BFGS')
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
    from ..echelle.trace import ApertureSet


    all_aperloc_lst = []
    # all_aperloc_lst  = [
    #  [tracename1: aper_loc, tracename2: aper_loc],
    #  [tracename1: aper_loc, tracename2: aper_loc],
    #  [tracename1: aper_loc, tracename2: aper_loc],
    # ]

    tracename_lst = []

    for itrace, (tracename, aperset) in enumerate(aperture_set_lst.items()):
        print(tracename, len(aperset))

        # add tracename ot tracename list
        if tracename not in tracename_lst:
            tracename_lst.append(tracename)

        for o in aperset:
            aper_loc = aperset[o]
            if itrace == 0:
                # append all the apertures in the first trace file into the
                # aperloc list
                all_aperloc_lst.append({tracename: aper_loc})
            else:
                insert = False
                for ilist, list1 in enumerate(all_aperloc_lst):
                    # one aperture should no contain more than 1 apertures
                    # from the same trace file.
                    if tracename in list1:
                        continue
                    # calculate the relative distances.
                    for _tracename, _aperloc in list1.items():
                        distance = aper_loc.get_distance(_aperloc)
                        if abs(distance)<3:
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
    save_fits(outfile, mos_flatdata)

    return mosaic_aperset

def mosaic_image(data_lst, head_lst, outfile, coeff_lst, disp_axis):
    mos_data = np.zeros(shape)

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

