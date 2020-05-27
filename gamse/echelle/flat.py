import os
import time
import math
import logging
logger = logging.getLogger(__name__)

import numpy as np
import numpy.polynomial as poly
import astropy.io.fits as fits
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.signal as sg
from scipy.integrate import simps

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from ..utils.onedarray  import pairwise, get_local_minima
from ..utils.regression import iterative_polyfit
from .imageproc         import array_to_table, table_to_array
from .trace             import ApertureSet

def mosaic_flat_interact(filename_lst, outfile, mosaic_file, reg_file,
    disp_axis=0, mask_suffix = '_msk'):
    """Display an interacitve GUI to mosaic the flat images.

    Args:
        filename_lst (list): A list containing filenames of flat images.
        outfile (str): Filename of the output image.
        mosaic_file (str): Name of the ascii file recording the coefficients
            of the mosaic boundaries.
        reg_file (str): Name of the `.reg` file to be displayed in SAO-DS9.
        disp_axis (int): Main dispersion axis of the input image. 0 means
            the echelle orders are along the *y* axis. 1 means along *x* axis.
        mask_suffix (str): Surfix of the filenames of masks.

    Returns:
        No returns.

    See Also:
        :func:`mosaic_flat_auto`

    """

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
        """Select an area when clicking on the figure.
        """
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
        """Add or remove a boundary when pressing 'a' or 'd' on the keyboard.
        """
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
    """Detect the curve of gap between two orders along Y-axis.

    This is realized by calculating the cross-correlation function and detect
    the position of the maximum value.

    Args:
        data (2-d :class:`numpy.ndarray`): Data image as a numpy 2d-array.
        x0 (int): Starting coordinate.
        ccf_ulimit (int): Upper limit to **x0** of the data segment used to
            calculate CCF.
        ccf_llimit (int): Lower limit to **x0** of the data segment used to
            calculate CCF.
        step (int): Step of searching the gap.
        order (int): Degree of polynomial used to fit the boundary.

    Returns:
        :class:`numpy.ndarray`: Cofficients of the polynomial.

    """

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
    """Read mosaic boundary information from an existing ASCII file.

    Args:
        filename (str): Name of the mosaic file.

    Returns:
        tuple: A tuple containing:

            * **coeff_lst** (*list*): A list containing coefficients of
              polynomials for the boundaries.
            * **select_area** (*dict*): A dict containing the selected areas.

    """
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


def mosaic_flat_auto(aperture_set_lst, max_count, name_lst):
    """Mosaic flat images automatically.

    Args:
        aperture_set_lst (list): Dict of
            :class:`~gamse.echelle.trace.ApertureSet`.
        max_count (float): Maximum count.
        name_lst (list): The sorted name list.

    Returns:
        :class:`~gamse.echelle.trace.ApertureSet`: The mosaiced aperture set.

    See Also:

        * :func:`mosaic_flat_interact`
        * :func:`mosaic_images`

    """

    # find the brightness order of all aperture sets

    '''
    name_satcount_lst = {}
    for name, aperset in aperture_set_lst.items():
        sat_count = 0
        for aper, aper_loc in aperset.items():
            if aper_loc.nsat > 0:
                sat_count += 1
        name_satcount_lst[name] = sat_count

    tmp_lst = sorted(name_satcount_lst, key=name_satcount_lst.get)
    ref_name, ref_aperset = tmp_lst[0]
    '''


    all_aperloc_lst = []
    # all_aperloc_lst  = [
    #  [name1: (aper4, aper_loc4), name2: (aper4, aper_loc4)],
    #  [name1: (aper5, aper_loc5), name2: (aper5, aper_loc5)],
    #  [name1: (aper6, aper_loc6), name2: (aper6, aper_loc6)],
    # ]

    for iaperset, name in enumerate(name_lst):
        aperset = aperture_set_lst[name]

        for aper, aper_loc in aperset.items():

            ## get local separations
            loc_sep = aperset.get_local_separation(aper)

            if iaperset == 0:
                # append all the apertures in the first trace file into the
                # aperloc list
                all_aperloc_lst.append({name: (aper, aper_loc)})
            else:
                # first we have to check if this aperture is belong to any
                # existing aperture in the all_aperloc_lst
                # this is done by calculating the distance of this aperture
                # to all existing apertures in teh all_aperloc_lst
                poly = aper_loc.position
                npoints = poly.domain[1] - poly.domain[0]
                if aper_loc.nsat/npoints > 0.7:
                    continue
                insert = False
                for iaperloc_lst, aperloc_lst in enumerate(all_aperloc_lst):
                    # one aperture should not contain more than 1 apertures
                    # from the same trace file.
                    if name in aperloc_lst:
                        continue
                    # calculate the relative distances.
                    for _name, (_aper, _aperloc) in aperloc_lst.items():
                        distance = aper_loc.get_distance(_aperloc)
                        if abs(distance) < 0.5*loc_sep:
                            # append this aperture to an existing aperture
                            all_aperloc_lst[iaperloc_lst][name] = (aper, aper_loc)
                            insert = True
                            break
                    # if already added to an existing aperture, skip the rest
                    # apertures
                    if insert:
                        break

                # if this aperture does not belong to any existing aperture,
                # append it as a new aperture
                if not insert:
                    all_aperloc_lst.append({name: (aper, aper_loc)})


    # prepare the information written to running log
    message = ['Aperture Information for Different Flat Files:']
    subtitle = 'center,  N (sat),  max'
    msg1 = [name.center(len(subtitle)) for name in name_lst]
    msg2 = [subtitle for name in name_lst]
    message.append('| '+(' | '.join(msg1))+' |')
    message.append('| '+(' | '.join(msg2))+' |')


    mosaic_aperset = ApertureSet()
    for list1 in all_aperloc_lst:
        # add information to running log
        msg = []
        for name in name_lst:
            if name in list1:
                aper, aper_loc = list1[name]
                msg.append('{:3d} {:6.1f} {:4d} {:10.1f}'.format(
                    aper, aper_loc.get_center(), aper_loc.nsat, aper_loc.max))
            else:
                msg.append(' '*26)
        message.append('| '+(' | '.join(msg))+' |')

        # pick up the best trace file for each aperture
        nosat_lst = {name: (aper, aper_loc)
                        for name, (aper, aper_loc) in list1.items()
                        if aper_loc.nsat == 0 and aper_loc.max < max_count}

        if len(nosat_lst)>0:
            # if there are apertures without saturated pixels, find the one
            # with largest median values
            nosat_sort_lst = sorted(nosat_lst.items(),
                                key=lambda item: item[1][1].median)
            pick_name, (pick_aper, pick_aperloc) = nosat_sort_lst[-1]
        else:
            # all apertures are saturated. Then find the aperture that has
            # the least number of saturated pixels.
            sat_sort_lst = sorted(list1.items(),
                                key=lambda item: item[1][1].nsat)
            pick_name, (pick_aper, pick_aperloc) = sat_sort_lst[0]

        # give a new attribute called "tracename"
        setattr(pick_aperloc, 'tracename', pick_name)
        # give a new attribute called "ori_aper"
        setattr(pick_aperloc, 'ori_aper', pick_aper)
        # add this aperloc to mosaic_aperset
        mosaic_aperset.add_aperture(pick_aperloc)

    logger.info((os.linesep+' '*3).join(message))

    # resort all the aperloc
    mosaic_aperset.sort()

    # make a summary and write it to log
    message = ['Flat Mosaic Information',
                'aper, yposition, flatname, N (sat), Max (count)']
    for aper, aper_loc in sorted(mosaic_aperset.items()):
        message.append('{:4d} {:7.2f} {:^15s} {:4d} {:10.1f}'.format(
            aper, aper_loc.get_center(), aper_loc.tracename, aper_loc.nsat,
            aper_loc.max))
    logger.info((os.linesep+' '*3).join(message))

    # check the shape consistency of mosaic_aperset
    # get the shape of the first aperloc in mosaic_aperset
    shape = list(mosaic_aperset.values())[0].shape

    for aper, aper_loc in sorted(mosaic_aperset.items()):
        if aper_loc.shape != shape:
            logger.error(
                'Shape of Aper %d (%d, %d) does not match the shape (%d, %d)'%(
                aper, aper_loc.shape[0], aper_loc.shape[1], shape[0], shape[1])
            )

    return mosaic_aperset


def mosaic_images(image_lst, mosaic_aperset):
    """Mosaic input images with the identifications in the input aperture set.

    Args:
        images_lst (dict): A dict containing {*names*: *images*}, where *names*
            are strings, and *images* are the corresponding 2d
            :class:`numpy.ndarray` objects.
        mosaic_aperset (:class:`~gamse.echelle.trace.ApertureSet`): The mosaiced
            aperture set.

    Returns:
        :class:`numpy.ndarray`: The final mosaiced image with the shapes and
            datatypes as images in **images_lst**.

    See Also:
        * :func:`mosaic_speclist`
        * :func:`mosaic_flat_auto`

    """

    # get the shape of the first aperloc in mosaic_aperset
    shape = list(mosaic_aperset.values())[0].shape

    # check the shape consistency of images
    for name, image in sorted(image_lst.items()):
        if image.shape != shape:
            logger.error(
                'Image shape of %s (%d, %d) does not match the shape (%d, %d)'%(
                name, image.shape[0], image.shape[1], shape[0], shape[1])
            )

    # get mosaic mask for each tracename
    maskdata_lst = {name: np.zeros(shape, dtype=np.bool)
                    for name in image_lst.keys()}

    h, w = shape
    yy, xx = np.mgrid[:h:, :w:]
    xlst = np.arange(w)
    ylst = np.arange(h)

    for iaper, (aper, aper_loc) in enumerate(sorted(mosaic_aperset.items())):
        tracename = aper_loc.tracename
        if iaper == 0:
            maskdata_lst[tracename][:,:] = True
        elif tracename != prev_tracename:
            prev_aper_loc = mosaic_aperset[iaper-1]
            if aper_loc.direct == 0:
                # aperture along Y axis
                center_line = aper_loc.position(ylst)
                prev_center_line = prev_aper_loc.position(ylst)
                cut_bound = (center_line + prev_center_line)/2.
                m = xx > np.round(cut_bound)
            elif aper_loc.direct == 1:
                # aperture along X axis
                center_line = aper_loc.position(xlst)
                prev_center_line = prev_aper_loc.position(xlst)
                cut_bound = (center_line + prev_center_line)/2.
                m = yy > np.round(cut_bound)
            maskdata_lst[prev_tracename][m] = False
            maskdata_lst[tracename][m] = True
        else:
            pass
        prev_tracename = tracename

    dtype = list(image_lst.values())[0].dtype
    # finally mosaic the images
    mosaic_image = np.zeros(shape, dtype=dtype)
    for _name, _maskdata in sorted(maskdata_lst.items()):
        _image = image_lst[_name]
        # filter out NaN values. otherwise the NaN will be passed to final image
        _image[np.isnan(_image)] = 0
        mosaic_image += _image*_maskdata

    return mosaic_image

def mosaic_spec(spec_lst, mosaic_aperset):
    """Mosaic input spectra list with the identifications in the input aperture
    set.

    Args:
        spec_lst (dict): A dict containing {*names*: *spec*}
        mosaic_aperset (:class:`~gamse.echelle.trace.ApertureSet`): The mosaiced
            aperture set.
    
    Returns:

    See Also:
        * :func:`mosaic_images`
        * :func:`mosaic_flat_auto`
    """

    spec = []

    for iaper, (aper, aper_loc) in enumerate(sorted(mosaic_aperset.items())):
        tracename = aper_loc.tracename
        ori_aper = aper_loc.ori_aper
        spec1 = spec_lst[tracename]
        mask = spec1['aperture'] == ori_aper
        row = spec1[mask][0]
        # udpate aperture number
        row['aperture'] = aper
        spec.append(tuple(row))

    spec = np.array(spec, dtype=spec1.dtype)

    return spec


def save_mosaic_reg(filename, coeff_lst, disp_axis, shape, npoints=20):
    """Save boundaries data in a SAO-DS9 region file.

    Args:
        filename (str): Filename of the output region file.
        coeff_lst (list): List of coefficients.
        disp_axis (int): 0 or 1, depending on the dispersion axis.
        shape (tuple): A tuple containing the shape of the image.
        npoints (int): Number of sampling points.
    """
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

def default_smooth_aperpar_A(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *A* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`): Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`): Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`): Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`): Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`get_fiber_flat`
        * :func:`default_smooth_aperpar_k`
        * :func:`default_smooth_aperpar_c`
        * :func:`default_smooth_aperpar_bkg`
    
    """
    has_fringe_lst = []
    aperpar = np.array([np.nan]*npoints)
    xpiece_lst     = np.array([np.nan]*newx_lst.size)
    ypiece_res_lst = np.array([np.nan]*newx_lst.size)
    mask_rej_lst   = np.array([np.nan]*newx_lst.size)
    allx = np.arange(npoints)
    # the dtype of xpiece_lst and ypiece_lst is np.float64

    # first try, scan every segment. find fringe by checking the local maximum
    # points after smoothing. Meanwhile, save the smoothing results in case the
    # smoothing will be used afterwards.
    for group in group_lst:
        i1, i2 = group[0], group[-1]
        p1, p2 = newx_lst[i1], newx_lst[i2]
        m = fitmask[group]
        xpiece = newx_lst[group]
        ypiece = ypara[group]
        # now fill the NaN values in ypiece
        if (~m).sum() > 0:
            f = intp.InterpolatedUnivariateSpline(xpiece[m], ypiece[m], k=3)
            ypiece = f(xpiece)
        # now xpiece and ypiece are ready

        _m = np.ones_like(ypiece, dtype=np.bool)
        for ite in range(3):
            f = intp.InterpolatedUnivariateSpline(xpiece[_m], ypiece[_m], k=3)
            ypiece2 = f(xpiece)
            win_len = (11, 21)[ypiece2.size>23]
            ysmooth = sg.savgol_filter(ypiece2, window_length=win_len, polyorder=3)
            res = ypiece - ysmooth
            std = res.std()
            _new_m = np.abs(res) < 3*std

            # prevent extrapolation at the boundaries
            if _new_m.size > 3:
                _new_m[0:3] = True
                _new_m[-3:] = True
            _new_m = _m*_new_m

            if _new_m.sum() == _m.sum():
                break
            _m = _new_m
        # now xpiece, ypiece, ypiece2, ysmooth, res, and _m have the same
        # lengths and meanings on their positions of elements

        f = intp.InterpolatedUnivariateSpline(xpiece, ysmooth, k=3)
        _x = np.arange(p1, p2+1)

        aperpar[_x] = f(_x)
        xpiece_lst[group] = xpiece
        ypiece_res_lst[group] = res
        mask_rej_lst[group] = ~_m

        # find out if this order is affected by fringes, by checking the
        # distribution of local maximum points
        imax, ymax = get_local_minima(-ysmooth, window=5)
        if len(imax) > 0:
            x = xpiece[imax]
        else:
            x = []
        # determine how many pixels in each bin.
        # if w=4000, then 500 pix. if w=2000, then 250 pix.
        npixbin = w//8
        bins = np.linspace(p1, p2, int(p2-p1)//npixbin+2)
        hist, _ = np.histogram(x, bins)

        n_nonzerobins = np.nonzero(hist)[0].size
        n_zerobins = hist.size - n_nonzerobins

        if p2-p1 < npoints/8 or n_zerobins <= 1 or \
            n_zerobins < n_nonzerobins or n_nonzerobins >= 3:
            # there's fringe
            has_fringe = True
        else:
            # no fringe
            has_fringe = False
        has_fringe_lst.append(has_fringe)

    # use global polynomial fitting if this order is affected by fringe and the
    # following conditions are satisified
    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2 \
        and has_fringe_lst.count(True) == len(has_fringe_lst):
        # fit polynomial over the whole order

        # prepare xpiece and y piece
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # fit with polynomial
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)
        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst = ~_m
    else:
        # scan again
        # fit polynomial for every segment
        for group, has_fringe in zip(group_lst, has_fringe_lst):
            xpiece = newx_lst[group]
            ypiece = ypara[group]
            xspan = xpiece[-1] - xpiece[0]
            if has_fringe:
                if   xspan > npoints/2: deg = 4
                elif xspan > npoints/4: deg = 3
                elif xspan > npoints/8: deg = 2
                else:                   deg = 1
            else:
                deg = 7
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, np.log(ypiece), deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)
            ypiece_fit = np.exp(ypiece_fit)
            ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.exp(np.polyval(coeff, ii/npoints))
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst


def default_smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *k* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`): Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`): Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`): Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`): Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`get_fiber_flat`
        * :func:`default_smooth_aperpar_A`
        * :func:`default_smooth_aperpar_c`
        * :func:`default_smooth_aperpar_bkg`
    
    """

    allx = np.arange(npoints)

    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2:

        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the polynomial degree
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # fit with polynomial
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*w)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]

            # determine the polynomial degree
            xspan = xpiece[-1] - xpiece[0]
            if   xspan > npoints/2: deg = 4
            elif xspan > npoints/4: deg = 3
            elif xspan > npoints/8: deg = 2
            else:                   deg = 1

            # fit with polynomial
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, ypiece, deg=deg, maxiter=10, lower_clip=3,
                upper_clip=3)

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/npoints)
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def default_smooth_aperpar_c(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *c* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`): Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`): Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`): Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`): Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`get_fiber_flat`
        * :func:`default_smooth_aperpar_A`
        * :func:`default_smooth_aperpar_k`
        * :func:`default_smooth_aperpar_bkg`
    
    """

    return default_smooth_aperpar_k(newx_lst, ypara, fitmask, group_lst, npoints)

def default_smooth_aperpar_bkg(newx_lst, ypara, fitmask, group_lst, npoints):
    """Smooth *bkg* of the four 2D profile parameters (*A*, *k*, *c*, *bkg*) of
    the fiber flat-fielding.

    Args:
        newx_lst (:class:`numpy.ndarray`): Sampling pixels of the 2D profile.
        ypara (:class:`numpy.ndarray`): Array of *A* at the sampling pixels.
        fitmask (:class:`numpy.ndarray`): Mask array of **ypara**.
        group_lst (list): Groups of (*x*:sub:`1`, *x*:sub:`2`, ... *x*:sub:`N`)
            in each segment, where *x*:sub:`i` are indices in **newx_lst**.
        npoints (int): Number of points in this order.

    Returns:
        tuple: A tuple containing:

            * **aperpar** (:class:`numpy.ndarray`): Reconstructed profile
              paramters at all pixels.
            * **xpiece_lst** (:class:`numpy.ndarray`): Reconstructed profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **ypiece_res_lst** (:class:`numpy.ndarray`): Residuals of profile
              parameters at sampling pixels in **newx_lst** for plotting.
            * **mask_rej_lst** (:class:`numpy.ndarray`): Mask of sampling pixels
              in **newx_lst** participating in fitting or smoothing.

    See Also:

        * :func:`get_fiber_flat`
        * :func:`default_smooth_aperpar_A`
        * :func:`default_smooth_aperpar_k`
        * :func:`default_smooth_aperpar_c`
    
    """

    allx = np.arange(npoints)

    # fit for bkg
    if len(group_lst) > 1 \
        and newx_lst[group_lst[0][0]] < npoints/2 \
        and newx_lst[group_lst[-1][-1]] > npoints/2:
        # fit polynomial over the whole order
        xpiece = np.concatenate([newx_lst[group] for group in group_lst])
        ypiece = np.concatenate([ypara[group] for group in group_lst])

        # determine the degree of polynomial
        xspan = xpiece[-1] - xpiece[0]
        if   xspan > npoints/2: deg = 4
        elif xspan > npoints/4: deg = 3
        elif xspan > npoints/8: deg = 2
        else:                   deg = 1

        # polynomial fitting
        coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
            xpiece/npoints, ypiece, deg=deg, maxiter=10,
            lower_clip=3, upper_clip=3)

        aperpar = np.polyval(coeff, allx/npoints)
        xpiece_lst     = xpiece
        ypiece_res_lst = ypiece_res
        mask_rej_lst   = ~_m
    else:
        # fit polynomial for every segment
        aperpar = np.array([np.nan]*npoints)
        xpiece_lst     = np.array([np.nan]*newx_lst.size)
        ypiece_res_lst = np.array([np.nan]*newx_lst.size)
        mask_rej_lst   = np.array([np.nan]*newx_lst.size)
        for group in group_lst:
            xpiece = newx_lst[group]
            ypiece = ypara[group]

            # determine the degree of polynomial
            xspan = xpiece[-1] - xpiece[0]
            if   xspan > npoints/2: deg = 7
            elif xspan > npoints/4: deg = 3
            elif xspan > npoints/8: deg = 2
            else:                   deg = 1

            scale = ('linear','log')[(ypiece<=0).sum()==0]
            if scale=='log':
                ypiece = np.log(ypiece)

            # polynomial fitting
            coeff, ypiece_fit, ypiece_res, _m, std = iterative_polyfit(
                xpiece/npoints, ypiece, deg=deg, maxiter=10,
                lower_clip=3, upper_clip=3)

            if scale=='log':
                ypiece = np.exp(ypiece)
                ypiece_fit = np.exp(ypiece_fit)
                ypiece_res = ypiece - ypiece_fit

            ii = np.arange(xpiece[0], xpiece[-1]+1)
            aperpar[ii] = np.polyval(coeff, ii/npoints)
            if scale=='log':
                aperpar[ii] = np.exp(aperpar[ii])
            xpiece_lst[group]     = xpiece
            ypiece_res_lst[group] = ypiece_res
            mask_rej_lst[group]   = ~_m

    return aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst

def get_fiber_flat(data, mask, apertureset, nflat, slit_step=64,
        q_threshold=30,
        smooth_A_func=default_smooth_aperpar_A,
        smooth_k_func=default_smooth_aperpar_k,
        smooth_c_func=default_smooth_aperpar_c,
        smooth_bkg_func=default_smooth_aperpar_bkg,
        fig_aperpar=None, fig_overlap=None,
        fig_slit=None, slit_file=None,
    ):
    """Get the flat fielding image from the input file.

    Args:
        data (:class:`numpy.ndarray`): Image data of flat fielding.
        mask (:class:`numpy.ndarray`): Mask data of flat fielding.
        apertureset (:class:`~gamse.echelle.trace.ApertureSet`): Echelle
            apertures detected in the input file.
        nflat (int): Number of flat fielding frames combined.
        slit_step (int): Step of slit scanning.
        q_threshold (float): Threshold of *Q*-factor.
        smooth_A_func (func): Function of smoothing the aperture parameter A.
        smooth_k_func (func): Function of smoothing the aperture parameter k.
        smooth_c_func (func): Function of smoothing the aperture parameter c.
        smooth_bkg_func (func): Function of smoothing the aperture parameter bkg.
        fig_aperpar (str): Path to the image of aperture profile parameters.
        fig_overlap (str): Path to the image of overlapped slit profiles.
        fig_slit (str): Path to the image of slit functions.
        slit_file (str): Path to the ASCII file of slit functions.

    Returns:
        tuple: A tuple containing:

            * :class:`numpy.ndarray`: 2D response map.
            * :class:`numpy.ndarray`: A dict of flat 1-d spectra.

    """
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
        return ydata - fitfunc2(p, xdata, interf)

    h, w = data.shape


    # find saturation mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)

    # find the central positions and boundaries for each aperture
    allx = np.arange(w)
    positions = apertureset.get_positions(allx)
    bounds = apertureset.get_boundaries(allx)

    plot_overlap = (fig_overlap is not None)
    plot_aperpar = (fig_aperpar is not None)
    plot_slit    = (fig_slit is not None)
    plot_single  = False
    fig_fitting = 'flat_single_%04d_%02d.png'
    plot_fitting = True

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
                        _newx = np.arange(b1, b2+1e-3, 0.1)
                        axi.plot(_newx, fitfunc(p1, _newx), 'r-')
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
            _xnodes, _ynodes = [], [] # pre-initialize nodes list.
            for c in xnodes:
                _m1 = np.abs(all_x[_m]-c) < step/2
                if _m1.sum()>0:
                    _xnodes.append(c)
                    _ynodes.append(all_y[_m][_m1].mean(dtype=np.float64))

            # construct the real slit function list with interpolating the empty
            # values
            _xnodes = np.array(_xnodes)
            _ynodes = np.array(_ynodes)
            f0 = intp.InterpolatedUnivariateSpline(_xnodes, _ynodes, k=3, ext=3)
            ynodes = f0(xnodes)

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

        string = ', '.join(['{:d}'.format(x) for x in x_lst])
        slitoutfile.write('COLUMNS = ' + string + os.linesep)

        string = ', '.join(['{:5.2f}'.format(x) for x in xnodes])
        slitoutfile.write('NODES = ' + string + os.linesep)

        for col in np.arange(x_lst.size):
            for row in np.arange(xnodes.size):
                slitoutfile.write(' {:12.8f}'.format(slit_array[row, col]))
            slitoutfile.write(os.linesep)

        slitoutfile.close()

    # plot the slit function
    if plot_slit:
        fig = plt.figure(figsize=(5,10), dpi=150)
        ax  = fig.add_axes([0.16, 0.06, 0.81, 0.92])
        for ix in np.arange(slit_array.shape[1]):
            ax.plot(xnodes, slit_array[:,ix] + ix*0.15, '-', color='C0')
            ax.text(2.5, 0.03+ix*0.15, 'X=%d'%(x_lst[ix]), fontsize=13)
        ax.set_xlim(xnodes[0], xnodes[-1])
        _y1, _y2 = ax.get_ylim()
        # has to be removed after plotting
        #ax.text(xnodes[0]+0.5, 0.05*_y1+0.95*_y2, 'HRS (Xinglong)', fontsize=21)
        ax.set_xlabel('$\sigma$', fontsize=15)
        ax.set_ylabel('Intensity', fontsize=15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
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
        ax.scatter(_x_lst, _y_lst, c=_z_lst, cmap='jet', lw=0, s=15, alpha=0.6)
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
    for x in allx:
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

    # prepare an x list
    newx_lst = np.arange(0, w-1, 10)
    if newx_lst[-1] != w-1:
        newx_lst = np.append(newx_lst, w-1)

    ###################### loop for every aperture ########################
    # use to remember the status of unsaved aperpar_fig
    # = True if there's unsaved figure in memory
    has_aperpar_fig = False

    # initialize 1-d spectra array of flat
    flatspec_lst = {aper: np.full(w, np.nan) for aper in apertureset}

    for iaper, aper in enumerate(sorted(apertureset.keys())):
        fitpar_lst  = [] # stores (A, k, c, bkg).has the same length as newx_lst
        aperpar_lst = []

        # prepare the figure for plotting the parameters of each aperture
        if plot_aperpar:
            if iaper%5==0:
                fig = plt.figure(figsize=(15,8), dpi=150)
                ax_lst = {}

        # central positions of this aperture
        position = positions[aper]
        lbound, ubound = bounds[aper]

        t1 = time.time()
        prev_p = None

        is_first_correct = False
        break_aperture = False

        # loop for every newx. find the fitting parameters for each column
        # prepar the blank parameter for insert
        blank_p = np.array([np.NaN, np.NaN, np.NaN, np.NaN])

        for x in newx_lst:
            # central position
            pos = position[x]
            # skip this column if central position excess the CCD range
            if pos<0 or pos>h:
                fitpar_lst.append(blank_p)
                continue
            # lower and upper bounds
            y1 = int(max(0, lbound[x]))
            y2 = int(min(h, ubound[x]))
            # construct fitting data (xdata, ydata)
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]
            # calculate saturation mask and bad-pixel mask
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            # skip this column if too many saturated or bad pixels
            if _satmask.sum()>=3 or _badmask.sum()>=3:
                fitpar_lst.append(blank_p)
                continue
            # estimate the SNR
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(h, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))
            # skip this column if sn is too low
            if sn < q_threshold:
                fitpar_lst.append(blank_p)
                continue

            # begin fitting
            interf = interf_lst[x]
            if prev_p is None:
                p0 = [ydata.max()-ydata.min(), 0.3, pos, max(0,ydata.min())]
            else:
                p0 = [ydata.max()-ydata.min(), abs(prev_p[1]), pos, max(0,ydata.min())]

            # skip this column if lower or upper 1 sigma excess the CCD range
            if pos-1./p0[1]<0 or pos+1./p0[1]>h:
                fitpar_lst.append(blank_p)
                continue

            # find A, k, c, bkg
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
            succ = p[0]>0 and 0<p[1]<1 and y1<p[2]<y2 and snr>5 and ier<=4
            prev_p = (None, p)[succ]

            if succ:
                if not is_first_correct:
                    is_first_correct = True
                    if x > 0.25*w:
                        break_aperture = True
                        break
                fitpar_lst.append(p)
            else:
                fitpar_lst.append(blank_p)

        if break_aperture:
            message = ('Aperture {:3d}: Skipped because of '
                       'break_aperture=True').format(aper)
            logger.debug(message)
            print(message)
            continue

        fitpar_lst = np.array(fitpar_lst)

        if np.isnan(fitpar_lst[:,0]).sum()>0.5*w:
            message = ('Aperture {:3d}: Skipped because of too many NaN '
                       'values in aperture parameters').format(aper)
            logger.debug(message)
            print(message)
            continue

        if (~np.isnan(fitpar_lst[:,0])).sum()<10:
            message = ('Aperture {:3d}: Skipped because of too few real '
                       'values in aperture parameters').format(aper)
            logger.debug(message)
            print(message)
            continue

        # pick up NaN positions in fitpar_lst and generate fitmask.
        # NaN = False. Real number = True
        fitmask = ~np.isnan(fitpar_lst[:,0])
        # divide the whole order into several groups
        xx = np.nonzero(fitmask)[0]
        group_lst = np.split(xx, np.where(np.diff(xx) > 4)[0]+1)
        # group_lst is composed of (x1, x2, ..., xN), where xi is index in
        # **newx_lst**
        # 4 means the maximum tolerance skipping value in fitmask is 3
        # filter out short segments
        # every index in group is index in newx_lst, NOT real pixel numbers
        group_lst = [group for group in group_lst
                     if newx_lst[group[-1]] - newx_lst[group[0]] > w/10]

        if len(group_lst) == 0:
            message = ('Aperture {:3d}: Skipped'.format(aper))
            print(message)
            logger.debug(message)
            continue

        # loop for A, k, c, bkg. Smooth these parameters
        for ipara in range(4):
            ypara = fitpar_lst[:,ipara]

            if ipara == 0:
                # fit for A
                res = smooth_A_func(newx_lst, ypara, fitmask, group_lst, w)
            elif ipara == 1:
                # fit for k
                res = smooth_k_func(newx_lst, ypara, fitmask, group_lst, w)
            elif ipara == 2:
                # fit for c
                res = smooth_c_func(newx_lst, ypara, fitmask, group_lst, w)
            else:
                # fit for bkg
                res = smooth_bkg_func(newx_lst, ypara, fitmask, group_lst, w)

            # extract smoothing results
            aperpar, xpiece_lst, ypiece_res_lst, mask_rej_lst = res

            # pack this parameter for every pixels
            aperpar_lst.append(aperpar)


            if plot_aperpar:
                ########### plot flat parametres every 5 orders ##############
                has_aperpar_fig = True
                irow = iaper%5
                for icol in range(4):
                    _x, _y = 0.04+icol*0.24, (4-irow)*0.19+0.05
                    ax = fig.add_axes([_x, _y, 0.20, 0.17])
                    ax_lst[(irow, icol)] = ax
                i1, i2 = newx_lst[group_lst[0][0]], newx_lst[group_lst[-1][-1]]
                # plot the parameters
                ax1 = ax_lst[(iaper%5, ipara)]

                # make a copy of ax1 and plot the residuals in the background
                ax2 = ax1.twinx()
                ax2.plot(xpiece_lst, ypiece_res_lst, color='gray', lw=0.5,
                        alpha=0.4, zorder=-2)
                ax2.axhline(y=0, color='gray', ls='--', lw=0.5,
                        alpha=0.4, zorder=-3)
                # plot rejected points with gray dots
                _m = mask_rej_lst>0
                if _m.sum()>0:
                    ax2.plot(xpiece_lst[_m], ypiece_res_lst[_m], 'o',
                            color='gray', lw=0.5, ms=2, alpha=0.4, zorder=-1)

                # plot data points
                ax1.plot(newx_lst, ypara, '-', color='C0', lw=0.5, zorder=1)
                # plot fitted value
                ax1.plot(allx[i1:i2], aperpar[i1:i2], '-', color='C1',
                    lw=1, alpha=0.8, zorder=2)

                #ax1.plot(newx_lst[~fitmask], ypara[~fitmask], 'o', color='C3',
                #        lw=0.5, ms=3, alpha=0.5)
                _y1, _y2 = ax1.get_ylim()
                if ipara == 0:
                    ax1.text(0.05*w, 0.15*_y1+0.85*_y2, 'Aperture %d'%aper,
                            fontsize=10)
                ax1.text(0.9*w, 0.15*_y1+0.85*_y2, 'AKCB'[ipara], fontsize=10)

                # fill the fitting regions
                for group in group_lst:
                    i1, i2 = newx_lst[group[0]], newx_lst[group[-1]]
                    ax1.fill_betweenx([_y1, _y2], i1, i2, color='C0', alpha=0.1)

                ax1.set_xlim(0, w-1)
                ax1.set_ylim(_y1, _y2)
                if iaper%5<4:
                    ax1.set_xticklabels([])

                for tick in ax1.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in ax2.yaxis.get_major_ticks():
                    tick.label2.set_fontsize(4)
                    tick.label2.set_color('gray')
                    tick.label2.set_alpha(0.6)
                for tickline in ax2.yaxis.get_ticklines():
                    tickline.set_color('gray')
                    tickline.set_alpha(0.6)
                if w<3000:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(100))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(100))
                else:
                    ax1.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax1.xaxis.set_minor_locator(tck.MultipleLocator(500))
                    ax2.xaxis.set_major_locator(tck.MultipleLocator(1000))
                    ax2.xaxis.set_minor_locator(tck.MultipleLocator(500))
                
                ########### plot flat parametres for every order ##############
                if False:
                    if ipara == 0:
                        fig5 = plt.figure(figsize=(8,5), dpi=150)
                        axes5_lst = [
                            fig5.add_axes([0.08, 0.57, 0.36, 0.41]),
                            fig5.add_axes([0.56, 0.57, 0.36, 0.41]),
                            fig5.add_axes([0.08, 0.10, 0.36, 0.41]),
                            fig5.add_axes([0.56, 0.10, 0.36, 0.41]),
                        ]
                    i1, i2 = newx_lst[group_lst[0][0]], newx_lst[group_lst[-1][-1]]
                    ax51 = axes5_lst[ipara]
                    
                    # make a copy of ax1 and plot the residuals in the background
                    ax52 = ax51.twinx()
                    ax52.plot(xpiece_lst, ypiece_res_lst, color='gray', lw=0.5,
                                alpha=0.6, zorder=-2)
                    ax52.axhline(y=0, color='gray', ls='--', lw=0.5, alpha=0.6,
                                zorder=-3)
                    # plot rejected points with gray dots
                    _m = mask_rej_lst>0
                    if _m.sum()>0:
                        ax52.plot(xpiece_lst[_m], ypiece_res_lst[_m], 'o',
                                color='gray', lw=0.5, ms=2, alpha=0.6, zorder=-1)
                    # adjust ticks and labels for ax52
                    for tick in ax52.yaxis.get_major_ticks():
                        tick.label2.set_fontsize(10)
                        tick.label2.set_color('gray')
                        tick.label2.set_alpha(0.8)
                    for tickline in ax52.yaxis.get_ticklines():
                        tickline.set_color('gray')
                        tickline.set_alpha(0.8)

                    # plot data points in ax51
                    ax51.plot(newx_lst, ypara, '-', color='C0', lw=0.8,
                                alpha=1.0, zorder=1)
                    # plot fitted value
                    ax51.plot(allx[i1:i2], aperpar[i1:i2], '-', color='C1',
                                lw=1, alpha=0.8, zorder=2)
                    _y1, _y2 = ax51.get_ylim()
                    ax51.set_xlim(0, w-1)
                    ax51.text(0.05*w, 0.15*_y1+0.85*_y2,
                            'AKCB'[ipara]+' (Aper %d)'%aper,
                            fontsize=13)
                    if ipara in [2, 3]:
                        ax51.set_xlabel('X', fontsize=13)

                    for tick in ax51.xaxis.get_major_ticks():
                        tick.label1.set_fontsize(11)
                    for tick in ax51.yaxis.get_major_ticks():
                        tick.label1.set_fontsize(11)
                    if w<3000:
                        ax51.xaxis.set_major_locator(tck.MultipleLocator(500))
                        ax51.xaxis.set_minor_locator(tck.MultipleLocator(100))
                    else:
                        ax51.xaxis.set_major_locator(tck.MultipleLocator(1000))
                        ax51.xaxis.set_minor_locator(tck.MultipleLocator(500))
                    if ipara == 3:
                        figname1 = fig_aperpar%aper
                        figname2 = '.'.join(figname1.split('.')[0:-1])+'i.pdf'
                        fig5.savefig(figname2)
                        plt.close(fig5)

        if plot_aperpar:
            # save and close the figure
            if iaper%5==4:
                fig.savefig(fig_aperpar%aper)
                plt.close(fig)
                has_aperpar_fig = False

        # find columns to be corrected in this order
        correct_x_lst = []
        for x in allx:
            pos = position[x]
            y1 = int(max(0, lbound[x]))
            y2 = int(min(h, ubound[x]))
            if (y2-y1)<5:
                continue
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]
            _icen = int(round(pos))
            _i1 = max(0, _icen-5)
            _i2 = min(h, _icen+6)
            sn = math.sqrt(max(0,np.median(ydata[_i1-y1:_i2-y1])*nflat))
            if sn>q_threshold and _satmask.sum()<3 and _badmask.sum()<3:
                correct_x_lst.append(x)

        # find the left and right boundaries of the correction region
        x1, x2 = correct_x_lst[0], correct_x_lst[-1]

        # now loop over columns in correction region
        for x in correct_x_lst:
            interf = interf_lst[x]
            pos = position[x]
            y1 = int(max(0, lbound[x]))
            y2 = int(min(h, ubound[x]))
            xdata = np.arange(y1, y2)
            ydata = data[y1:y2, x]
            _satmask = sat_mask[y1:y2, x]
            _badmask = bad_mask[y1:y2, x]

            # correct flat for this column
            #coeff_A, coeff_k, coeff_c, coeff_bkg = aperpar_lst
            A   = aperpar_lst[0][x]
            k   = aperpar_lst[1][x]
            c   = aperpar_lst[2][x]
            bkg = aperpar_lst[3][x]

            lcorr, rcorr = corr_mask_array[x]
            normx = (xdata-c)*k
            corr_mask = (normx > lcorr)*(normx < rcorr)
            flat = ydata/fitfunc2([A,k,c,bkg], xdata, interf)
            flatmask = corr_mask*~_satmask*~_badmask
            # adopt a decay length at the edges of flat fielding correction
            # zones in each apertures if not all the pixels are flat corrected
            #decay_length = 100
            #if x1 > 0 and x < x1+decay_length:
            #    decay = 1/(math.exp(-(x-(x1+decay_length/2))/4)+1)
            #    flat[flatmask] = (flat[flatmask]-1)*decay + 1
            flatdata[y1:y2, x][flatmask] = flat[flatmask]

            # extract the 1d spectra of the modeled flat using super-sampling
            # integration
            y1s = max(0, np.round(lbound[x]-2, 1))
            y2s = min(h, np.round(ubound[x]+2, 1))
            xdata2 = np.arange(y1s, y2s, 0.1)
            flatmod = fitfunc2([A,k,c,bkg], xdata2, interf)
            # use trapezoidal integration
            # np.trapz(flatmod, x=xdata2)
            # use simpson integration
            flatspec_lst[aper][x] = simps(flatmod, x=xdata2)
            
            #if aper==0:
            #    print(x, y1, y2, flatmask)

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
        message = ('Aperture {:3d}: {:2d} group{:1s}; '
                   'correct {:4d} pixels from {:4d} to {:4d}; '
                   't = {:6.1f} ms').format(
                    aper, len(group_lst), (' ','s')[len(group_lst)>1],
                    len(correct_x_lst),
                    correct_x_lst[0], correct_x_lst[-1],
                    (t2-t1)*1e3
                    )
        print(message)

    # pack the final 1-d spectra of flat
    flatspectable = [(aper, flatspec_lst[aper])
                     for aper in sorted(apertureset.keys())]

    # define the datatype of flat 1d spectra
    flatspectype = np.dtype(
                    {'names':   ['aperture', 'flux'],
                     'formats': [np.int32, (np.float32, w)],}
                    )
    flatspectable = np.array(flatspectable, dtype=flatspectype)

    ###################### aperture loop ends here ########################
    if plot_aperpar and has_aperpar_fig:
        # there's unsaved figure in memory. save and close the figure
        fig.savefig(fig_aperpar%aper)
        plt.close(fig)
        has_aperpar_fig = False


    return flatdata, flatspectable


def default_smooth_flux(x, y, w):
    """

    Args:
        x ():
        y ():

    Returns:

    """
    deg = 7
    mask = np.ones_like(y, dtype=np.bool)
    while(True):
        coeff = np.polyfit(x[mask], y[mask], deg=deg)
        yfit = np.polyval(coeff, x/w)
        yres = y - yfit
        std = yres[fmask].std()
        newmask = np.abs(yres) < 3*std
        if newmask.sum() == mask.sum():
            break
        mask = newmask

    newx = np.arange(w)
    newy = np.polyval(coeff, newx/w)
    return newy, mask


def get_slit_flat(data, mask, apertureset, spectra1d,
        lower_limit=5, upper_limit=5, deg=7, q_threshold=500,
        smooth_flux_func=default_smooth_flux,
        figfile=None
    ):
    """Get the flat fielding image for the slit-fed flat fielding image.

    Args:
        data (:class:`numpy.ndarray`): Image data of flat fielding.
        mask (:class:`numpy.ndarray`): Mask data of flat fielding.
        apertureset (:class:`~gamse.echelle.trace.ApertureSet`): Echelle
            apertures detected in the input file.
    
    Returns:
        :class:`numpy.ndarray`: 2D response map.

    """
    h, w = data.shape

    # find saturation mask and bad pixel mask
    sat_mask = (mask&4 > 0)
    bad_mask = (mask&2 > 0)
    gap_mask = (mask&1 > 0)

    newx = np.arange(w)
    flatmap = np.ones_like(data, dtype=np.float64)

    yy, xx = np.mgrid[:h:,:w:]

    for aper, aper_loc in sorted(apertureset.items()):
        spec = spectra1d[aper]

        domain = aper_loc.position.domain
        d1, d2 = int(domain[0]), int(domain[1])+1
        newx = np.arange(d1, d2)

        fluxdata = spec['flux_mean'][d1:d2]

        position = aper_loc.position(newx)
        lower_line = position - lower_limit
        upper_line = position + upper_limit
        mask = np.zeros_like(data, dtype=np.bool)
        mask[:,d1:d2] = (yy[:,d1:d2] > lower_line)*(yy[:,d1:d2] < upper_line)

        # fit flux
        yfit, fmask = smooth_flux_func(newx, fluxdata, w)

        if figfile is not None:
            fig = plt.figure(dpi=150)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            ax1.plot(spec['flux_sum'],  ls='-',lw=0.5, color='C0')
            ax2.plot(spec['flux_mean'], ls='-',lw=0.5, color='C0')
            ax2.plot(newx, yfit, ls='-',lw=0.5, color='C1')
            ax3.plot(spec['nsum'], ls='-',lw=0.5)
            print(aper,spec['mask'], spec['mask'].sum())
            xx = np.arange(spec['flux_sum'].size)[spec['mask']]
            group_lst = np.split(xx, np.where(np.diff(xx) > 1)[0]+1)
            fig.suptitle('Aperture %3d'%aper)
            fig.savefig(figfile%aper)
            plt.close(fig)

        # construct a 1-D mask marking the pixels with enough S/N
        fluxmask = yfit > q_threshold

        # build a 2-D mask with every row equal to fluxmask
        imgmask = np.zeros_like(data, dtype=np.bool)
        imgmask[:,d1:d2] = fluxmask

        # mark the pixels within the the current aperture
        imgmask = imgmask*mask

        # remove the saturated pixels and bad pixels
        imgmask = imgmask*(~sat_mask)
        imgmask = imgmask*(~bad_mask)
        imgmask = imgmask*(~gap_mask)

        # initialize the constructed pseudo flat
        fitimg = np.zeros_like(data, dtype=np.float64)
        fitimg[:,d1:d2] = yfit

        # build the sensitivity map
        flatmap[imgmask] = data[imgmask]/fitimg[imgmask]
        print(aper)
    return flatmap
