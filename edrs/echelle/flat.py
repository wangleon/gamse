import os
import math
import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as fits

import matplotlib.pyplot as plt

from ..ccdproc import save_fits, array_to_table, table_to_array
from ..utils.onedarray import pairwise

def mosaic_flat_interact(filename_lst, outfile,
                         mosaic_file,
                         reg_file,
                         disp_axis   = 0,
                         mask_surfix = '_msk',
                         ):
    '''
    Display an interacitve interface to mosaic the flat images

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
        ''' Select an area when clicking on the figure
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

def detect_gap(
        data, x0,
        ccf_ulimit = 50,
        ccf_llimit = 50,
        step       = 50,
        order      = 4, 
        ):
    '''
    Detect the curve of gap between two orders along Y-xias.

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
    Read mosaic boundary information from an existing ascii file.

    Args:
        filename (string): Name of the mosaic file.
    Returns:
        tuple: A tuple containing:

            * **coeff_lst** (*list*): A list containing coefficients of
                polynomials for the boundaries
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
        aperture_set_lst (list): List of :class:`ApertureSet`.
        max_count (float): Maximum count.
    Returns:
        No returns.
    See Also:
        :func:`mosaic_flat_interact`
    '''

    from ..echelle.trace import select_ref_tracefile, align_apertures

    for channel, aperset_lst in sorted(aperture_set_lst.items()):

        print(channel, aperset_lst)


        all_aperloc_lst = []
        # all_aperloc_lst  = [
        #  [tracename1: aper_loc, tracename2: aper_loc],
        #  [tracename1: aper_loc, tracename2: aper_loc],
        #  [tracename1: aper_loc, tracename2: aper_loc],
        # ]

        for itrace, (tracename, aperset) in enumerate(aperset_lst.items()):
            print(tracename, len(aperset))
            for o in aperset:
                aper_loc = aperset[o]
                print(o, aper_loc, aper_loc.mean, aper_loc.nsat)
                if itrace == 0:
                    all_aperloc_lst.append({tracename: aper_loc})
                else:
                    insert = False
                    for ilist, list1 in enumerate(all_aperloc_lst):
                        if tracename in list1:
                            continue
                        for _tracename, _aperloc in list1.items():
                            distance = aper_loc.distance(_aperloc)
                            if abs(distance)<3:
                                all_aperloc_lst[ilist][tracename] = aper_loc
                                insert = True
                                break
                        if insert:
                            break
            for list1 in all_aperloc_lst:
                for tracename, aperloc in list1.items():
                    print(tracename, aperloc)
                print('-----')
        

        # select the reference flat
        ref_flatname = select_ref_tracefile(aperset_lst)

        # align all the orders in different flats
        aperset_lst = align_apertures(aperset_lst, ref_flatname)

        # search for minimum and maximum aperture
        min_aperture_lst, max_aperture_lst = [], []

        # find the minimum and maximum aperture number
        min_aper = min([min(aper_set.dict.keys())
                    for aper_set in aperset_lst.values()])
        max_aper = max([max(aper_set.dict.keys())
                        for aper_set in aperset_lst.values()])
        logger.info('Aperture range: %d - %d (%d aperture)'%(
                    min_aper, max_aper, max_aper - min_aper + 1))

        # now mosaic an entire order list
        comp_key = 'median' # can be changed to ['max'|'mean'|'median']
        apeture_select_lst = {}
        select_flat_lst = []
        for aperture in range(min_aper, max_aper+1):
            # search all flat files and find the one with maxium flux
            max_flatname = None
            max_flux     = -999.
            for flatname, aperture_set in sorted(aperture_set_lst.items()):
                if aperture not in aperture_set:
                    continue
                aperture_loc = aperture_set[aperture]

            if aperture_loc.nsat > 0:
                # skip the one with saturation pixels
                continue
            if aperture_loc.max > max_count:
                # skip orders with peak flux larger than max_count
                continue
            if getattr(aperture_loc, comp_key) > max_flux:
                max_flux = getattr(aperture_loc, comp_key)
                max_flatname = flatname

        aperture_select_lst[aperture] = max_flatname
        logger.debug('"%s" is selected for order %d with %s=%f'%(
                      max_flatname, aperture, comp_key, max_flux))
        if max_flatname not in select_flat_lst:
            select_flat_lst.append(max_flatname)

        # write selected filename in running log
        message_lst = ['selected flat names:']
        message_lst.append('order flatname')
        for aperture, flatname in aperture_select_lst.items():
            message_lst.append('  %2d %s'%(aperture, flatname))
        logger.info(os.linesep.join(message_lst))

        # find mask

        # read data
        prev_shape = None
        flatdata_lst, maskdata_lst = {}, {}
        for filename in filename_lst:
            flatname = os.path.basename(filename)[0:-5]
            data = fits.getdata(filename)
            flatdata_lst[flatname] = data
            maskdata_lst[flatname] = np.zeros_like(data, dtype=np.bool)
            shape = data.shape
            if prev_shape is not None and shape != prev_shape:
                logger.error(
                    'Image shape of "%s" (%d x %d) does not match previous (%d x %d)'%(
                    flatname, shape[0], shape[1], prev_shape[0], prev_shape[1])
                    )
            prev_shape = shape

        print(aperture_select_lst)
        for aperture in range(min_aperture, max_aperture+1):
            flatname = aperture_select_lst[order]
            if aperture == min_aperture:
                maskdata_lst[flatname][:,:] = True
            elif flatname != prev_flatname:
                prev_aperture_loc = aperture_set_lst[prev_flatname][aperture-1]
                this_aperture_loc = aperture_set_lst[flatname][aperture]
                h, w = this_aperture_loc.shape

                # upper coeff of previous order
                upper_coeff = np.array(prev_order_loc.coeff_upper)
                # lower coeff of this order
                lower_coeff = np.array(this_order_loc.coeff_lower)

                # if length of coefficients of above polynomials are not equal,
                # add zeros in front of the coefficients
                n_upper = len(upper_coeff)
                n_lower = len(lower_coeff)
                if n_upper < n_lower:
                    for i in range(abs(n_upper-n_lower)):
                        upper_coeff = np.insert(upper_coeff, 0, 0.0)
                elif n_upper > n_lower:
                    for i in range(abs(n_upper-n_lower)):
                        lower_coeff = np.insert(lower_coeff, 0, 0.0)
                # find the coefficients of the boundary polynomial
                bound_coeff = (upper_coeff + lower_coeff)/2.
                cut_bound = np.polyval(bound_coeff,np.arange(w)/float(w))*h

                yy, xx = np.mgrid[:h:,:w:]
                m = yy > np.round(cut_bound)
                maskdata_lst[prev_flatname][m] = False
                maskdata_lst[flatname][m] = True

            prev_flatname = flatname

        mos_flatdata = np.zeros(shape, dtype=np.float32)
        for flatname, maskdata in sorted(maskdata_lst.iteritems()):
            flatdata = flatdata_lst[flatname]
            mos_flatdata += flatdata*maskdata

        # save the mosaic flat as FITS file
        save_fits(outfile, mos_flatdata)

def mosaic_image(data_lst, head_lst, outfile, coeff_lst, disp_axis):
    mos_data = np.zeros(shape)

def save_mosaic_reg(filename, coeff_lst, disp_axis, shape, npoints=20):
    '''
    Save boundaries data in a `.reg` file
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

