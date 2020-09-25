import numpy as np

from ...echelle.trace import TraceFigureCommon

def print_wrapper(string, item):
    """A wrapper for log printing for HIRES pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    imgtype = item['imgtype']
    obj     = item['object']

    if len(obj)>=4 and obj[0:4]=='bias':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif imgtype=='sci':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif len(obj)>=8 and obj[0:8]=='flatlamp':
        # flat images, analyze nsat
        nsat_1 = item['nsat_1']
        nsat_2 = item['nsat_2']
        nsat_3 = item['nsat_3']
        q95_1  = item['q95_1']
        q95_2  = item['q95_2']
        q95_3  = item['q95_3']
        q_lst = [q95_1 if q95_1 < 6e4 else -1,
                 q95_2 if q95_2 < 6e4 else -1,
                 q95_3 if q95_3 < 6e4 else -1]

        maxccd = np.argmax(q_lst)

        if max(q_lst)<0:
            # all CCDs are saturated
            return string

        elif 'quartz1' in obj and maxccd == 0:
            # quartz1 for UV, use light magenta (95)
            return '\033[95m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 0:
            # blue flat, use light blue (94)
            return '\033[94m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 1:
            # green flat, use light green (92)
            return '\033[92m'+string.replace('\033[0m', '')+'\033[0m'

        elif maxccd == 2:
            # red flat, use light red (91)
            return '\033[91m'+string.replace('\033[0m', '')+'\033[0m'

        else:
            # no idea
            return string

    elif len(obj)>=7 and obj[0:7]=='arclamp':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'
    else:
        return string

def parse_3ccd_images(hdu_lst):
    """Parse the 3 CCD images.

    Args:
        hdu_lst (:class:`astropy.io.fits.HDUList`): Input HDU list.

    Returns:
        tuple: A tuple containing:

            * **data_lst** (*tuple*): A tuple of (Image1, Image2, Image3).
            * **mask_lst** (*tuple*): A tuple of (Mask1, Mask2, Mask3).

    """
    if len(hdu_lst) != 4:
        raise ValueError

    # get CCD Binning
    tmp = hdu_lst[0].header['CCDSUM'].split()
    binx, biny = int(tmp[0]), int(tmp[1])
    # get data sect rectanle
    dataset_lst = {(2, 1): ('[7:1030,1:4096]', (6, 1030), (0, 4096)),
                   (2, 2): ('[7:1030,1:2048]', (6, 1030), (0, 2048)),
                  }
    datasec, (x1, x2), (y1, y2) = dataset_lst[(binx, biny)]
    # get data section
    data_lst = [hdu_lst[i+1].data[y1:y2, x1:x2] for i in range(3)
                if hdu_lst[i+1].header['DATASEC']==datasec]

    # get saturated masks
    mask_sat1 = data_lst[0]==65535   # for UV CCD, saturated pixels are 65535.
    mask_sat2 = data_lst[1]==0       # for green & red CCDs, saturated pixels
    mask_sat3 = data_lst[2]==0       # are 0.
    # get bad pixel masks
    #mask_bad1 = np.zeros_like(mask_sat1, dtype=np.bool)
    #mask_bad2 = np.zeros_like(mask_sat1, dtype=np.bool)
    #mask_bad3 = np.zeros_like(mask_sat1, dtype=np.bool)
    mask_bad1 = get_badpixel_mask((binx, biny), ccd=1)
    mask_bad2 = get_badpixel_mask((binx, biny), ccd=2)
    mask_bad3 = get_badpixel_mask((binx, biny), ccd=3)
    # pack masks
    mask1 = np.int16(mask_sat1)*4 + np.int16(mask_bad1)*2
    mask2 = np.int16(mask_sat2)*4 + np.int16(mask_bad2)*2
    mask3 = np.int16(mask_sat3)*4 + np.int16(mask_bad3)*2

    mask_lst = (mask1, mask2, mask3)

    # fix saturated pixels in the green and red CCDs
    data_lst[1][mask_sat2] = 65535
    data_lst[2][mask_sat3] = 65535

    return (data_lst, mask_lst)

def get_badpixel_mask(binning, ccd=0):
    """Get bad pixel mask for HIRES CCDs.

    Args:
        binning (tuple): CCD binning (*bin_x*, *bin_y*).
        ccd (int): CCD number.

    Returns:
        mask (:class:`numpy.ndarray`): Mask Image.

    """
    # for only 1 CCD
    if ccd == 0:
        if binning == (1, 1):
            # all Flase
            mask = np.zeros((2048, 2048), dtype=np.bool)
            mask[:,    1127] = True
            mask[:375, 1128] = True
            mask[:,    2007] = True
            mask[:,    2008] = True
    # for 3 CCDs
    elif ccd == 1:
        # for Blue CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[3878:,   4]  = True
            mask[3008:, 219]  = True
            mask[4005:, 337]  = True
            mask[1466:, 411]  = True
            mask[1466:, 412]  = True
            mask[3486:, 969]  = True
            mask[:,     994:] = True
    elif ccd == 2:
        # for Green CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[3726:, 323] = True
            mask[3726:, 324] = True
    elif ccd == 3:
        # for Red CCD
        if binning == (2, 1):
            # all False
            mask = np.zeros((4096, 1024), dtype=np.bool)
            mask[1489:2196, 449]  = True
            mask[:,         0:45] = True
    return np.int16(mask)


def mosaic_3_images(data_lst, mask_lst):
    """Mosaic three images.

    Args:
        data_lst (list): List of image data.
        mask_lst (list): List of mask data.

    Returns:
        tuple:
    """
    data1, data2, data3 = data_lst
    mask1, mask2, mask3 = mask_lst
    gap_rg, gap_gb = 26, 20

    # mosaic image: allimage and allmask
    h3, w3 = data3.shape
    h2, w2 = data2.shape
    h1, w1 = data1.shape

    hh = h3 + gap_rg + h2 + gap_gb + h1
    allimage = np.ones((hh, w3), dtype=data1.dtype)
    allmask = np.zeros((hh, w3), dtype=np.int16)
    r1, g1, b1 = 0, h3+gap_rg, h3+gap_rg+h2+gap_gb
    r2, g2, b2 = r1+h3, g1+h2, b1+h1
    allimage[r1:r2] = data3
    allimage[g1:g2] = data2
    allimage[b1:b2] = data1
    allmask[r1:r2] = mask3
    allmask[g1:g2] = mask2
    allmask[b1:b2] = mask1
    # fill gap with gap pixels
    allmask[r2:g1] = 1
    allmask[g2:b1] = 1

    return allimage, allmask


class TraceFigure(TraceFigureCommon):
    """Figure to plot the order tracing.
    """
    def __init__(self):
        TraceFigureCommon.__init__(self, figsize=(20,10), dpi=150)
        self.ax1 = self.add_axes([0.05,0.07,0.50,0.86])
        self.ax2 = self.add_axes([0.59,0.55,0.36,0.34])
        self.ax3 = self.add_axes([0.59,0.13,0.36,0.34])
        self.ax4 = self.ax3.twinx()
