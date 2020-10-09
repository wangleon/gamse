import numpy as np
import astropy.io.fit as fits
import scipy.interpolate as intp
import matplotlib.pyplot as plt

def get_region_lst(header):
    """Get a list of array indices.

    Args:
        header ():
    Returns:
        tuple:
    """
    nx = header['NAXIS1']
    ny = header['NAXIS2']
    binx = header['BIN-FCT1']
    biny = header['BIN-FCT2']

    if (nx, ny)==(2148, 4100) and (binx, biny)==(1, 1):
        sci1_x1, sci1_x2 = 0, 1024
        sci1_y1, sci1_y2 = 0, 4100
        ovr1_x1, ovr1_x2 = 1024, 1024+50
        ovr1_y1, ovr1_y2 = 0, 4100

        sci2_x1, sci2_x2 = 1024+50*2, 1024*2+50*2
        sci2_y1, sci2_y2 = 0, 4100
        ovr2_x1, ovr2_x2 = 1024+50, 1024+50*2
        ovr2_y1, ovr2_y2 = 0, 4100

    elif (nx, ny)==(1124, 2050) and (binx, biny)==(2, 2):
        sci1_x1, sci1_x2 = 0, 512
        sci1_y1, sci1_y2 = 0, 2050
        ovr1_x1, ovr1_x2 = 512, 512+50
        ovr1_y1, ovr1_y2 = 0, 2050

        sci2_x1, sci2_x2 = 512+50*2, 512*2+50*2
        sci2_y1, sci2_y2 = 0, 2050
        ovr2_x1, ovr2_x2 = 512+50, 512+50*2
        ovr2_y1, ovr2_y2 = 0, 2050

    else:
        print(nx, ny, binx, biny)
        pass

    return [
            ((sci1_x1, sci1_x2, sci1_y1, sci1_y2),
             (ovr1_x1, ovr1_x2, ovr1_y1, ovr1_y2)),
            ((sci2_x1, sci2_x2, sci2_y1, sci2_y2),
             (ovr2_x1, ovr2_x2, ovr2_y1, ovr2_y2)),
            ]

def get_std_setup(header):
    """Get standard setup.

    Args:
        header:

    Returns:
        str:
    """
    objtype  = header['DATA-TYP']
    objname  = header['OBJECT']
    ccd_id   = header['DET-ID']
    filter1  = header['FILTER01']
    filter2  = header['FILTER02']
    collim   = header['H_COLLIM']
    crossd   = header['H_CROSSD']
    ech_ang  = header['H_EROTAN']
    cro_ang  = header['H_CROTAN']
    wave_min = int(round(header['WAV-MIN']))
    wave_max = int(round(header['WAV-MAX']))

    std_lst = {
            'StdUa':   (310, 387, 400,  476, 'BLUE', 4.986, 'FREE',  'FREE'),
            'StdUb':   (298, 370, 382,  458, 'BLUE', 4.786, 'FREE',  'FREE'),
            'StdBa':   (342, 419, 432,  508, 'BLUE', 5.386, 'FREE',  'FREE'),
            'StdBc':   (355, 431, 445,  521, 'BLUE', 5.561, 'FREE',  'FREE'),
            'StdYa':   (403, 480, 494,  566, 'BLUE', 6.136, 'FREE',  'FREE'),
            #'StdYb':   (414, 535, 559,  681, 'RED',  4.406, 'FREE',  'KV370'),
            'StdYb':   (414, 540, 559,  681, 'RED',  4.406, 'FREE',  'KV370'),
            'StdYc':   (442, 566, 586,  705, 'RED',  4.619, 'FREE',  'KV389'),
            'StdYd':   (406, 531, 549,  666, 'RED',  4.336, 'FREE',  'KV370'),
            'StdRa':   (511, 631, 658,  779, 'RED',  5.163, 'FREE',  'SC46'),
            'StdRb':   (534, 659, 681,  800, 'RED',  5.336, 'FREE',  'SC46'),
            'StdNIRa': (750, 869, 898, 1016, 'RED',  7.036, 'OG530', 'FREE'),
            'StdNIRb': (673, 789, 812,  937, 'RED',  6.386, 'OG530', 'FREE'),
            'StdNIRc': (617, 740, 759,  882, 'RED',  5.969, 'OG530', 'FREE'),
            'StdI2a':  (498, 618, 637,  759, 'RED',  5.036, 'FREE',  'SC46'),
            'StdI2b':  (355, 476, 498,  618, 'RED',  3.936, 'FREE',  'FREE'),

            }
    for stdname, setup in std_lst.items():
        if objtype=='BIAS' and objname=='BIAS':
            if collim == setup[4] and crossd == setup[4] \
                and abs(cro_ang - setup[5]) < 0.05 \
                and (ccd_id, wave_min, wave_max) in [
                (1, setup[2], setup[3]), (2, setup[0], setup[1])
                ]:
                return stdname
        else:
            if collim == setup[4] and crossd == setup[4] \
                and filter1 == setup[6] and filter2 == setup[7] \
                and abs(cro_ang - setup[5]) < 0.05 \
                and (ccd_id, wave_min, wave_max) in [
                (1, setup[2], setup[3]), (2, setup[0], setup[1])
                ]:
                return stdname
    print('Unknown setup:',collim, crossd, filter1, filter2, cro_ang, ccd_id,
            wave_min, wave_max)
    return 'nonStd'

def print_wrapper(string, item):
    """A wrapper for log printing for Subaru/HDS pipeline.

    Args:
        string (str): The output string for wrapping.
        item (:class:`astropy.table.Row`): The log item.

    Returns:
        str: The color-coded string.

    """
    objtype = item['objtype']
    objname = item['object']

    if objtype=='BIAS' and objname=='BIAS':
        # bias images, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif objtype=='COMPARISON'and objname=='COMPARISON':
        # arc lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'

    elif objtype=='FLAT' and objname=='FLAT':
        # flat, use light red (91)
        return '\033[91m'+string.replace('\033[0m', '')+'\033[0m'

    elif objtype=='OBJECT':
        # sci images, use highlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    else:
        return string


def get_badpixel_mask(ccd_id, binx, biny):
    """Get bad pixel mask for Subaru/HDS CCDs.

    Args:
        ccd_id (int):
        binx (int):
        biny (int):

    """
    if ccd_id == 1:
        if (binx,biny)==(1,1):
            # all False
            mask = np.zeros((4100,2048), dtype=np.bool)
            mask[1124:2069,937] = True
            mask[1124:2059,938] = True
            mask[1136:1974,939] = True
            mask[1210:2018,940] = True
            mask[1130:2056,941] = True
            mask[1130:1994,942] = True
            mask[1130:,1105]    = True
            mask[:,1106]        = True
            mask[1209:,1107]    = True
            mask[1136:,1108]    = True
            mask[1124:,1109]    = True
            mask[1124:,1110]    = True
        elif (binx,biny)==(2,2):
            mask = np.zeros((2050,1024), dtype=np.bool)
            mask[:,653]    = True
            mask[723:,654] = True
            mask[726:,655] = True

    elif ccd_id == 2:
        if (binx,biny)==(1,1):
            mask = np.zeros((4100,2048), dtype=np.bool)
            mask[628:,1720]    = True
            mask[131:,1741]    = True
            mask[131:,1742]    = True
            mask[3378:,1426]   = True
            mask[2674:,127]    = True
            mask[2243:,358]    = True
            mask[2243:,359]    = True
            mask[3115:3578,90] = True
            mask[3003:,88]     = True
            mask[2694:,75]     = True
            mask[2839:,58]     = True
            mask[2839:,59]     = True
            mask[2839:,60]     = True
        elif (binx,biny)==(2,2):
            mask = np.zeros((2050,1024), dtype=np.bool)
            mask[66:,870]   = True
            mask[66:,871]   = True
            mask[315:,860]  = True
            mask[1689:,713] = True
            mask[1121:,179] = True
            mask[1337:,63]  = True
            mask[1348:,37]  = True
            mask[1420:,29]  = True
            mask[1503:,44]  = True
    else:
        print('Error: Wrong det_id:', ccd_id)

    return np.int16(mask)

def fix_image(data, mask):
    if mask.shape != data.shape:
        raise ValueError

    for i, v in enumerate(mask.sum(axis=1)):
        if v > 0:
            m = np.logical_not(mask[i,:])
            x = np.arange(m.size)[m]
            y = data[i,:][m]
            f = intp.InterpolatedUnivariateSpline(x,y)
            data[i,:] = f(np.arange(m.size))
    return data

def correct_overscan(data, header):
    """Correct overscan for an input image and update related information in the
    FITS header.
    
    Args:
        data (:class:`numpy.ndarray`): Input image data.

    Returns:
        tuple: A tuple containing:

            * **data** (:class:`numpy.ndarray`) – Output image with overscan
              corrected.

    """
    binx   = header['BIN-FCT1']
    biny   = header['BIN-FCT2']
    osmin1 = header['H_OSMIN1']
    osmax1 = header['H_OSMAX1']

    #width of overscan region
    ovs_width = osmax1 - osmin1 + 1

    ny, nx = data.shape

    if nx != 2048/binx + ovs_width:
        print('error on image shape of x axis')
        raise ValueError
    if ny != 4100/biny:
        print('error on image shape of y axis')
        raise ValueError

    # initialize bias data
    bias = np.float32(np.zeros_like(data))
    # initialize overscaned data
    overdata = np.zeros((ny, nx-ovs_width), dtype=np.float32)

 def get_bias(config, logtable):
    """Get bias image.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.
    """
    mode = config['reduce'].get('mode')
    bias_file = config['reduce.bias'].get('bias_file')

    if mode=='debug' and os.path.exists(bias_file):
        pass
    else:
        return combine_bias(config, logtable)

def combine_bias(config, logtable):
    """Combine bias images.

    Args:
        config (:class:`configparser.ConfigParser`): Config object.
        logtable (:class:`astropy.table.Table`): Table of Observing log.

    Returns:

    """
    rawpath = config['data']['rawpath']

    bias_filter = lambda item: item['objtype']=='BIAS' \
                           and item['object']=='BIAS'
    logitem_lst = filter(bias_filter, logtable)
    for logitem in logitem_lst:
        fname1 = '{}.fits'.format(logitem['fileid1'])
        filename1 = os.path.join(rawpath, fname1)
        data, head = fits.getdata(filename1, header=True)
        correct_overscan(data, head)
