import os
import astropy.io.fits as fits

def get_metadata(filename):
    """Get meta data of HARPS raw FITS file.
    
    Args:
        filename(str): Filename of HARPS raw FITS file.

    Returns:
        dict
    """

    fname = os.path.basename(filename)

    # read data
    hdulist = fits.open(filename)
    head0 = hdulist[0].header
    head1 = hdulist[1].header
    head2 = hdulist[2].header
    data1 = hdulist[1].data
    data2 = hdulist[2].data
    hdulist.close()

    if len(hdulist)!=3:
        print('ERROR: HDUList lenth =', len(hdulist))
        return None

    # regular check
    if head0['TELESCOP']!='ESO-3P6':
        print('ERROR: {} has wrong telescope name: {}'.format(
                fname, head0['TELESCOP']))
        return None

    if head0['INSTRUME']!='HARPS':
        print('ERROR: {} has wrong instrument name: {}'.format(
                fname, head0['INSTRUME']))
        return None

    exptime = head0['EXPTIME']
    if abs(exptime - round(exptime))<1e-2:
        exptime = int(round(exptime))

    obsdate  = head0['DATE-OBS']
    objname  = head0['OBJECT']
    imgtype  = head0['ESO DPR TYPE']
    category = head0['ESO DPR CATG']
    expid    = head0['ESO DET EXP NO']
    piname   = head0['PI-COI']
    ra       = head0.get('RA', None)
    dec      = head0.get('DEC', None)

    # check equinox and ra-dec system
    if None not in (ra, dec):
        equinox = head0['EQUINOX']
        if equinox != 2000.:
            print('ERROR: Wrong equinox: {}'.format(equinox))
        radecsys = head0['RADECSYS']
        if radecsys != 'FK5':
            print('ERROR: Wrong RADECSYS: {}'.format(radecsys))

    # instrument mode.
    # HAM (high accuracy) mode: INS HEFS ST=F, INST MODE=HARPS
    # EGGS (high efficiency) mode: INS HEFS ST=T, INST MODE=EGGS
    _hefs = head0['ESO INS HEFS ST']
    _mode = head0['ESO INS MODE']
    if (_mode, _hefs)==('HARPS', False):
        mode = 'HAM'
    elif (_mode, _hefs)==('EGGS', True):
        mode = 'EGGS'
    else:
        mode = ''
        print('ERROR: mode error: MODE={}, HEFS_ST={}'.format(_mode, _hefs))

    # ccd bin
    binx = head0['ESO DET WIN1 BINX']
    biny = head0['ESO DET WIN1 BINY']
    # check image size of both CCD chips
    for i in [1, 2]:
        hdu = hdulist[i]
        head = hdu.header
        data = hdu.data
        nx   = head['ESO DET CHIP{} NX'.format(i)]
        ny   = head['ESO DET CHIP{} NY'.format(i)]
        prex = head['ESO DET OUT1 PRSCX']
        ovrx = head['ESO DET OUT1 OVSCX']
        prey = head['ESO DET OUT1 PRSCY']
        ovry = head['ESO DET OUT1 OVSCY']
        if prex + binx*nx + ovrx != data1.shape[1]:
            print('ERROR: CCD size mismatch along X')
        if  prey + biny*ny + ovry != data1.shape[0]:
            print('ERROR: CCD size mismatch along Y')
    # CCD Gain and Readout noise (ron)
    gain1 = hdulist[1].header['ESO DET OUT1 GAIN']
    gain2 = hdulist[2].header['ESO DET OUT1 GAIN']
    ron1  = hdulist[1].header['ESO DET OUT1 RON']
    ron2  = hdulist[2].header['ESO DET OUT1 RON']

    return {
            'expid':    expid,
            'category': category,
            'imgtype':  imgtype,
            'objname':  objname,
            'exptime':  exptime,
            'ra':       ra,
            'dec':      dec,
            'obsdate':  obsdate,
            'mode':     mode,
            'binx':     binx,
            'biny':     biny,
            'gain':     (gain1, gain2),
            'ron':      (ron1, ron2),
            'piname':   piname,
            }
