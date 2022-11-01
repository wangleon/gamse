import os
import astropy.io.fits as fits

def get_metadata(filename):
    """Get meta data of ESPRESSO raw FITS file.
    
    Args:
        filename(str): 

    Returns:
        dict
    """
    # read data
    fname = os.path.basename(filename)

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
    if head0['TELESCOP'][0:7]!='ESO-VLT':
        print('ERROR: {} has wrong telescope name: {}'.format(
                fname, head0['TELESCOP']))
        return None

    if head0['INSTRUME']!='ESPRESSO':
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
    mode = head0['ESO INS MODE']

    # ccd bin
    binx = head0['ESO DET BINX']
    biny = head0['ESO DET BINY']
    # check image size of both CCD chips
    for i in [1, 2]:
        hdu = hdulist[i]
        head = hdu.header
        data = hdu.data

        # check pixel size of the whole CCD
        nx   = head['ESO DET CHIP NX']
        ny   = head['ESO DET CHIP NY']
        prex = head['ESO DET CHIP PRSCX']
        ovrx = head['ESO DET CHIP OVSCX']
        prey = head['ESO DET CHIP PRSCY']
        ovry = head['ESO DET CHIP OVSCY']
        if prex*8 + nx + ovrx*8 != data1.shape[1]*binx:
            print('ERROR: CCD size mismatch along X')
            print('    nx = {}, prex = {}, ovrx = {}, binx = {}'.format(
                nx, prex, ovrx, binx))
            print('    data has {} columns'.format(data1.shape[1]))
        if  prey + ny + ovry != data1.shape[0]*biny:
            print('ERROR: CCD size mismatch along Y')
            print('    ny = {}, prey = {}, ovry = {}, biny = {}'.format(
                ny, prey, ovry, biny))
            print('    data has {} rows'.format(data1.shape[0]))

        # check pixel size of each readout amplifier
        for i in range(1, 17):
            nx   = head['ESO DET OUT{} NX'.format(i)]
            ny   = head['ESO DET OUT{} NY'.format(i)]
            prex = head['ESO DET OUT{} PRSCX'.format(i)]
            ovrx = head['ESO DET OUT{} OVSCX'.format(i)]
            prey = head['ESO DET OUT{} PRSCY'.format(i)]
            ovry = head['ESO DET OUT{} OVSCY'.format(i)]
            if (prex + nx + ovrx)*8 != data1.shape[1]*binx:
                print('ERROR: CCD size mismatch along X for amplifier {}'.format(i))
            if (prey + ny + ovry)*2 != data1.shape[0]*biny:
                print('ERROR: CCD size mismatch along Y for amplifier {}'.format(i))


        
    # CCD Gain and Readout noise (ron)
    #gain1 = hdulist[1].header['ESO DET OUT1 GAIN']
    #gain2 = hdulist[2].header['ESO DET OUT1 GAIN']
    #ron1  = hdulist[1].header['ESO DET OUT1 RON']
    #ron2  = hdulist[2].header['ESO DET OUT1 RON']

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
            #'gain':     (gain1, gain2),
            #'ron':      (ron1, ron2),
            'piname':   piname,
            }
