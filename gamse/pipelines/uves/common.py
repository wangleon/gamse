import os
import astropy.io.fits as fits

def get_metadata(filename):
    # read data
    fname = os.path.basename(filename)

    head0 = fits.getheader(filename)

    naxis1 = head0['NAXIS1']
    naxis2 = head0['NAXIS2']
    binning1 = int(head0['CDELT1'])
    binning2 = int(head0['CDELT2'])


    exptime = head0['EXPTIME']
    if abs(exptime - round(exptime))<1e-2:
        exptime = int(round(exptime))

    obsdate  = head0['DATE-OBS']
    objname  = head0['OBJECT']
    targname = head0.get('ESO OBS TARG NAME', '')
    imgtype  = head0['ESO DPR TYPE']
    category = head0['ESO DPR CATG']
    expid    = head0['ESO DET EXP NO']
    expoid   = head0['OSEXPOID']
    detname  = head0['ESO DET NAME']
    if detname=='ccdUvB - uvesb':
        detector = 'b'
    elif detname=='ccdUvR - ccdUvr':
        detector = 'r'
    progid   = head0['ESO OBS PROG ID']
    piname   = head0['PI-COI']
    ra       = head0.get('RA', None)
    dec      = head0.get('DEC', None)

    # get slit width and length
    path     = head0.get('ESO INS PATH', None)
    if path is None:
        slitwid, slitlen = None, None
    elif path=='BLUE':
        slitwid  = head0.get('ESO INS SLIT2 WID', None)
        slitlen  = head0.get('ESO INS SLIT2 LEN', None)
    elif path=='RED':
        slitwid  = head0.get('ESO INS SLIT3 WID', None)
        slitlen  = head0.get('ESO INS SLIT3 LEN', None)
    else:
        slitwid, slitlen = None, None

    mode     = head0.get('ESO INS MODE', '')
    orifile = head0['ORIGFILE']

    return {
            'naxis1':   naxis1,
            'naxis2':   naxis2,
            'expid':    expid,
            'expoid':   expoid,
            'category': category,
            'imgtype':  imgtype,
            'objname':  objname,
            'targname': targname,
            'exptime':  exptime,
            'ra':       ra,
            'dec':      dec,
            'obsdate':  obsdate,
            'mode':     mode,
            'detname':  detname,
            'detector': detector,
            'slitwid':  slitwid,
            'slitlen':  slitlen,
            'binning':  (binning1, binning2),
            'progid':   progid,
            'piname':   piname,
            }
