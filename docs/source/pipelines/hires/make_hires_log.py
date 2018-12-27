#!/usr/bin/env python3
import os,sys,re
import astropy.io.fits as fits

def main():
    fn_fmt = 'HI\.\d{8}\.\d{5}\.fits'
    head_lst = [
            'fname','datatype','frameno','n_ccds',
            'target','exptime','date','utc',
            'i2','ra','dec',
            'deckname','filter1','filter2',
            'snr','gain','ron',
            'slit_len','slit_wid','resolution','pix_scale',
            'progid','progpi'
            ]
    c = 0
    string_lst = []
    for fname in sorted(os.listdir(sys.argv[1])):
        if not re.match(fn_fmt,fname):
            continue
        c += 1
        f = fits.open(os.path.join(sys.argv[1],fname))
        try:
            target   = f[0].header['TARGNAME']
        except:
            target   = ''
        if len(f)>1:
            n_ccds   = len(f)-1
        else:
            n_ccds   = 1
        try:
            frame = f[0].header['FRAME']
        except:
            frame = ''
        frameno  = f[0].header['FRAMENO']
        try:
            equinox  = f[0].header['EQUINOX']
        except:
            equinox  = ''
        try:
            ra       = f[0].header['RA']
        except:
            ra       = ''
        try:
            dec      = f[0].header['DEC']
        except:
            dec      = ''
        date     = f[0].header['DATE-OBS']
        try:
            utc  = f[0].header['UTC']
        except:
            utc  = f[0].header['UT']
        progid   = f[0].header['PROGID']
        progpi   = f[0].header['PROGPI']
        exptime  = f[0].header['ELAPTIME']
        try:
            i2in  = f[0].header['IODIN']
            i2out = f[0].header['IODOUT']
        except:
            i2in  = False
            i2out = True
        imgtype  = f[0].header['IMAGETYP']
        try:
            deckname = f[0].header['DECKNAME']
        except:
            deckname = ''
        try:
            filter1  = f[0].header['FIL1NAME']
        except:
            filter1  = ''
        try:
            filter2  = f[0].header['FIL2NAME']
        except:
            filter2 = ''
        snr      = f[0].header['SIG2NOIS']
        slit_len = f[0].header['SLITLEN']
        slit_wid = f[0].header['SLITWIDT']
        resolution = f[0].header['SPECRES']
        pix_scale  = f[0].header['SPATSCAL']

        gain = ','.join(['%4.2f'%f[0].header['CCDGN%02d'%(i+1)] for i in range(n_ccds)])
        ron = ','.join(['%4.2f'%f[0].header['CCDRN%02d'%(i+1)] for i in range(n_ccds)])

        f.close()

        # check
        if imgtype.strip() == 'object':
            # science frame
            datatype = 'sci'
            if frame.strip() != 'FK5':
                print(' warning: science file %s frame = %s'%(fname,frame))
            if equinox.strip() != '2000.0':
                print(' warning: science file %s equinox = %s'%(fname,equinox))
        else:
            # calibration frame
            datatype = 'cal'
            if target.strip() not in ['unknown','horizon lock','dome flat','']:
                print(' warning: calibration file %s target = %s'%(fname,target))
            if frame.strip() not in ['mount az/el','FK5']:
                print(' warning: calibration file %s frame = %s'%(fname,frame))
            target = imgtype
        if i2in + i2out != 1:
            print(' warning: %s i2in = %d, i2out = %d'%(fname,i2in,i2out))
        if c != 1 and frameno != prev_frameno + 1:
            print(' warning: %s frameno (%d) != prev_frameno (%d) + 1'%(fname,frameno,prev_frameno))

        output = [
                '%17s'%fname[0:-5],
                '%3s'%datatype,
                '%4d'%frameno,
                '%1d'%n_ccds,
                '%15s'%target,
                '%6.1f'%exptime,
                '%10s'%date,
                '%11s'%utc,
                '%1d'%i2in,
                '%10s'%ra,
                '%11s'%dec,
                #'%4s'%frame,
                #'%6s'%equinox,
                '%2s'%deckname,
                '%5s'%filter1,
                '%5s'%filter2,
                '%5.1f'%snr,
                '%14s'%gain,
                '%14s'%ron,
                #'%6.3f'%slit_len,
                '%6s'%str(slit_len),
                #'%5.3f'%slit_wid,
                '%5s'%str(slit_wid),
                #'%6d'%resolution,
                '%6s'%str(resolution),
                '%5.3f'%pix_scale,
                '%6s'%progid,
                '%s'%progpi,
                #'%10s'%imgtype,
                ]
        string_lst.append('|'.join(output))

        prev_frameno = frameno


    # save the log file
    log_filename = '%s.log'%date
    if os.path.exists(log_filename):
        print('####ERROR: file %s exists'%log_filename)
        exit()
    log_file = open(log_filename,'w')
    log_file.write('#'+(','.join(head_lst))+os.linesep)
    for string in string_lst:
        log_file.write(string+os.linesep)
    log_file.close()


if __name__=='__main__':
    main()
