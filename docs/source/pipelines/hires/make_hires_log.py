#!/usr/bin/env python3
import os,sys,re
import astropy.io.fits as fits

def main():
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
        if not re.match('HI\.\d{8}\.\d{5}\.fits', fname):
            continue
        c += 1
        filepath = os.path.join(sys.argv[1], fname)
        hdu_lst = fits.open(filepath)
        head0 = hdu_lst[0].header
        hdu_lst.close()
        target = head0.get('TARGNAME', '')
        if len(hdu_lst)>1:
            n_ccds   = len(hdu_lst)-1
        else:
            n_ccds   = 1
        frame      = head0.get('FRAME', '')
        frameno    = head0.get('FRAMENO')
        equinox    = head0.get('EQUINOX', '')
        ra         = head0.get('RA', '')
        dec        = head0.get('DEC', '')
        date       = head0.get('DATE-OBS')
        utc        = head0.get('UTC', head0.get('UT'))
        progid     = head0.get('PROGID')
        progpi     = head0.get('PROGPI')
        exptime    = head0.get('ELAPTIME')
        i2in       = head0.get('IODIN', False)
        i2out      = head0.get('IODOUT', True)
        imgtype    = head0.get('IMAGETYP')
        deckname   = head0.get('DECKNAME', '')
        filter1    = head0.get('FIL1NAME', '')
        filter2    = head0.get('FIL2NAME', '')
        snr        = head0.get('SIG2NOIS')
        slit_len   = head0.get('SLITLEN')
        slit_wid   = head0.get('SLITWIDT')
        resolution = head0.get('SPECRES')
        pix_scale  = head0.get('SPATSCAL')

        gain = ','.join(['%4.2f'%head0['CCDGN%02d'%(i+1)] for i in range(n_ccds)])
        ron  = ','.join(['%4.2f'%head0['CCDRN%02d'%(i+1)] for i in range(n_ccds)])

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
            print(' warning: %s frameno (%d) != prev_frameno (%d) + 1'%(
                fname,frameno,prev_frameno))

        output = ['%17s'%fname[0:-5],   '%3s'%datatype,  '%4d'%frameno,
                  '%1d'%n_ccds,         '%15s'%target,   '%6.1f'%exptime,
                  '%10s'%date,          '%11s'%utc,      '%1d'%i2in,
                  '%10s'%ra,            '%11s'%dec,      '%2s'%deckname,
                  '%5s'%filter1,        '%5s'%filter2,   '%5.1f'%snr,
                  '%14s'%gain,          '%14s'%ron,      '%6s'%str(slit_len),
                  '%5s'%str(slit_wid),  '%6s'%str(resolution),
                  '%5.3f'%pix_scale,    '%6s'%progid,    '%s'%progpi,
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
