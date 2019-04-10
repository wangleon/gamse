import os
import dateutil.parser
import astropy.io.fits as fits

def make_obslog(path):

    # scan the raw files
    fname_lst = sorted(os.listdir(path))
    
    names = ('frameid', 'fileid', 'objectname', 'exptime', 'obsdate',
            'saturation', 'quantile95')
            

    logdata = [[], [], [], [], []]
    
    # start scanning the raw files
    frameid = 0
    for fname in fname_lst:
        if fname[-5:] != '.fits' or fname[-7:] != '.fits.Z':
            continue
        fileid = fname[0:29]
        filename = os.path.join(path, fname)
        data, head = fits.getdata(filename, header=True)
        
        obsdate = dateutil.parser.parse(head['DATE-OBS'])
        exptime = head['EXPTIME']

        objectname = head['OBJECT']

        # determine the total number of saturated pixels
        saturation = (data>=65535).sum()

        # find the 95% quantile
        quantile95 = np.sort(data.flatten())[int(data.size*0.95)]

        logdata[0].append(frameid)
        logdata[1].append(fileid)
        logdata[2].append(objectname)
        logdata[3].append(exptime)
        logdata[4].append(obsdate)
        logdata[5].append(saturation)
        logdata[6].append(quantile95)

        frameid += 1

    logtable = Table(logdata, names=names)
    date = logtable[0]['obsdate'].date
    logtable.pprint(max_lines=-1)
    #logtable.write('obs.txt', for)
