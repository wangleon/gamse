import re
import os
from astropy.table import Table

from .common import get_metadata

def make_metatable(rawpath):

    # prepare metatable
    metatable = Table(dtype=[
                        ('expid',    'i4'),
                        ('fileid',   'S28'),
                        ('category', 'S8'),
                        ('imgtype',  'S13'),
                        ('object',   'S20'),
                        ('ra',       'f8'),
                        ('dec',      'f8'),
                        ('exptime',  'f4'),
                        ('obsdate',  'S23'),
                        ('mode',     'S4'),
                        ('binning',  'S7'),
                        #('biny',     'i4'),
                        #('gain_r',   'f4'),
                        #('gain_b',   'f4'),
                        #('ron_r',    'f4'),
                        #('ron_b',    'f4'),
                        ('pi',       'S50'),
                ], masked=True)
    pattern = '(ESPRE\.\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\.fits'
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(pattern, fname)
        if not mobj:
            continue
   
        fileid = mobj.group(1)

        filename = os.path.join(rawpath, fname)
        meta = get_metadata(filename)

        binning = '({}, {})'.format(meta['binx'], meta['biny'])
        #gain    = '{}, {}'.format(meta['gain'][0], meta['gain'][1])
        #ron     = '{}, {}'.format(meta['ron'][0],  meta['ron'][1])

        mask_ra  = (meta['category']=='CALIB' or meta['ra'] is None)
        mask_dec = (meta['category']=='CALIB' or meta['dec'] is None)

        item = [(meta['expid'],     False),
                (fileid,            False),
                (meta['category'],  False),
                (meta['imgtype'],   False),
                (meta['objname'],   False),
                (meta['ra'],        mask_ra),
                (meta['dec'],       mask_dec),
                (meta['exptime'],   False),
                (meta['obsdate'],   False),
                (meta['mode'],      False),
                (binning,           False),
                #(meta['gain'][0],  False),
                #(meta['gain'][1],  False),
                #(meta['ron'][0],   False),
                #(meta['ron'][1],   False),
                (meta['piname'],    False),
                ]
        value, mask = list(zip(*item))

        metatable.add_row(value, mask=mask)

        # print information
        string_lst = [
                '{:5d}'.format(meta['expid']),
                fileid,
                '{:7s}'.format(meta['category']),
                '{:20s}'.format(meta['imgtype']),
                '{:20s}'.format(meta['objname']),
                ' '*9 if mask_ra  else '{:9.5f}'.format(meta['ra']),
                ' '*9 if mask_dec else '{:9.5f}'.format(meta['dec']),
                '{:6g}'.format(meta['exptime']),
                meta['obsdate'],
                '{:10s}'.format(meta['mode']),
                binning,
                meta['piname'],
                ]
        print(' '.join(string_lst))

    metatable['ra'].info.format='%10.6f'
    metatable['dec'].info.format='%9.5f'
    maxlen = max([len(s) for s in metatable['category']])
    metatable['category'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['imgtype']])
    metatable['imgtype'].info.format='%-{}s'.format(maxlen)
    maxlen = max([len(s) for s in metatable['object']])
    metatable['object'].info.format='%-{}s'.format(maxlen)
    #metatable['gain_r'].info.format='%4.2f'
    #metatable['gain_b'].info.format='%4.2f'
    #metatable['ron_r'].info.format='%4.2f'
    #metatable['ron_b'].info.format='%4.2f'

    return metatable

def split_obslog():
    for fname in sorted(os.listdir(os.curdir)):
        pass




def reduce():
    
    # get mode and bin for science
    pass
