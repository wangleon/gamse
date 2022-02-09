import re
import os
from astropy.table import Table

from .common import get_metadata

def make_metatable(rawpath, outputname):

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
                        #('binx',     'i4'),
                        #('biny',     'i4'),
                        #('gain_r',   'f4'),
                        #('gain_b',   'f4'),
                        #('ron_r',    'f4'),
                        #('ron_b',    'f4'),
                        ('pi',       'S50'),
                ], masked=True)
    pattern = '(HARPS\.\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\.fits'
    for fname in sorted(os.listdir(rawpath)):
        mobj = re.match(pattern, fname)
        if not mobj:
            continue
   
        fileid = mobj.group(1)

        filename = os.path.join(rawpath, fname)
        meta = get_metadata(filename)

        binsize = '{}, {}'.format(meta['binx'], meta['biny'])
        gain    = '{}, {}'.format(meta['gain'][0], meta['gain'][1])
        ron     = '{}, {}'.format(meta['ron'][0],  meta['ron'][1])

        item = [(meta['expid'],    False),
                (fileid,           False),
                (meta['category'], False),
                (meta['imgtype'],  False),
                (meta['objname'],  False),
                (meta['ra'],       meta['ra'] is None),
                (meta['dec'],      meta['dec'] is None),
                (meta['exptime'],  False),
                (meta['obsdate'],  False),
                (meta['mode'],     False),
                #(meta['binx'],     False),
                #(meta['biny'],     False),
                #(meta['gain'][0],  False),
                #(meta['gain'][1],  False),
                #(meta['ron'][0],   False),
                #(meta['ron'][1],   False),
                (meta['piname'],   False),
                ]
        value, mask = list(zip(*item))

        metatable.add_row(value, mask=mask)

    metatable['ra'].info.format='%10.6f'
    metatable['dec'].info.format='%9.5f'
    #metatable['gain_r'].info.format='%4.2f'
    #metatable['gain_b'].info.format='%4.2f'
    #metatable['ron_r'].info.format='%4.2f'
    #metatable['ron_b'].info.format='%4.2f'

    metatable.write(outputname, format='ascii.fixed_width_two_line',
                overwrite=True)
