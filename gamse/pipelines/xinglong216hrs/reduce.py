import os
import logging
logger = logging.getLogger(__name__)
import configparser

from .reduce_single import reduce_singlefiber
from .reduce_double import reduce_doublefiber

def reduce():
    """2D to 1D pipeline for the High Resolution spectrograph on Xinglong 2.16m
    telescope.
    """

    # find obs log
    logname_lst = [fname for fname in os.listdir(os.curdir)
                        if fname[-7:]=='.obslog']
    if len(logname_lst)==0:
        print('No observation log found')
        exit()
    elif len(logname_lst)>1:
        print('Multiple observation log found:')
        for logname in sorted(logname_lst):
            print('  '+logname)
    else:
        pass

    # read obs log
    logtable = read_obslog(logname_lst[0])

    # load both built-in and local config files
    config = configparser.ConfigParser(
                inline_comment_prefixes = (';','#'),
                interpolation           = configparser.ExtendedInterpolation(),
                )

    # find local config file
    for fname in os.listdir(os.curdir):
        if re.match ('Xinglong216HRS\S*.cfg', fname):
            config.read(fname)
            print('Load Congfile File: {}'.format(fname))
            break

    fibermode = config['data']['fibermode']

    if fibermode == 'single':
        reduce_singlefiber(logtable, config)
    elif fibermode == 'double':
        reduce_multifiber(logtable, config)
    else:
        print('Invalid fibermode:', fibermode)
