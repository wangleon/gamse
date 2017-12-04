import os
import logging
logger = logging.getLogger(__name__)
from configparser import ConfigParser

def read_config():
    '''
    Read the config file ended with `.cfg` in the current directory.

    Returns
    ---------
    : config object or None
        A config object.
    
    Notes
    ------
    Default config file is `reduction.cfg`. If the file does not exist,
    find a file ended with `.cfg`. If such file does not exist, return
    *None*.

    '''

    # scan the .cfg files
    filename_lst = []
    for fname in sorted(os.listdir('./')):
        if fname[-4:]=='.cfg':
            filename_lst.append(fname)

    if len(filename_lst)==0:
        logger.error('Cannot find the config file (*.cfg)')
        return None
    elif len(filename_lst)==1:
        conf_file = filename_lst[0]
        logger.info('Found config file: "%s"'%conf_file)
        config = ConfigParser()
        config.read(conf_file)
        return config
    elif len(filename_lst)>1:
        logger.error('There are %d config files (*.cfg)'%len(filename_lst))
        return None

