import os
import logging
logger = logging.getLogger(__name__)
from configparser import ConfigParser

def read_config():
    '''
    Read the config file ended with `.cfg` in the current directory.

    Default config file is `reduction.cfg`. If the file does not exist, find a
    file ended with `.cfg`. If such file does not exist, return *None*.

    Returns:
        :class:`ConfigParser`: A :class:`ConfigParser` instance.
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

def find_config(path):
    '''Find the config file in the given directory.

    Args:
        path (string): Path to the searching directory.
    Returns:
        string or None: Path to the config file with filename ended with `.cfg`.
            If not found, return *None*.
    '''
    cfg_lst = [fname for fname in sorted(os.path.listdir(path))
                        if fname[-4:]=='.cfg']
    if len(cfg_lst)==1:
        return cfg_lst[0]
    else:
        print('Error: Multi config file found')
        return None
