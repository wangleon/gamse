import os
import re
import sys
import logging
logger = logging.getLogger(__name__)
import getpass
import platform
import subprocess

import numpy
import scipy
import astropy
import matplotlib

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk

from .. import __version__

def write_system_info():
    """Get system information, and write them into running log.
    """

    # get system information, and write them into the log file
    system, node, release, version, machine, processor = platform.uname()

    if system in ['Linux']:
        # find how many physical processers
        p = subprocess.Popen('grep "physical id" /proc/cpuinfo|sort|uniq|wc -l',
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        processor_number = int(p.stdout.readlines()[0])

        # find the model name of the processors
        p = subprocess.Popen('grep "model name" /proc/cpuinfo|uniq', shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        processor = '; '.join([row.decode('utf-8').split(':')[1].strip()
                               for row in p.stdout.readlines()])

        # find how many cores
        p = subprocess.Popen('grep "cpu cores" /proc/cpuinfo|uniq',shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        cores = int(p.stdout.readlines()[0].decode('utf-8').split(':')[1])

        # get the memory
        p = subprocess.Popen('free -mh',shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        row = p.stdout.readlines()[1]
        info = row.split()
        memory = '{:s} (total); {:s} (used); {:s} (free)'.format(
                    info[1].decode('ascii'),
                    info[2].decode('ascii'),
                    info[3].decode('ascii'))
    else:
        processor_number = 0
        processor        = processor
        cores            = 0
        memory           = 'Unknown'


    distribution = ' '.join(platform.dist())
    username = getpass.getuser()
    node = platform.node()
    abspath = os.path.abspath(os.curdir)
    python_version = platform.python_version()

    info = ['Start reduction.',
            'Node:                  {:s}'.format(node),
            'Processor:             {:d} x {:s} ({:d} cores)'.format(
                                    processor_number, processor, cores),
            'System:                {:s} {:s} {:s}'.format(
                                    system, release, machine),
            'Distribution:          {:s}'.format(distribution),
            'Memory:                {:s}'.format(memory),
            'Username:              {:s}'.format(username),
            'Python version:        {:s}'.format(python_version),
            'Numpy version:         {:s}'.format(numpy.__version__),
            'Scipy version:         {:s}'.format(scipy.__version__),
            'Astropy version:       {:s}'.format(astropy.__version__),
            'Matplotlib version:    {:s}'.format(matplotlib.__version__),
            'Tkinter.ttk version:   {:s}'.format(ttk.__version__),
            'Gamse version:         {:s}'.format(__version__),
            'Working directory:     {:s}'.format(abspath),
            ]
    separator = os.linesep + '  '
    logger.info(separator.join(info))

def extract_date(string):
    """Extract datetime from a string.

    Args:
        string (str): Input string.

    Returns:
        string (str):
        
    """
    date_exp = re.compile('(\d{4})[-]?(\d{2})[-]?(\d{2})')
    mobj = date_exp.match(string)
    if mobj is None:
        return None
    year  = int(mobj.group(1))
    month = int(mobj.group(2))
    day   = int(mobj.group(3))

    # validate the date
    datestr = '{:04d}-{:02d}-{:02d}'.format(year, month, day)
    try:
        day = datetime.datetime.strptime(datestr, '%Y-%m-%d')
        return None
    except:
        return datestr
