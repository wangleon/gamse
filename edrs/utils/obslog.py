import os
import re
import logging
logger = logging.getLogger(__name__)


class LogItem(object):
    '''Class for observing log items
    '''
    def __init__(self,**kwargs):
        for key in kwargs:
            value = kwargs[key]
            object.__setattr__(self,key,value)

class Log(object):
    '''
    Class for observing log.

    Attributes:
        nchannels (integer): Number of fiber channels.
        item_list (list): List containing :class:`LogItem` instances.
    Examples:
        Initialization
    
        .. code-block:: python
    
            log = Log()
    
        Add new items
    
        .. code-block:: python
    
            log = Log()
            log.add_item(item)
    
        Interation:
    
        .. code-block:: python
    
            for item in log:
                print(item.frameid, item.exptime)
    
    '''
    def __init__(self):
        self.item_list = []

    def __iter__(self):
        return _LogIterator(self.item_list)

    def add_item(self, item):
        '''Add a :class:`LogItem` instance into the observing log.'''
        self.item_list.append(item)

    def get_frameid_lst(self, objectname=None, exptime=None, channel=None):
        '''Get a list of frameids from given conditions.'''
        if self.nchannels > 1:
            # for multi channels
            if channel is not None:
                if channel == 'A':
                    channel_id = 0
                elif channel == 'B':
                    channel_id = 1

            # scan all items
            for item in self.item_list:
                match = False
                if objectname is not None:
                    pass
                if item.objectname:
                    pass
        else:
            for item in self.item_list:
                pass


class _LogIterator(object):
    '''
    Iterator class for :class:`Log`.
    '''
    def __init__(self, item_list):
        self.item_list = item_list
        self.n = len(self.item_list)
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return self.item_list[i]
        else:
            raise StopIteration()

def read_logitem(string, names, types, nchannels=1, column_separator='|',
                 channel_separator=';'):
    '''Read log items.

    Args:
        string (str): Input string.
        names (list): A list of names.
        types (list): A list of type strings.
        nchannels (int): Number of channels.
        column_separator (str): Separator of columns.
        channel_separator (str): Separator of channels in "bjectname" column.
    
    Returns:
        :class:`LogItem`: A :class:`LogItem` instance.

    '''
    logitem = LogItem()

    g = string.split(column_separator)
    for i,value in enumerate(g):
        if i >= len(names):
            continue
        name = names[i]
        if types[i] == 'i':
            value = int(value)
        elif types[i] == 'f':
            value = float(value)
        else:
            value = value.strip()

        # parse object names for multi channels
        if name == 'objectname' and nchannels > 1:
            value = [v.strip() for v in value.split(channel_separator)]

        setattr(logitem, name, value)

    return logitem

def read_log(filename):
    '''Read observing log from an ascii file.

    Args:
        filename (string): Name of the observing log file.

    Returns:
        tuple: A tuple containing:
        
            * **log** (*:class:`Log`*): The observing log.
            * **frame_lst** (*tuple*): Frame list.
    '''

    frame_lst = {}
    log = Log()
    infile = open(filename)
    for row in infile:
        row = row.strip()
        if len(row) == 0 or row[0] == '#':
            continue
        elif row[0] == '%' and '=' in row:
            # read keywords and their values
            g = row[1:].split('=')
            key = g[0].strip()
            val = g[1].strip()
            if key == 'columns':
                # read names and types of each column
                names = [col[0:col.index('(')].strip()
                            for col in val.split(',')]
                types = [col[col.index('(')+1:col.index(')')].strip()
                            for col in val.split(',')]
                setattr(log, 'names', names)
                setattr(log, 'types', types)
            else:
                # convert value to int, float, or string
                if val[0] in ["'", '"'] and val[0] == val[-1]:
                    val = str(val[1:-1])
                elif val.isdigit():
                    val = int(val)
                else:
                    val = float(val)
                setattr(log, key, val)
        else:
            # parse log items
            if hasattr(log, 'nchannels'):
                nchannels = log.nchannels
            else:
                nchannels = 1
            logitem = read_logitem(row, names, types, nchannels=nchannels)
            log.add_item(logitem)

    infile.close()
    logger.info('Observational log file "%s" loaded'%filename)

    return log

def sort_log(log, keyword):
    '''
    Sort the observational log by given keyword.

    Args:
        log (list): A list containing :class:`LogItem` instances.
        keyword (string): Keyword by which the items are sorted.

    Returns:
        list: A new list containing :class:`LogItem` instances.
    '''
    new_dict = {}
    for logitem in log:
        key = getattr(logitem, keyword)
        new_dict[key] = logitem
    new_log = []
    for key in sorted(new_dict.keys()):
        new_log.append(new_dict[key])
    return new_log

def get_input_fileids(log, string):
    '''Get the fileids of the input.

    Args:
        log (:class:`Log`): A :class:`Log` instance
        string (string): The input string
    Returns:
        list: The list of file IDs
    '''

    if len(string.strip())==0:
        return []

    fileid_lst = {}
    for item in log:
        fileid_lst[item.frameid] = item.fileid

    lst = parse_num_seq(string)
    return [fileid_lst[n] for n in lst]

def parse_num_seq(string):
    '''
    Convert the number sequence to list of numbers

    Args:
        string (string): The input string to be parsed.

    Returns:
        list: A list of integer numbers.

    '''
    lst = []
    g1 = string.strip().split(',')
    for rec in g1:
        rec = rec.strip()
        if '-' in rec:
            g2 = rec.split('-')
            n1 = int(g2[0])
            n2 = int(g2[1])
            for n in range(n1,n2+1):
                lst.append(n)
        elif rec.isdigit():
            lst.append(int(rec))
        else:
            continue
    return lst
