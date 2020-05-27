import os
import re
import logging
logger = logging.getLogger(__name__)

from astropy.io import registry
from astropy.table import Table, MaskedColumn
from astropy.time import Time

def _read_obslog(filename):
    """A customized function use to read the observing log

    Args:
        filename (str): Filename of the obsereving log file.

    Returns:
        :class:`astropy.table.Table`: An observing log object.

    """
    infile = open(filename)
    count_row = 0
    for row in infile:
        if len(row.strip())==0 or row[0] in ['#']:
            # skip this line if blank or start with '#'
            continue
        count_row += 1

        if count_row == 1:
            # first row: column names
            name_row = row
            continue

        if count_row == 2:
            # second row: column types
            dtype_row = row
            continue

        if count_row == 3:
            # third row: horizontal lines
            separator_row = row
            row = row.strip()
            index_lst = []

            # find the delimiter 
            aset = set(row.replace('-',''))
            if len(aset) == 1:
                delimiter = aset.pop()
            else:
                print('Wrong delimiter:', aset)
                delimiter = ' '

            g = row.split(delimiter)
            lens = [len(v) for v in g]
            i2 = -1
            for i, l in enumerate(lens):
                i1 = i2 + 1
                i2 = i1 + l
                index_lst.append((i1, i2))

            # parse column names
            names = [name_row[i1:i2].strip() for (i1, i2) in index_lst]
            data = [[] for title in names]
            mask = [[] for title in names]

            # parse data type list
            dtypes = [dtype_row[i1:i2].strip() for (i1, i2) in index_lst]
            continue

        # row_count > 3: parsing the data

        if row == separator_row:
            # if this line is a separator, skip
            continue

        g = [row[i1:i2].strip() for (i1, i2) in index_lst]
        if dtypes[0]=='int' and ('-' in g[0] or ',' in g[0]):
            # the first column is an abbreviation
            fid_lst = parse_num_seq(g[0])
            for fid in fid_lst:
                data[0].append(fid)
                mask[0].append(False)
                for iv, v in enumerate(g[1:]):
                    v = v.strip()
                    data[iv+1].append(v)
                    mask[iv+1].append(len(v)==0)
        else:
            for iv, v in enumerate(g):
                v = v.strip()
                data[iv].append(v)
                mask[iv].append(len(v)==0)

    infile.close()

    # convert data to their corresponding types according to the header
    for icol, dtype in enumerate(dtypes):
        if dtype=='int':
            data[icol] = [(int(v), 0)[mask[icol][i]]
                            for i, v in enumerate(data[icol])]
        elif dtype=='float':
            data[icol] = [(float(v), -990.)[mask[icol][i]]
                            for i, v in enumerate(data[icol])]
        elif dtype=='bool':
            data[icol] = [(v=='True', False)[mask[icol][i]]
                            for i, v in enumerate(data[icol])]
        elif dtype=='time':
            data[icol] = Time([(v, '1970-01-01T00:00:00')[mask[icol][i]]
                            for i, v in enumerate(data[icol])])
        else:
            pass
            # default is string

    # convert data to astropy table
    logtable = Table(masked=True)
    for icol, name in enumerate(names):
        column = MaskedColumn(data[icol], name=name, mask=mask[icol])
        logtable.add_column(column)

    return logtable


def read_obslog(filename):
    """Read the observing log.

    Args:
        filename (str): Filename of the obsereving log file.

    Returns:
        :class:`astropy.table.Table`: An observing log object.
    
    """
    registry.register_reader('obslog', Table, _read_obslog)
    table = Table.read(filename, format='obslog')
    registry.unregister_reader('obslog', Table)
    return table

def _write_obslog(table, filename, overwrite=False):
    """Write the log table into a file.

    Args:
        table (:class:`astropy.table.Table`): Observing log table.
        filename (str): Name of the output ASCII file.
        overwrite (bool): 

    """

    if not overwrite and os.path.exists(filename):
        print('Warning: File "{}" already exists'.format(filename))
        raise ValueError

    pformat_lst = table.pformat_all()
    separator = pformat_lst[1]

    outfile = open(filename, 'w')
    outfile.write(pformat_lst[0]+os.linesep)

    # find the length of each column
    collen_lst = [len(string) for string in separator.split()]

    # find a list of datatypes
    row = table[0]
    dtype_string_lst = []
    for (name, dtype) in table.dtype.descr:
        if dtype in ['<i2', '<i4']:
            dtype_string = 'int'
        elif dtype in ['<f4', '<f8']:
            dtype_string = 'float'
        elif dtype[0:2] == '|S':
            dtype_string = 'str'
        elif dtype == '|O':
            if isinstance(row[name], Time):
                dtype_string = 'time'
            else:
                dtype_string = ''
        dtype_string_lst.append(dtype_string)

    # get a list of space-filled datatypes
    dtypes = [dtype.center(collen_lst[i])
                for i, dtype in enumerate(dtype_string_lst)]

    # write data types
    outfile.write(' '.join(dtypes)+os.linesep)

    # write the separator
    outfile.write(separator+os.linesep)

    # write rows
    prev_row = row
    for i, row in enumerate(table):

        # write a separator between cal and sci
        if 'imgtype' in table.colnames \
            and row['imgtype'] != prev_row['imgtype']:
            outfile.write(separator+os.linesep)

        # write a separator between different cal frames
        if 'imgtype' in table.colnames \
            and row['imgtype']=='cal' and prev_row['imgtype']=='cal' \
            and row['object']!=prev_row['object']:
            outfile.write(separator+os.linesep)

        outfile.write(pformat_lst[i+2]+os.linesep)
        prev_row = row

    outfile.close()

def write_obslog(table, filename):
    """Write an observing log table to an ASCII file.

    Args:
        table (:class:`astropy.table.Table`): Observing log table.
        filename (str): Name of the output ASCII file.

    """

    registry.register_writer('obslog', Table, _write_obslog)
    table.write(filename, format='obslog')
    registry.unregister_writer('obslog', Table)


class LogItem(object):
    """Class for observing log items
    """
    def __init__(self,**kwargs):
        for key in kwargs:
            value = kwargs[key]
            object.__setattr__(self,key,value)

class Log(object):
    """
    Class for observing log.

    Attributes:
        nchannels (int): Number of fiber channels.
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
    
    """
    def __init__(self):
        self.item_list = []

    def __iter__(self):
        return _LogIterator(self.item_list)

    def find_nchannels(self):
        """Find the number of channels by checking the column of "objectname".
        """
        self.nchannels = max([len(item.objectname) for item in self.item_list])

    def add_item(self, item):
        """Add a :class:`LogItem` instance into the observing log.

        Args:
            item (:class:`LogItem`): A :class:`LogItem` instance to be added
                into this observing log.
        
        """
        self.item_list.append(item)

    def sort(self, key):
        """Sort the items by the given keyword.

        Args:
            key (str): Keyword to sort.
        """

        new_item_list = sorted(self.item_list,
                               key=lambda item: getattr(item, key))
        self.item_list = new_item_list

    def get_frameid_lst(self, objectname=None, exptime=None, channel=None):
        """Get a list of frameids from given conditions.

        Args:
            objectname (str): Name of objects.
            exptime (float): Exposure time.
            channel (str): Name of channel.
        
        """
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
    """Iterator class for :class:`Log`.
    """
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

def read_logitem(string, names, types, column_separator='|', multi_object=False,
    object_separator=';'):
    """Read log items.

    If **multi_object** is *True*, the mulit-object mode is on, the "objectname"
    in each item is splitted into a list of names by the character given by
    **object_separator**.

    Args:
        string (str): Input string.
        names (list): A list of names.
        types (list): A list of type strings.
        column_separator (str): Separator of columns.
        multi_object (bool): If turn on the multi-object mode.
        object_separator (str): Separator of channels in "objectname"
            column.
    
    Returns:
        :class:`LogItem`: A :class:`LogItem` instance.

    """
    logitem = LogItem()

    g = string.split(column_separator)
    for i, value in enumerate(g):
        if i >= len(names):
            continue
        name = names[i]
        if types[i] == 'i':
            value = int(value)
        elif types[i] == 'f':
            value = float(value)
        else:
            value = value.strip()

        # parse object names for multi-channels
        if multi_object and name == 'objectname':
            value = [v.strip() for v in value.split(object_separator)]

        setattr(logitem, name, value)

    return logitem

def read_log(filename, multi_object=False):
    """Read observing log from an ascii file.

    Args:
        filename (str): Name of the observing log file.
        multi_object (bool): If turn on the multi-object mode.

    Returns:
        tuple: A tuple containing:
        
            * **log** (:class:`Log`) – An observing log instance.
            * **frame_lst** (*tuple*) – Frame list.
    """

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
            logitem = read_logitem(row, names, types, multi_object=multi_object)
            log.add_item(logitem)

    infile.close()

    if multi_object:
        log.find_nchannels()
    else:
        log.nchannels = 1

    logger.info('Observational log file "%s" loaded'%filename)

    return log

def get_input_fileids(log, string):
    """Get the fileids of the input.

    Args:
        log (:class:`Log`): A :class:`Log` instance.
        string (str): The input string.

    Returns:
        list: The list of file IDs.
    """

    if len(string.strip())==0:
        return []

    fileid_lst = {}
    for item in log:
        fileid_lst[item.frameid] = item.fileid

    lst = parse_num_seq(string)
    return [fileid_lst[n] for n in lst]

def parse_num_seq(string):
    """Convert the number sequence to list of numbers.

    Args:
        string (str): The input string to be parsed.

    Returns:
        list: A list of integer numbers.

    """
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

def find_log(path):
    """Find the log file in the given directory.

    The name of the observing log file should terminate in `.log`, but should
    not be `edrs.log`, which is used for the name of the running log of EDRS.

    Args:
        path (str): Searching directory.

    Returns:
        *str* or *None*: Name of the log file. Return *None* if not found or
            more than one file found.
    
    """
    filename_lst = [fname for fname in sorted(os.listdir(path))
                        if len(fname)>4 and fname[-4:]=='.log' and
                        fname != 'edrs.log']

    if len(filename_lst)==1:
        return os.path.join(path, filename_lst[0])
    elif len(filename_lst)==0:
        print('Error: Log file not found')
        return None
    else:
        print('Error: More than 1 log file found')
        return None

