import os
import sys
import time
import hashlib
import urllib.request

def get_file_md5(filename):
    """Get md5 string of file.

    Args:
        filename (str): Input filename.

    Returns:
        md5sum (str): md5 hash of the input file.
    """
    m = hashlib.md5()
    infile = open(filename, 'rb')
    m.update(infile.read())
    infile.close()
    return m.hexdigest()

def download_file(url, filename, show_progress=True):
    """Download file from the given url.

    Args:
        url (str):
        filename (str):
        show_progress (bool): Display a progress bar in the terminal if *True*.

    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    def get_human_readable_size(byte):
        unit = 'B'
        readable_size = byte
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'kB'
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'MB'
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'GB'
        return readable_size, unit

    def get_human_readable_time(second):
        readable_time = second
        if second < 60:
            return '{:02d}s'.format(int(second))

        minute = int(second/60)
        second = second - minute*60
        if minute < 60:
            return '{:d}m{:02d}s'.format(minute, int(second))

        hour = int(minute/60)
        minute = minute - hour*60
        if hour < 24:
            return '{:d}h{:02d}m'.format(hour, int(minute))

        day = int(hour/24)
        hour = hour - day*24
        return '{:d}d{:02d}h'.format(day, int(hour))

    def callback(block_num, block_size, total_size):
        t1 = time.time()
        # downloaded size in byte
        down_size = block_num * block_size
        speed = (down_size - param[1])/(t1-param[0])
        if speed > 0:
            # ETA in seconds
            eta = (total_size - down_size)/speed
            eta = get_human_readable_time(eta)
        else:
            eta = '--'
        ratio = min(down_size/total_size, 1.0)
        percent = min(ratio*100., 100)

        disp_speed, unit_speed = get_human_readable_size(speed)
        disp_size,  unit_size  = get_human_readable_size(total_size)

        # get the width of the terminal
        term_size = os.get_terminal_size()

        str1 = 'Downloading {}'.format(os.path.basename(filename))
        str2 = '{:6.2f}% of {:6.1f} {}'.format(percent, disp_size, unit_size)
        str3 = '{:6.1f} {}/s'.format(disp_speed, unit_speed)
        str4 = 'ETA: {}'.format(eta)

        n = term_size.columns-len(str1)-len(str2)-len(str3)-len(str4)-20
        progressbar = '>'*int(ratio*n)
        progressbar = progressbar.ljust(n, '-')

        msg = '\r {} |{}| {} {} {}'.format(
                str1, progressbar, str2, str3, str4)
        sys.stdout.write(msg)
        sys.stdout.flush()

    param = [time.time(), 0]
    if show_progress:
        urllib.request.urlretrieve(url, filename, callback)
        # use light green color
        print('\033[92m Completed\033[0m')
    else:
        urllib.request.urlretrieve(url, filename)


def get_gamse_cloud_url():
    """Get the url of gamse cloud.

    """
    if time.timezone==-28800:
        # timezone is UTC+8, probably in China
        # return Aliyun url
        return 'https://gamse.oss-cn-beijing.aliyuncs.com/'
    else:
        # for the rest of the world, return Amazon AWS url
        return 'https://gamse.s3.eu-central-1.amazonaws.com/'


def get_file(filepath, md5, show_progress=True):
    """Get file from either local cache or remote server.

    Args:
        filepath (str): The relative path of required file.
        md5 (str): MD5 hash of the required file.
        show_progress (bool): Display a progress bar in the terminal if *True*.

    """
    cache_path = os.path.join(os.path.expanduser('~'), '.gamse')
    filename = os.path.join(cache_path, filepath)

    gamse_url = get_gamse_cloud_url()
    url = os.path.join(gamse_url, filepath)

    if os.path.exists(filename):
        if get_file_md5(filename)==md5:
            return filename
        else:
            # file need to be updated
            download_file(url, filename, show_progress=show_progress)
            if get_file_md5(filename)==md5:
                return filename
            else:
                # the file downloaded from the remote server does not match
                # the one saved in the index file
                print('Error: Md5 does not match: {}, {}'.formt(url, filename))
                return None
    else:
        download_file(url, filename, show_progress=show_progress)
        if get_file_md5(filename)==md5:
            return filename
        else:
            # the file downloaded from the remote server does not match
            # the one saved in the index file
            print('Error: Md5 does not match: {}, {}'.formt(url, filename))
            return None
