import re
from pathlib import Path
import numpy as np
import astropy.io.fits as fits

from ..base import DataFrame, ImageFrame, SpectrumFrame

def read_dataframe(filepath, logitem=None):

    FRAME_CLASSES = (
            RawImageFrame,
            ImageFrame,
            SpectrumFrame,
            )

    with fits.open(filepath) as hdulst:
        for cls in FRAME_CLASSES:
            if cls.match(hdulst):
                obj = cls.from_hdulst(hdulst)

                # append logitem into extra_head
                if logitem is not None:
                    for col in logitem.colnames:
                        value = logitem[col]
                        key = 'HIERARCH LOGINFO ' + col.upper()
                        obj.extra_head.append((key, value))
                return obj

    raise TypeError('Unknown FITS format')

class RawImageFrame(ImageFrame):

    @classmethod
    def match(cls, hdulst):
        return (len(hdulst)==1
                and hdulst[0].data.ndim == 2
                and hdulst[0].data.dtype == np.uint16
                )

    @classmethod
    def from_hdulst(cls, hdulst):
        data = hdulst[0].data
        head = hdulst[0].header
        mask = cls._make_raw_mask(data)
        return cls(data, head, mask=mask)

    @staticmethod
    def _make_raw_mask(data):
        sat_mask = data >= 65535
        mask = np.int16(sat_mask)*4
        return mask

    def print_to_console(self):

        fileid  = self.extra_head['LOGINFO FILEID']
        obstype = self.extra_head['LOGINFO OBSTYPE']
        objname = self.extra_head['LOGINFO OBJECT']
        exptime = self.extra_head['LOGINFO EXPTIME']

        print(
                f'* -'
                f'  FILEID: {fileid}'
                f'  OBSTYPE: {obstype}'
                f'  OBJECT: {objname}'
                f'  EXPTIME: {exptime}'
                )
