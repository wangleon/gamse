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
                return obj

    raise TypeError('Unknown FITS format')


class RawImageFrame(ImageFrame):

    def __init__(self, data: np.ndarray, head: fits.Header, mask=None):

        super().__init__(data, head, mask=mask)

        self._clean_header()

    def _clean_header(self):
        new_cards = []
        new_info = []        
        for card in self.head.cards:
            keyword = card.keyword
            value   = card.value
            comment = card.comment

            ## extract GAMSE key and put them into self.info
            #if (mobj := re.match(r'^HIERARCH GAMSE (\s\S)*', keyword)):
            #    cardname    = mobj.group(1).strip()
            #    cardvalue   = value
            #    cardcomment = comment
            #    # append into info
            #    new_info.append((cardname, cardvalue, cardcomment))
            #    continue

            # remove COMMENT cards with "Reseved space."
            if keyword == 'COMMENT' and value.startswith(' Reserved space.'):
                # do nothing, which means remove them from new header
                continue

            # for other cards, append them into the new header
            new_cards.append(card)

        # clear the previous head
        self.head.clear()
        # append new cards into self.head
        for card in new_cards:
            self.head.append(card, end=True)

        # append new info
        for item in new_info:
            self.extra_head.append(item)


    @classmethod
    def match(cls, hdulst):
        return (hdulst[0].data is None
                and len(hdulst)==2
                and hdulst[1].data.ndim == 2
                and hdulst[1].data.dtype == np.uint16
                )

    @classmethod
    def from_hdulst(cls, hdulst):
        data = hdulst[1].data
        head = hdulst[1].header
        mask = cls._make_raw_mask(data)
        return cls(data, head, mask=mask)

    @staticmethod
    def _make_raw_mask(data):
        sat_mask = data >= 65535
        mask = np.int16(sat_mask)*4
        return mask

    def print_to_console(self):

        # determine the color by obstype
        obstype = self.head['OBSTYPE']

        if obstype == 'BIAS':
            # bias images, use dim (2)
            color = '\033[2m'
        elif obstype == 'OBJEC':
            # sci images, use highlights (1)
            color = '\033[1m'
        elif obstype == 'COMPARISON':
            # arc lamp, use light yellow (93)
            color = '\033[93m'
        else:
            color = ''

        print(
                f'{color}'
                f'* -'
                f'  FILEID: {self.head["FILENAME"]:>8s}'
                f'  OBSTYPE: {self.head["OBSTYPE"]:<10s}'
                f'  EXPTIME: {self.head["EXPTIME"]:6.1f}s'
                f'  INSTMODE: {self.head["INSTMODE"]:<30s}'
                '\033[0m'
                )

