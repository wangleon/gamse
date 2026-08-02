import re
from pathlib import Path
import numpy as np
import astropy.io.fits as fits

from ..base import DataFrame

def get_rawdata_mask(data, head):
    sat_mask = data >= 65535
    mask = np.int16(sat_mask)*4
    return mask

class ESPADONSFrame(DataFrame):

    def __init__(self, data: np.ndarray, head: fits.Header, mask=None, info=[],
                 is_raw=False):
        self.data = data
        if mask is None:
            mask = np.zeros_like(data, dtype=np.int16)
        self.mask = mask
        self.head = head
        self.info = info

        if is_raw:
            self.extract_info_remove_cards()

    def extract_info_remove_cards(self):
        new_cards = []

        self.info = []        
        for card in self.head.cards:
            keyword = card.keyword
            value   = card.value
            comment = card.comment

            # extract GAMSE key and put them into self.info
            if (mobj := re.match(r'^HIERARCH GAMSE (\s\S)*', keyword)):
                cardname    = mobj.group(1).strip()
                cardvalue   = value
                cardcomment = comment
                # append into self.info
                self.info.append((cardname, cardvalue, cardcomment))
                continue

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

    @classmethod
    def read(cls, filepath):
        hdulst = fits.open(filepath)
        # generate mask for raw image and other images
        if hdulst[0].data is None and len(hdulst)==2 \
            and hdulst[1].data.dtype==np.uint16:
            # the input file is a raw image
            data = hdulst[1].data
            head = hdulst[1].header
            mask = get_rawdata_mask(data, head)
            is_raw = True
        else:
            # first HDU is image, second HDU is mask
            data = hdulst[0].data
            head = hdulst[0].header
            if len(hdulst)>1:
                mask = hdulst[1].data
            else:
                mask = np.zeros_like(data, dtype=np.int16)
            is_raw = False
        hdulst.close()
        return cls(data=data, head=head, mask=mask, is_raw=is_raw)

    def save(self, filename, overwrite=False):
        head = self.head.copy()
        if len(self.info)>0:
            for key, value in self.info:
                head.append(('HIERARCH GAMSE '+key, value))

        hdulst = fits.HDUList([
                    fits.PrimaryHDU(header=head, data=self.data),
                    fits.ImageHDU(data=self.mask),
                ])
        filepath = Path(filename).resolve()
        if filepath.exists() and not overwrite:
            print('Error: {} exists. use overwrite=True'.format(filepath))
        else:
            hdulst.writeto(filepath, overwrite=True)

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
