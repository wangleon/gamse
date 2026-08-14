import re
from pathlib import Path
#from typing import List, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
import astropy.io.fits as fits

class Instrument(ABC):

    name = None

    @abstractmethod
    def read(self, filename):
        ...


class DataFrame(ABC):

    #info: List[Tuple[str, Any]] = []

    @abstractmethod
    def save(self, filepath):
        pass

    def _extract_head(self):
        """Extract extra_head from header and put them into the self.extra_head.

        """
        
        new_cards = []      # new_cards is for the self.head after shuffling
        extra_cards = []    # extra_cards is for the self.extra_head
        for card in self.head.cards:
            keyword = card.keyword
            value   = card.value
            comment = card.comment

            # extract GAMSE key and put them into self.extra_head
            if (mobj := re.match(r'^HIERARCH REDUCTION (\s\S*)', keyword)):
                #cardname    = mobj.group(1).strip()
                cardname    = keyword
                cardvalue   = value
                cardcomment = comment
                # append into extra_cards
                extra_cards.append((cardname, cardvalue, cardcomment))
                continue

            # for other cards, append them into the new header
            new_cards.append(card)

        # clear the previous head
        self.head.clear()

        # append new cards into self.head
        for card in new_cards:
            self.head.append(card, end=True)
            # put them into the end. if end=False all COMMENT cards will be put
            # into the end

        # append extra_cards into self.extra_head
        for card in extra_cards:
            self.extra_head.append(card, end=True)

class ImageFrame(DataFrame):

    def __init__(self,
                 data: np.ndarray,
                 head: fits.Header,
                 mask = None,
                 extra_head = None,
                 ):

        self.data = data

        if mask is None:
            mask = np.zeros_like(data, dtype=np.int16)

        self.mask = mask
        self.head = head

        if extra_head is None:
            self.extra_head = fits.Header()
        else:
            self.extra_head = extra_head

        self._extract_head()

    @classmethod
    def match(cls, hdulst):
        return(len(hdulst)==2
               and hdulst[0].data.ndim==2
               and hdulst[1].data.ndim==2
               and hdulst[0].data.dtype.names is None
               and hdulst[1].data.dtype.names is None
               )

    @classmethod
    def from_hdulst(cls, hdulst):
        data = hdulst[0].data
        head = hdulst[0].header
        mask = hdulst[1].data
        return cls(data, head, mask)

    def save(self, filename, overwrite=False):

        head = self.head.copy()

        if len(self.extra_head)>0:
            for card in self.extra_head.cards:
                head.append(card, end=True)

        hdulst = fits.HDUList([
                    fits.PrimaryHDU(header=head, data=self.data),
                    fits.ImageHDU(data=self.mask),
                ])
        filepath = Path(filename).resolve()
        if filepath.exists() and not overwrite:
            print('Error: {} exists. use overwrite=True'.format(filepath))
        else:
            hdulst.writeto(filepath, overwrite=True)

class SpectrumFrame(DataFrame):

    def __init__(self, data, head, extra_head=None, ident_lst=None):
        self.data = data
        self.head = head


        if extra_head is None:
            self.extra_head = fits.Header()
        else:
            self.extra_head = extra_head

        self.ident_lst = ident_lst

        self._extract_head()

    @classmethod
    def match(cls, hdulst):
        return(len(hdulst)>=2
               # no data in the primary HDU
               and hdulst[0].data is None
               # spectral table in the second HDU
               and hdulst[1].data.ndim == 1
               and hdulst[1].data.dtype.names is not None
               )

    @classmethod
    def from_hdulst(cls, hdulst):
        head = hdulst[0].header
        data = hdulst[1].data
        if len(hdu_lst)>2:
            ident_lst = hdulst[2].data
        return cls(data, head, ident_lst=ident_lst)

    def save(self, filepath, overwrite=False):

        head = self.head.copy()

        if len(self.extra_head)>0:
            for card in self.extra_head.cards:
                head.append(card, end=True)

        hdulst = fits.HDUList([
                    fits.PrimaryHDU(header=head),
                    fits.BinTableHDU(data=self.data),
                    ])
        # add ident list in the third HDU
        if self.ident_lst is not None:
            hdulst.append(fits.BinTableHDU(data=self.ident_lst))

        filepath = Path(filepath).resolve()
        if filepath.exists() and not overwrite:
            print('Error: {} exists. use overwrite=True'.format(filepath))
        else:
            hdulst.writeto(filepath, overwrite=True)
