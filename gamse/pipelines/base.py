from typing import List, Tuple, Any
from abc import ABC, abstractmethod

class Instrument(ABC):

    name = None

    @abstractmethod
    def read(self, filename):
        ...


class DataFrame(ABC):

    cards: List[Tuple[str, Any]] = []

    @abstractmethod
    def save(self, filepath):
        pass

    @classmethod
    @abstractmethod
    def read(cls, filepath):
        pass

