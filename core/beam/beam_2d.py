from abc import abstractmethod

from .beam import Beam


class Beam2D(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def info(self):
        """Information about beam"""
