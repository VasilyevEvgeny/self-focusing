from abc import abstractmethod

from .beam import Beam


class Beam2D(Beam):
    """
    Subclass for 2-dimensional beams
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._half = None  # flag to use only half of the distribution

        # distribution type determination
        if self._M == 0:
            self._distribution_type = 'gauss'
            self._half = False
        elif self._M > 0:
            self._half = kwargs['half']
            if self._half:
                self._distribution_type = 'half of ring'
            else:
                self._distribution_type = 'ring'
        else:
            Exception('Wrong M!')

    @abstractmethod
    def info(self):
        """Beam type"""
