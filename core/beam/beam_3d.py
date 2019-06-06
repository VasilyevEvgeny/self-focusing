from abc import abstractmethod

from .beam import Beam
from core.functions import calculate_p_gauss, calculate_p_vortex


class Beam3D(Beam):
    """Subclass for 3-dimensional beams"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # critical power of Gaussian beam self-focusing
        self.__p_gauss = calculate_p_gauss(self._lmbda, self._medium.n_0, self._medium.n_2)

        self._m = kwargs['m']  # topological charge

        # distribution type determination
        if self._m == 0:
            if self._M == 0:
                self._distribution_type = 'gauss'
            else:
                self._distribution_type = 'ring'
        else:
            if self._M == 0:
                raise Exception('Gauss with vortex phase is a wrong initial mode!')
            self._distribution_type = 'vortex'

        # calculation of the initial power depending on the excess over the critical power
        if self._distribution_type == 'gauss':
            self.__p_0_to_p_gauss = kwargs.get('p_0_to_p_gauss', 1.0)
            self._p_0 = self.__p_0_to_p_gauss * self.__p_gauss
        elif self._distribution_type == 'ring':
            self.__p_0_to_p_gauss = kwargs.get('p_0_to_p_gauss', 1.0)
            self._p_0 = self.__p_0_to_p_gauss * self.__p_gauss
        elif self._distribution_type == 'vortex':
            self.__p_vortex = calculate_p_vortex(self._m, self.__p_gauss)
            self.__p_0_to_p_vortex = kwargs.get('p_0_to_p_vortex', 1.0)
            self._p_0 = self.__p_0_to_p_vortex * self.__p_vortex

    @abstractmethod
    def info(self):
        """Beam type"""

    @property
    def m(self):
        return self._m

    @property
    def p_0_to_p_gauss(self):
        return self.__p_0_to_p_gauss

    @property
    def p_0_to_p_vortex(self):
        return self.__p_0_to_p_vortex

    @property
    def p_0(self):
        return self._p_0
