import abc
from numpy import pi
from scipy.special import gamma

from .beam import Beam


class Beam3D(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__p_gauss = self.calculate_p_gauss()

        self._m = kwargs['m']

        if self._m == 0:
            if self._M == 0:
                self._distribution_type = 'gauss'
            else:
                self._distribution_type = 'ring'
        else:
            if self._M == 0:
                raise Exception('Gauss with vortex phase is a wrong initial mode!')
            self._distribution_type = 'vortex'

        if self._distribution_type == 'gauss':
            self.__p_0_to_p_gauss = kwargs['p_0_to_p_gauss']
            self._p_0 = self.__p_0_to_p_gauss * self.__p_gauss
        elif self._distribution_type == 'ring':
            self.__p_0_to_p_gauss = kwargs['p_0_to_p_gauss']
            self._p_0 = self.__p_0_to_p_gauss * self.__p_gauss
        elif self._distribution_type == 'vortex':
            self.__p_vortex = self.calculate_p_vortex()
            self.__p_0_to_p_vortex = kwargs['p_0_to_p_vortex']
            self._p_0 = self.__p_0_to_p_vortex * self.__p_vortex
        else:
            raise Exception('Wrong distribution type: "%s".' % self._distribution_type)

    @abc.abstractmethod
    def info(self):
        """Information about beam"""

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

    def calculate_p_gauss(self):
        return 3.77 * self._lmbda ** 2 / (8 * pi * self._medium.n_0 * self._medium.n_2)

    def calculate_p_vortex(self):
        return self.__p_gauss * 2**(2 * self._m + 1) * gamma(self._m + 1) * gamma(self._m + 2) / \
               (2 * gamma(2 * self._m + 1))
