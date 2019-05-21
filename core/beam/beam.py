import abc
from numpy import pi
from scipy.special import gamma

from core.medium import Medium
from core.m_constants import M_Constants


class Beam(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.__m_constants = M_Constants()
        self._lmbda = kwargs['lmbda']

        self._medium = Medium(name=kwargs['medium'],
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self.__p_G = self.calculate_p_G()

        self._M = kwargs['M']
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
            self.__p_0_to_p_G = kwargs['p_0_to_p_G']
            self._p_0 = self.__p_0_to_p_G * self.__p_G
        elif self._distribution_type == 'ring':
            self.__p_0_to_p_G = kwargs['p_0_to_p_G']
            self._p_0 = self.__p_0_to_p_G * self.__p_G
        elif self._distribution_type == 'vortex':
            self.__p_V = self.calculate_p_V()
            self.__p_0_to_p_V = kwargs['p_0_to_p_V']
            self._p_0 = self.__p_0_to_p_V * self.__p_V
        else:
            raise Exception('Wrong distribution type: "%s".' % self._distribution_type)

        self._field, self._intensity, self._i_max, self._i_0, self._z_diff = None, None, None, None, None

    @abc.abstractmethod
    def info(self):
        """Information about beam"""

    @property
    def medium(self):
        return self._medium

    @property
    def lmbda(self):
        return self._lmbda

    @property
    def distribution_type(self):
        return self._distribution_type

    @property
    def m(self):
        return self._m

    @property
    def M(self):
        return self._M

    @property
    def i_0(self):
        return self._i_0

    @property
    def i_max(self):
        return self._i_max

    @property
    def intensity(self):
        return self._intensity

    @property
    def p_0_to_p_G(self):
        return self.__p_0_to_p_G

    @property
    def p_0_to_p_V(self):
        return self.__p_0_to_p_V

    @property
    def p_0(self):
        return self._p_0

    @property
    def z_diff(self):
        return self._z_diff

    def calculate_p_G(self):
        return 3.77 * self._lmbda ** 2 / (8 * pi * self._medium.n_0 * self._medium.n_2)

    def calculate_p_V(self):
        return self.__p_G * 2**(2 * self._m + 1) * gamma(self._m + 1) * gamma(self._m + 2) / (2 * gamma(2 * self._m + 1))
