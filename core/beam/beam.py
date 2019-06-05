from abc import ABCMeta, abstractmethod

from core.medium import Medium
from core.m_constants import MathConstants


class Beam(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.__m_constants = MathConstants()
        self._lmbda = kwargs['lmbda']

        self._medium = Medium(name=kwargs['medium'],
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._M = kwargs['M']

        self._distribution_type, self._field, self._intensity, self._i_max, self._i_0, self._z_diff, self._r_kerr = \
            None, None, None, None, None, None, None

    @abstractmethod
    def info(self):
        """Information about beam"""

    @abstractmethod
    def update_intensity(self):
        """update intensity"""

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
    def z_diff(self):
        return self._z_diff

    @property
    def r_kerr(self):
        return self._r_kerr
